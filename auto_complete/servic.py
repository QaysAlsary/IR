from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from textblob import TextBlob
from transformers import pipeline, AutoModelForMaskedLM, AutoTokenizer
from pydantic import BaseModel
import json
import os
import pathlib
import uvicorn
import xml.etree.ElementTree as ET
from fastapi.responses import JSONResponse

app = FastAPI()

# Define base paths for lifestyle and clinical datasets
LIFESTYLE_BASE_PATH = pathlib.Path.cwd() / "files" / 'lifestyle'
CLINICAL_BASE_PATH = pathlib.Path.cwd() / "files" / 'clinicaltrials'
CLINICAL_DATA_BASE_PATH = "C:/Users/ASUS/Desktop/5th year/فصل تاني/IR/dataset/ClinicalTrials"

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the query expansion model
model_directory = pathlib.Path.cwd() / "auto_complete" / 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_directory)
model = AutoModelForMaskedLM.from_pretrained(model_directory)
query_expansion_pipeline = pipeline("fill-mask", model=model, tokenizer=tokenizer)

def load_json_file(path):
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return {}

def get_paths_for_dataset(dataset):
    if dataset == "lifestyle":
        base_path = LIFESTYLE_BASE_PATH
    elif dataset == "clinicaltrials":
        base_path = CLINICAL_BASE_PATH
    else:
        raise HTTPException(status_code=400, detail="Invalid dataset specified.")
    
    return {
        "autocomplete_dict_path": base_path / "autocomplete_dict.json",
        "query_frequencies_path": base_path / "query_frequencies.json",
        "user_queries_log_path": base_path / "user_queries_log.json"
    }

def get_autocomplete_suggestions_for_term(term, autocomplete_dict, limit=5):
    if not term:
        return []
    suggestions = [k for k in autocomplete_dict.keys() if k.startswith(term)]
    suggestions.sort(key=lambda x: autocomplete_dict[x], reverse=True)
    return suggestions[:limit]

def get_autocomplete_suggestions(query, autocomplete_dict, limit=5):
    query_terms = query.split()
    if not query_terms:
        return []

    last_term = query_terms[-1]
    term_suggestions = get_autocomplete_suggestions_for_term(last_term, autocomplete_dict, limit)
    suggestions = [' '.join(query_terms[:-1] + [suggestion]) for suggestion in term_suggestions]

    seen = set()
    unique_suggestions = []
    for suggestion in suggestions:
        if suggestion not in seen:
            unique_suggestions.append(suggestion)
            seen.add(suggestion)
    return unique_suggestions[:limit]

def get_alternative_suggestions(query, query_frequencies, limit=5):
    query_words = set(query.lower().split())
    matching_queries = [q for q in query_frequencies.keys()  if any(word in query_words for word in q.lower().split())]
    matching_queries.sort(key=lambda q: query_frequencies[q], reverse=True)
    return matching_queries[:limit]

@app.get("/refine_query")
def refine_query(dataset: str, query: str = Query(..., min_length=1)):
    paths = get_paths_for_dataset(dataset)
    autocomplete_dict = load_json_file(paths["autocomplete_dict_path"])
    query_frequencies = load_json_file(paths["query_frequencies_path"])

    autocomplete_suggestions = get_autocomplete_suggestions(query, autocomplete_dict, limit=2)
    blob = TextBlob(query)
    corrected_query = str(blob.correct())
    alternative_suggestions = get_alternative_suggestions(query, query_frequencies, limit=2)

    expanded_terms = []
    if query:
        masked_input = f"{query} [MASK]"
        predictions = query_expansion_pipeline(masked_input)
        expanded_terms = [pred['token_str'] for pred in predictions if pred['token_str'] not in query.split()][:2]

    result = {
        "corrected_query": corrected_query,
        "autocomplete_suggestions": autocomplete_suggestions,
        "alternative_suggestions": alternative_suggestions,
        "expanded_terms": expanded_terms
    }

    return result

# Load the lifestyle dataset
LIFESTYLE_DATA_PATH = LIFESTYLE_BASE_PATH / "corpus.json"
with open(LIFESTYLE_DATA_PATH, 'rb') as f:
    lifestyle_data = json.load(f)

@app.get("/get_document")
async def get_document(dataset: str, doc_id: str):
    if dataset == "lifestyle":
        document = lifestyle_data.get(doc_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found.")
        return JSONResponse(content={"content": document}, media_type="application/json")

    elif dataset == "clinicaltrials":
        # Traverse through the clinicaltrials directory structure
        for root, _, files in os.walk(CLINICAL_DATA_BASE_PATH):
            if f"{doc_id}.xml" in files:
                xml_file_path = os.path.join(root, f"{doc_id}.xml")
                tree = ET.parse(xml_file_path)
                root_element = tree.getroot()
                xml_str = ET.tostring(root_element, encoding='unicode', method='xml')
                return JSONResponse(content={"xml_content": xml_str}, media_type="application/json")

        raise HTTPException(status_code=404, detail="Document not found.")
    else:
        raise HTTPException(status_code=400, detail="Invalid dataset specified.")

@app.get("/user_queries")
def get_user_queries(dataset: str):
    paths = get_paths_for_dataset(dataset)
    user_queries_log = load_json_file(paths["user_queries_log_path"])
    return user_queries_log

class Feedback(BaseModel):
    query: str
    doc_id: str
    relevance: bool

@app.post("/log_feedback")
async def log_feedback(dataset: str, feedback: Feedback):
    paths = get_paths_for_dataset(dataset)
    autocomplete_dict = load_json_file(paths["autocomplete_dict_path"])
    query_frequencies = load_json_file(paths["query_frequencies_path"])

    query = feedback.query
    relevance = feedback.relevance  # Boolean indicating if the document was relevant

    if relevance:
        # Split the query into individual terms
        query_terms = query.split()
        for term in query_terms:
            autocomplete_dict[term] = autocomplete_dict.get(term, 0) + 1
        
        query_frequencies[query] = query_frequencies.get(query, 0) + 1

        with open(paths["autocomplete_dict_path"], 'w') as f:
            json.dump(autocomplete_dict, f)
        with open(paths["query_frequencies_path"], 'w') as f:
            json.dump(query_frequencies, f)

    return {"status": "success"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8007)
