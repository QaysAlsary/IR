from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pathlib
import json
import pickle
import nltk
from text_processing.TextProcessor import TextProcessor
from Tfidf.Tf_idf_Service import TfidfService
from Ranking.ranking_service import RankingService

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from any origin (you might want to restrict this in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tfidf_service = TfidfService()
textprocessor = TextProcessor()
rankingservice = RankingService()

# Load the saved vectorizer, TF-IDF matrix, and corpus
cwd = pathlib.Path().cwd()
lifestyle_folder = cwd / "files" / "lifestyle/"
clinical_folder = cwd / "files" / "clinicaltrials/"    
tfidf_service.preload(lifestyle_folder, clinical_folder) 
def load_data(dataset: str):
    if dataset == "clinicaltrials":
        tfidf_matrix_path = cwd / 'files' / 'clinicaltrials' / 'tfidf_matrix.pkl'
        corpus_key_path = cwd / 'files' / 'clinicaltrials' / 'corpuskey.json'
        model_path = cwd / "files" / 'clinicaltrials' / "kmeans_model.pkl"
        inverted_index_path = cwd / "files" / 'clinicaltrials' / "inverted_index_kmeans.json"
    elif dataset == "lifestyle":
        tfidf_matrix_path = cwd / 'files' / 'lifestyle' / 'tfidf_matrix.pkl'
        corpus_key_path = cwd / 'files' / 'lifestyle' / 'corpuskey.json'
        model_path = cwd / "files" / 'lifestyle' / "kmeans_model.pkl"
        inverted_index_path = cwd / "files" / 'lifestyle' / "inverted_index_kmeans.json"
    else:
        raise HTTPException(status_code=400, detail="Invalid dataset specified.")

    with open(corpus_key_path, 'r') as f:
        corpus_keys = json.load(f)

    with open(model_path, 'rb') as file:
        kmeans_model = pickle.load(file)

    with open(inverted_index_path, 'r') as file:
        inverted_index = json.load(file)

    return corpus_keys, kmeans_model, inverted_index

def docsIdsSearchWithKmeans(query_text, corpus_keys, dataset, kmeans_model, inverted_index):
    processed_query = textprocessor.process_text(query_text)
    query_vector = tfidf_service.vectorize_query_evaluation(processed_query, dataset)
    # Predict the cluster for the query vector
    cluster_label = kmeans_model.predict(query_vector)[0]
    # Retrieve documents in the predicted cluster
    cluster_docs = inverted_index.get(str(cluster_label), [])
    if not cluster_docs:
        return []  # No documents found in the cluster
    # Calculate similarity scores for documents in the selected cluster
    cluster_docs_vectors = tfidf_service.get_docs_vectors(cluster_docs, dataset)
    similarity_scores = cosine_similarity(query_vector, cluster_docs_vectors).flatten()
    # Rank documents based on similarity scores
    ranked_docs = rankingservice.rank_and_sort(similarity_scores)
    return [corpus_keys[doc_id] for doc_id, _ in ranked_docs[:10]]

@app.post("/docsIdsSearchWithKmeans")
async def search_documents(data: dict):
    try:
        query = data.get("query")
        dataset = data.get("dataset")
        corpus_keys, kmeans_model, inverted_index = load_data(dataset)
        doc_ids = docsIdsSearchWithKmeans(query, corpus_keys, dataset, kmeans_model, inverted_index)
        return doc_ids
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8009)
