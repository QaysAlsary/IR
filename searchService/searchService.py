from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional
import requests
from fastapi.middleware.cors import CORSMiddleware
import pathlib
import json

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now, you can restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

QUERY_PROCESSING_URL = "http://127.0.0.1:8000/process_text/"
VECTORIZE_QUERY_URL = "http://localhost:8001/vectorize_query"
CALCULATE_SIMILARITY_URL = "http://localhost:8001/calculate_similarity"
RANK_DOCUMENTS_URL = "http://localhost:8002/rank_documents"

# Path to save user queries log
log_dir_path = pathlib.Path.cwd() / "files"

def get_log_path(dataset: str) -> pathlib.Path:
    dataset_log_path = log_dir_path / dataset / "user_queries_log.json"
    return dataset_log_path

class QueryRequest(BaseModel):
    query: str
    dataset: str
    dataset_type: Optional[str] = "generic"

@app.post("/docs_ids_search")
async def docs_ids_search(request: Request):
    try:
        request_body = await request.json()
        query = request_body.get("query")
        dataset_type = request_body.get("dataset_type", "generic")
        dataset = request_body.get("dataset")

        # Validate input
        if not query or not dataset:
            raise HTTPException(status_code=400, detail="Query and dataset must be provided.")
        if dataset_type not in ["generic", "specific"]:
            raise HTTPException(status_code=400, detail="Invalid dataset type.")

 
        # Determine the correct log path based on the dataset
        user_queries_log_path = get_log_path(dataset)

          # Load existing user queries log if available
        if user_queries_log_path.exists():
            with open(user_queries_log_path, 'r') as f:
                user_queries_log = json.load(f)
        else:
            user_queries_log = []

        # Log the user's query
        user_queries_log.append(query)
   # Save the updated user queries log to file
        user_queries_log_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
        with open(user_queries_log_path, 'w') as f:
            json.dump(user_queries_log, f)


        # Step 1: Process the text
        response = requests.post(QUERY_PROCESSING_URL, json={"query": query, "dataset_type": dataset_type})
        response.raise_for_status()
        processed_tokens = response.json().get('processed_tokens')

        if not processed_tokens:
            raise HTTPException(status_code=500, detail="Processed tokens not returned from text processing service.")

        # Step 2: Vectorize the query
        response = requests.post(VECTORIZE_QUERY_URL, json={"query_tokens": processed_tokens, "dataset": dataset})
        response.raise_for_status()
        query_vector = response.json().get('query_vector')

        if not query_vector:
            raise HTTPException(status_code=500, detail="Query vector not returned from vectorization service.")

        # Step 3: Calculate similarity
        response = requests.post(CALCULATE_SIMILARITY_URL, json={"query_vector": query_vector, "dataset": dataset})
        response.raise_for_status()
        similarities = response.json().get('similarities')

        if similarities is None:
            raise HTTPException(status_code=500, detail="Similarities not returned from similarity calculation service.")

        # Step 4: Rank documents and retrieve IDs
        response = requests.post(RANK_DOCUMENTS_URL, json={"similarities": similarities, "dataset": dataset})
        response.raise_for_status()
        result_docs = response.json().get('result_docs')

        if not result_docs:
            raise HTTPException(status_code=500, detail="Result documents not returned from ranking service.")

    except requests.HTTPError as http_err:
        raise HTTPException(status_code=response.status_code, detail=str(http_err))
    except requests.RequestException as req_err:
        raise HTTPException(status_code=500, detail=str(req_err))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return { "result_docs": result_docs }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
