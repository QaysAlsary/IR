from fastapi import FastAPI, HTTPException, Request
from typing import List
import pathlib
from Tf_idf_Service import TfidfService
from fastapi import FastAPI, HTTPException, Request
import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity


app = FastAPI()
tfidf_service = TfidfService()

@app.on_event("startup")
def load_data():
    lifestyle_folder = pathlib.Path("C:/Users/ASUS/Desktop/5th year/فصل تاني/project/files/lifestyle/")
    clinical_folder = pathlib.Path("C:/Users/ASUS/Desktop/5th year/فصل تاني/project/files/clinicaltrials/")
    tfidf_service.preload(lifestyle_folder, clinical_folder)

@app.post("/process_json/")
async def process_json_file(request: Request):
    try:
        data = await request.json()
        input_file_path = data.get("input_file_path")
        tfidf_matrix_path = data.get("tfidf_matrix_path")
        vectorizer_path = data.get("vectorizer_path")

        if not pathlib.Path(input_file_path).is_file():
            raise HTTPException(status_code=400, detail="Input file does not exist")
        
        tfidf_service.process_json_file(input_file_path, tfidf_matrix_path, vectorizer_path)
        return {"message": "TF-IDF matrix and vectorizer saved successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vectorize_query/")
async def vectorize_query(request: Request):
    try:
        data = await request.json()
        query_tokens = data.get("query_tokens")
        dataset = data.get("dataset")
     
        query_vector = tfidf_service.vectorize_query(query_tokens, dataset)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return {"query_vector": query_vector}

@app.post("/calculate_similarity/")
async def calculate_similarity(request: Request):
    try:
        data = await request.json()
        query_vector_data = data.get("query_vector")
        dataset = data.get("dataset")

        if not query_vector_data:
            raise HTTPException(status_code=400, detail="Query vector data is missing")

        # Reconstruct the sparse matrix
        query_vector_sparse = sp.csr_matrix(
            (query_vector_data["data"], query_vector_data["indices"], query_vector_data["indptr"]),
            shape=query_vector_data["shape"]
        )
        
        similarities = tfidf_service.calculate_similarity_api(query_vector_sparse, dataset)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return {"similarities": similarities}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
