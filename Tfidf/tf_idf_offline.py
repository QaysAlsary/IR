from fastapi import FastAPI, HTTPException, Request
from typing import List
import pathlib
from Tf_idf_Service import TfidfService
from fastapi import FastAPI, HTTPException, Request
import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path


app = FastAPI()
tfidf_service = TfidfService()


@app.post("/process_json/")
async def process_json_file(request: Request):
    try:
        data = await request.json()
        input_file_path = Path.cwd() / data.get("input_file_path")
        tfidf_matrix_path =  Path.cwd() / data.get("tfidf_matrix_path")
        vectorizer_path =  Path.cwd() / data.get("vectorizer_path")

        if not pathlib.Path(input_file_path).is_file():
            raise HTTPException(status_code=400, detail="Input file does not exist")
        
        tfidf_service.process_json_file(input_file_path, tfidf_matrix_path, vectorizer_path)
        return {"message": "TF-IDF matrix and vectorizer saved successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)
