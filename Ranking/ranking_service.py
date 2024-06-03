
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import json

app = FastAPI()

class RankingService:
    def __init__(self):
        self.lifestyle_corpus = {}
        self.clinicaltrials_corpus = {}

    def load_lifestyle_corpus(self, file_path):
        with open(file_path, "r") as f:
            self.lifestyle_corpus = json.load(f)

    def load_clinicaltrials_corpus(self, file_path):
        with open(file_path, "r") as f:
            self.clinicaltrials_corpus = json.load(f)

    def rank_and_sort(self, similarity_scores):
        ranks = {i: similarity_scores[i] for i in range(len(similarity_scores))}
        sorted_ranks = sorted(ranks.items(), key=lambda x: x[1], reverse=True)
        # Filter out documents with a similarity score of 0.0
        filtered_sorted_ranks = [(idx, score) for idx, score in sorted_ranks]
        return filtered_sorted_ranks
ranking_service = RankingService()

class RankRequest(BaseModel):
    similarities: List[float]
    dataset: str

@app.on_event("startup")
async def startup_event():
    # Load the corpora separately
    ranking_service.load_lifestyle_corpus("files/lifestyle/corpuskey.json")
    ranking_service.load_clinicaltrials_corpus("files/clinicaltrials/corpuskey.json")

@app.post("/rank_documents")
async def rank_documents(request: RankRequest):
    if request.dataset == "lifestyle":
        corpus = ranking_service.lifestyle_corpus
    elif request.dataset == "clinicaltrials":
        corpus = ranking_service.clinicaltrials_corpus
    else:
        raise HTTPException(status_code=400, detail=f"Dataset {request.dataset} not found")

    sorted_ranks = ranking_service.rank_and_sort(request.similarities)
    result_docs = [corpus[doc_idx] for doc_idx, _ in sorted_ranks]
    
    return {"sorted_ranks": sorted_ranks, "result_docs": result_docs}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
