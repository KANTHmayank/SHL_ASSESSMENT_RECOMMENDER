from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.model import get_retriever

app = FastAPI(title="SHL Assessment Recommender")
retriever = get_retriever()

class Query(BaseModel):
    query: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/recommend")
def recommend(q: Query):
    qtext = q.query.strip()
    if not qtext:
        raise HTTPException(400, "Query cannot be empty")
    docs = retriever.get_relevant_documents(qtext)
    if not docs:
        raise HTTPException(404, "No assessments found")
    # Only return metadata
    return {"recommendations": [d.metadata for d in docs]}
