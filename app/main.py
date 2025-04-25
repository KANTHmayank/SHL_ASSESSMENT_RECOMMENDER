from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import numpy as np
from langchain.embeddings import OpenAIEmbeddings
import re
import streamlit as st
import os

openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]

app = FastAPI()

# ====== 1. Load scraped data ======
with open("data/assessments4.json", "r") as f:
    metadata = json.load(f)

texts = [f"{item['name']}\n\n{item['description']}" for item in metadata]

# ====== 2. Build vector embeddings ======
embedder = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
doc_embs = np.array(embedder.embed_documents(texts))
doc_embs /= np.linalg.norm(doc_embs, axis=1, keepdims=True)

# ====== 3. Test type mapping ======
TEST_TYPE_MAP = {
    "A": "Ability and aptitude",
    "B": "Biodata and situational judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment exercises",
    "K": "Knowledge & skills",
    "P": "Personality & behavior",
    "S": "Simulations"
}

class Query(BaseModel):
    query: str

# ====== 4. Endpoints ======

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/recommend")
def recommend(q: Query):
    user_query = q.query.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Query is empty.")

    # --- Parse query filters (optional) ---
    duration_limit = None
    m = re.search(r"(\d+)\s*min", user_query, re.I)
    if m:
        duration_limit = int(m.group(1))
    
    required_test_types = []
    if re.search(r"cognitive", user_query, re.I):
        required_test_types.append("C")
    if re.search(r"personality", user_query, re.I):
        required_test_types.append("P")
    if re.search(r"knowledge", user_query, re.I):
        required_test_types.append("K")

    remote_required = bool(re.search(r"remote", user_query, re.I))

    # --- Embed user query ---
    query_emb = np.array(embedder.embed_query(user_query))
    query_emb /= np.linalg.norm(query_emb)

    # --- Apply metadata filtering before similarity ranking ---
    candidate_idxs = []
    for idx, item in enumerate(metadata):
        # Duration filter
        dur = item.get("duration_minutes")
        if duration_limit and dur:
            if int(dur) > duration_limit:
                continue
        # Remote testing filter
        if remote_required and item.get("remote_testing") != "Yes":
            continue
        # Test types filter
        if required_test_types:
            if not any(tt in item.get("test_types", []) for tt in required_test_types):
                continue
        candidate_idxs.append(idx)

    if not candidate_idxs:
        raise HTTPException(status_code=404, detail="No matching assessments found.")

    # --- Semantic ranking ---
    scores = doc_embs[candidate_idxs].dot(query_emb)
    top_n = 10
    top_idxs = np.argsort(scores)[-top_n:][::-1]

    # --- Build final response ---
    results = []
    for rank in top_idxs:
        i = candidate_idxs[rank]
        item = metadata[i]
        results.append({
            "name": item["name"],
            "url": item["url"],
            "duration_minutes": item.get("duration_minutes", ""),
            "remote_testing": item.get("remote_testing", ""),
            "adaptive_irt": item.get("adaptive_irt", ""),
            "test_types": [
                {code, TEST_TYPE_MAP.get(code, "Unknown")}
                for code in item.get("test_types", [])
            ],
            "score": round(float(scores[rank]), 4)
        })

    return {"recommendations": results}
