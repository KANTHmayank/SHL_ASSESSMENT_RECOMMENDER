# SHL Assessment Recommendation System

An end‑to‑end pipeline that scrapes SHL’s product catalog, builds an embedding‑based vector index with OpenAI + FAISS via LangChain, and serves recommendations via FastAPI or a self‑contained Streamlit app.

---

## 🚀 Features

- **Scraper** (`scrape_shl_full.py`):  
  • Collects every “Pre‑packaged Job” and “Individual Test” product URL (handles pagination)  
  • Visits each detail page and extracts: name, URL, description, duration (mins), remote‑testing flag, adaptive‑IRT flag  
  • Outputs `data/assessments.json`

- **Ingest & Index** (`ingest_and_index.py`):  
  • Loads `data/assessments.json`  
  • Wraps each record as a LangChain `Document`  
  • Creates a FAISS vector store using OpenAI embeddings (`text-embedding-3-small`)  
  • Persists index under `faiss_index/`

- **API** (`app/main.py`, `app/model.py`):  
  • `GET /health` → `{ "status": "ok" }`  
  • `POST /recommend` → `{ "recommendations": [ { name, url, duration_minutes, remote_testing, adaptive_irt }, … ] }`  
  • Powered by FastAPI + Uvicorn

- **UI (optional)** (`streamlit_app.py`):  
  • Self‑contained Streamlit app  
  • Query box + “Get Recommendations” button  
  • Displays results with metadata and links

---

## 📋 Project Structure

shl-recommender/ ├── data/ │ └── assessments.json # scraper output ├── faiss_index/ # vector store (generated) ├── app/ │ ├── init.py │ ├── main.py # FastAPI server │ └── model.py # retriever loader ├── scrape_shl_full.py # production scraper ├── ingest_and_index.py # embedding + FAISS indexer ├── streamlit_app.py # optional Streamlit UI ├── requirements.txt # dependencies ├── one-pager.pdf # concise approach summary └── README.md # this file


