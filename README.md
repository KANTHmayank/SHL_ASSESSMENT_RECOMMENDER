# SHL Assessment Recommendation System

An AI-powered recommendation engine for SHL assessments. It lets hiring managers enter free-form queries and returns relevant SHL tests using OpenAI embeddings and semantic search.

---

## 🚀 Features

- 🔍 Full catalog scraping from SHL product pages
- 📊 Metadata extraction: name, description, duration, test types, remote/IRT
- 🤖 Embedding + semantic retrieval using OpenAI
- ⚡ FastAPI API endpoint (`/recommend`)
- 🎯 Streamlit UI to query interactively
- 🧠 Filters on duration, remote availability, and test types (like cognitive, personality)

---

## 📁 Project Structure

shl-recommender/ └── main.py # FastAPI backend ├── data/ │ └── assessments.json # Scraped assessment metadata ├── streamlit_app.py # UI for querying ├── scrape.py # SHL catalog scraper ├── requirements.txt └── README.md

## 🧩 Technologies Used

- **Scraping**: `requests`, `beautifulsoup4`
- **Semantic Search**: `OpenAI`, `LangChain`, `NumPy`
- **Backend**: `FastAPI`, `Uvicorn`
- **Frontend**: `Streamlit`
- **Hosting**: `Render`, `Streamlit Cloud`

---

## ⚙️ Setup Instructions

### 1. Clone Repo
git clone https://github.com/YOUR_USERNAME/shl-recommender.git
cd shl-recommender

2. Install Dependencies
   pip install -r requirements.txt

3. Set OpenAI API Key
   openai.api_key=st.secrets["OPENAI_API_KEY"]

🧪 Run Locally
FastAPI Server:
uvicorn app.main:app --reload


Test it with:
curl -X POST http://localhost:8000/recommend \
     -H "Content-Type: application/json" \
     -d '{"query":"30-minute cognitive test for remote analysts"}'

Streamlit UI:
streamlit run streamlit_app.py




