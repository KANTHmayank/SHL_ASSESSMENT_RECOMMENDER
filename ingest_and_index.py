# ingest_and_index.py
import json, os
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def main():
    # 1) load your scraped JSON
    with open("data/assessments4.json") as f:
        items = json.load(f)

    # 2) turn each item into a LangChain Document
    docs = []
    for it in items:
        text = f"{it['name']}\n\n{it['description']}"
        docs.append(Document(
            page_content=text,
            metadata={
                "name": it["name"],
                "url": it["url"],
                "duration_minutes": it.get("duration_minutes",""),
                "remote_testing": it.get("remote_testing",""),
                "adaptive_irt": it.get("adaptive_irt","")
            }
        ))

    # 3) embed with OpenAI
    embedder = OpenAIEmbeddings(model="text-embedding-3-small",api_key="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")
    vs = FAISS.from_documents(docs, embedder)

    # 4) persist locally
    os.makedirs("faiss_index", exist_ok=True)
    vs.save_local("faiss_index")
    print(f"✅ Indexed {len(docs)} items → faiss_index/")

if __name__ == "__main__":
    main()
