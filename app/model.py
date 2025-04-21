from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def get_retriever(k=10):
    embedder = OpenAIEmbeddings(model="text-embedding-3-small")
    db = FAISS.load_local("faiss_index", embedder, allow_dangerous_deserialization=True)
    return db.as_retriever(search_kwargs={"k": k})
