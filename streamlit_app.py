import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

st.set_page_config(page_title="SHL Assessment Recommender")

@st.cache_resource
def load_retriever():
    embedder = OpenAIEmbeddings(model="text-embedding-3-small")
    db = FAISS.load_local("faiss_index", embedder, allow_dangerous_deserialization=True)
    return db.as_retriever(search_kwargs={"k": 10})

retriever = load_retriever()

st.title("🔍 SHL Assessment Recommender")
query = st.text_area("Enter job description or role requirements:")

if st.button("Get Recommendations"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        docs = retriever.get_relevant_documents(query)
        if not docs:
            st.error("No relevant assessments found.")
        for d in docs:
            m = d.metadata
            st.markdown(f"### [{m['name']}]({m['url']})")
            st.write(f"• Duration: {m['duration_minutes']} mins  |  Remote: {m['remote_testing']}  |  Adaptive/IRT: {m['adaptive_irt']}")
            st.write("---")
