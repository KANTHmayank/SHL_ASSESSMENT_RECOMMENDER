import streamlit as st
import json
import numpy as np
import re
from langchain.embeddings import OpenAIEmbeddings

# --- Config ---
st.set_page_config(page_title="SHL Assessment Recommender", layout="wide")

# --- Constants ---
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

# --- Load data ---
@st.cache_data
def load_data():
    with open("data/assessments4.json", "r") as f:
        metadata = json.load(f)
    texts = [f"{item['name']}\n\n{item['description']}" for item in metadata]
    return metadata, texts

metadata, texts = load_data()

# --- Embeddings ---
@st.cache_resource
def embed_docs(texts):
    embedder = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=st.secrets["OPENAI_API_KEY"])
    embs = np.array(embedder.embed_documents(texts))
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    return embedder, embs

embedder, doc_embs = embed_docs(texts)

# --- UI ---
st.title("🔍 SHL Assessment Recommendation Engine")
query = st.text_area("Describe your hiring scenario or query:", height=120)

if st.button("Get Recommendations"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Thinking..."):
            # ---- Filters (duration, test type, remote) ----
            duration_limit = None
            m = re.search(r"(\d+)\s*min", query, re.I)
            if m:
                duration_limit = int(m.group(1))

            required_test_types = []
            if re.search(r"cognitive", query, re.I): required_test_types.append("C")
            if re.search(r"personality", query, re.I): required_test_types.append("P")
            if re.search(r"knowledge", query, re.I): required_test_types.append("K")

            remote_required = bool(re.search(r"remote", query, re.I))

            # ---- Filter assessments ----
            candidate_idxs = []
            for i, item in enumerate(metadata):
                dur = item.get("duration_minutes")
                if duration_limit and dur:
                    if int(dur) > duration_limit:
                        continue
                if remote_required and item.get("remote_testing") != "Yes":
                    continue
                if required_test_types:
                    if not any(tt in item.get("test_types", []) for tt in required_test_types):
                        continue
                candidate_idxs.append(i)

            if not candidate_idxs:
                st.error("No matching assessments found.")
            else:
                # ---- Rank via embeddings ----
                q_emb = np.array(embedder.embed_query(query))
                q_emb /= np.linalg.norm(q_emb)
                sims = doc_embs[candidate_idxs].dot(q_emb)
                top_n = 10
                top_idxs = np.argsort(sims)[-top_n:][::-1]

                st.success(f"Top {len(top_idxs)} Recommendations")
                for rank in top_idxs:
                    i = candidate_idxs[rank]
                    item = metadata[i]
                    score = sims[rank]

                    st.markdown(f"### 🔗 [{item['name']}]({item['url']})")
                    st.markdown(f"- ⏱️ Duration: `{item.get('duration_minutes') or 'Unknown'} mins`")
                    st.markdown(f"- 🏠 Remote Testing: `{item.get('remote_testing', 'No')}`")
                    st.markdown(f"- ♻️ Adaptive/IRT: `{item.get('adaptive_irt', 'No')}`")

                    test_types = item.get("test_types", [])
                    if test_types:
                        readable = [f"`{code}` ({TEST_TYPE_MAP.get(code, 'Unknown')})" for code in test_types]
                        st.markdown("- 🧪 Test Types: " + ", ".join(readable))

                    st.markdown(f"📄 {item.get('description', '')}")
                    st.markdown("---")
