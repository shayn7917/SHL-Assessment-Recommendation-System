import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


# =============================
# Load model and FAISS index
# =============================
@st.cache_resource
def load_resources():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index("data/processed/faiss.index")
    meta = pd.read_parquet("data/processed/meta.parquet")
    return model, index, meta

model, index, meta = load_resources()

# =============================
# Recommendation Function
# =============================
def recommend(query, top_k=5):
    query_vector = model.encode([query])
    D, I = index.search(np.array(query_vector).astype("float32"), top_k)
    results = meta.iloc[I[0]].copy()
    results["score"] = D[0]
    return results[["assessment_name", "url", "test_type", "score"]]

# =============================
# Streamlit UI
# =============================


st.title("SHL GenAI Assessment Recommender")
st.markdown(
    "Enter a job description or natural language query below. "
    "The system will recommend the most relevant SHL assessments from the catalog."
)

query = st.text_area("Enter job description or query:")
top_k = st.slider("Number of recommendations", 5, 10, 5)

if st.button("Recommend"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Generating recommendations..."):
            try:
                results = recommend(query, top_k)
                st.success("Recommendations generated successfully:")
                st.dataframe(results)
            except Exception as e:
                st.error(f"Error: {e}")
