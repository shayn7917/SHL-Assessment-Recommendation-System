import streamlit as st
import requests
import pandas as pd

# Point to your running API
API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="SHL GenAI Assessment Recommender", layout="wide")
st.title("SHL GenAI Assessment Recommender")

st.markdown(
    """
    Enter a job description or natural language query below.
    The system will recommend the most relevant SHL assessments from the catalog.
    """
)

query = st.text_area("Enter job description or query:")
top_k = st.slider("Number of recommendations", 5, 10, 10)

if st.button("Recommend"):
    if not query.strip():
        st.warning("Please enter a query first.")
    else:
        with st.spinner("Generating recommendations..."):
            try:
                payload = {"query": query, "top_k": top_k}
                response = requests.post(f"{API_URL}/recommend", json=payload, timeout=60)
                if response.ok:
                    recs = response.json()["recommendations"]
                    df = pd.DataFrame(recs)[["assessment_name", "test_type", "url", "score"]]
                    st.success("Recommendations generated successfully:")
                    st.dataframe(df, use_container_width=True)
                    if "explanation" in response.json():
                        st.markdown("### 💡 Gemini Explanation")
                        st.markdown(
                            f"<div style='background-color:#1a1a1a; padding:15px; border-radius:10px; color:#e6e6e6;'>"
                            f"{response.json()['explanation']}</div>",
                            unsafe_allow_html=True
                        )


                else:
                    st.error(f"API error {response.status_code}")
            except Exception as e:
                st.error(f"Error: {e}")
