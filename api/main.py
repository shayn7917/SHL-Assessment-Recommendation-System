from models.generator import generate_summary
from fastapi import FastAPI
from pydantic import BaseModel
from models.retriever import CatalogRetriever
import pandas as pd



app = FastAPI(title="SHL Assessment Recommender")

# load retriever and index
retriever = CatalogRetriever()
retriever.load()

class RecommendRequest(BaseModel):
    query: str
    top_k: int = 10

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/recommend")
def recommend(req: RecommendRequest):
    k = max(1, min(req.top_k, 10))
    df = retriever.search(req.query, k=20)

    # basic balancing logic for mixed intent queries
    q = req.query.lower()
    wants_p = any(w in q for w in ["personality", "behavior", "behaviour", "collaborat", "culture"])
    wants_k = any(w in q for w in ["technical", "developer", "engineer", "sql", "python", "java", "coding", "skill"])
    if wants_p and wants_k:
        k_items = df[df.test_type == "K"].head(k // 2)
        p_items = df[df.test_type == "P"].head(k - len(k_items))
        df = pd.concat([k_items, p_items]).head(k)
    else:
        df = df.head(k)

        recs = [
        {
            "assessment_name": r.assessment_name,
            "url": r.url,
            "test_type": r.test_type,
            "score": float(r.score)
        } for _, r in df.iterrows()
    ]

    explanation = generate_summary(req.query, df)

    return {
        "query": req.query,
        "recommendations": recs,
        "explanation": explanation
    }
