import faiss, numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer

EMB = "all-MiniLM-L6-v2"

class CatalogRetriever:
    def __init__(self, catalog_csv="data/processed/catalog.csv",
                 idx_path="data/processed/faiss.index",
                 meta_path="data/processed/meta.parquet"):
        self.catalog_csv, self.idx_path, self.meta_path = catalog_csv, idx_path, meta_path
        self.model = SentenceTransformer(EMB)
        self.index = None
        self.catalog = None

    def _text(self, row):
        return " | ".join([
            row.assessment_name or "",
            row.description or "",
            f"Type:{row.test_type}"
        ])

    def build(self):
        self.catalog = pd.read_csv(self.catalog_csv)
        corpus = self.catalog.apply(self._text, axis=1).tolist()
        X = self.model.encode(corpus, normalize_embeddings=True)
        dim = X.shape[1]
        idx = faiss.IndexFlatIP(dim)
        idx.add(X.astype("float32"))
        faiss.write_index(idx, self.idx_path)
        self.catalog.to_parquet(self.meta_path, index=False)

    def load(self):
        if self.index is None:
            self.index = faiss.read_index(self.idx_path)
            self.catalog = pd.read_parquet(self.meta_path)

    def search(self, query: str, k: int = 10):
        self.load()
        qv = self.model.encode([query], normalize_embeddings=True).astype("float32")
        D, I = self.index.search(qv, k)
        out = self.catalog.iloc[I[0]].copy()
        out["score"] = D[0]
        return out.reset_index(drop=True)
