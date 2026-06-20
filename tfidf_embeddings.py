"""
A fully-offline embedding function for Chroma based on TF-IDF + SVD
(scikit-learn only, no model downloads required).

This is a stand-in for a real neural embedding model (like
sentence-transformers' all-MiniLM-L6-v2) for environments without internet
access to HuggingFace. Retrieval quality is lower than dense embeddings
(no semantic understanding, just weighted term overlap + latent topics),
but it's enough to demonstrate the full RAG pipeline end-to-end.

If you have internet access in your own environment, swap this out for:

    from chromadb.utils import embedding_functions
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
"""

import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from chromadb import EmbeddingFunction, Documents, Embeddings


class TfidfEmbeddingFunction(EmbeddingFunction):
    def __init__(self, n_components=256):
        self.n_components = n_components
        self.vectorizer = TfidfVectorizer(
            max_features=20000,
            stop_words="english",
            token_pattern=r"(?u)\b\w[\w\.]*\w\b",  # keep things like "foo.bar"
        )
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self._fitted = False

    def fit(self, documents):
        tfidf_matrix = self.vectorizer.fit_transform(documents)
        n_comp = min(self.n_components, tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1] - 1)
        n_comp = max(n_comp, 2)
        self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
        self.svd.fit(tfidf_matrix)
        self._fitted = True

    def __call__(self, input: Documents) -> Embeddings:
        if not self._fitted:
            raise RuntimeError(
                "TfidfEmbeddingFunction must be fit() on a corpus before use, "
                "or loaded from a saved .pkl via load()."
            )
        tfidf_matrix = self.vectorizer.transform(input)
        reduced = self.svd.transform(tfidf_matrix)
        # Normalize so cosine-similarity-based distance metrics behave well
        norms = np.linalg.norm(reduced, axis=1, keepdims=True)
        norms[norms == 0] = 1
        reduced = reduced / norms
        return reduced.tolist()

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({
                "vectorizer": self.vectorizer,
                "svd": self.svd,
                "n_components": self.n_components,
                "_fitted": self._fitted,
            }, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(n_components=state["n_components"])
        obj.vectorizer = state["vectorizer"]
        obj.svd = state["svd"]
        obj._fitted = state["_fitted"]
        return obj
