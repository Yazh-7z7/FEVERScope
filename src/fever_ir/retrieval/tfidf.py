"""TF-IDF and Cosine Similarity Retriever.

Follows Single Responsibility Principle (SRP) and Liskov Substitution Principle (LSP):
Implements BaseRetriever using scikit-learn's TfidfVectorizer.
Persists and loads indexes via pickle.
"""

from pathlib import Path
import pickle
from typing import Sequence
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from fever_ir.retrieval.base import BaseRetriever, Hit, Corpus
from fever_ir.utils.logging import get_logger

logger = get_logger("tfidf_retriever")


class TFIDFRetriever(BaseRetriever):
    """TF-IDF sparse vectorizer with cosine similarity scoring and disk persistence."""

    def __init__(self, use_title_augmentation: bool = True) -> None:
        self._name = "tfidf"
        self.use_title_augmentation = use_title_augmentation
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None  # scipy.sparse.csr_matrix
        self.doc_ids: list[str] = []
        self._is_ready: bool = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_ready(self) -> bool:
        """Return True if index is built or loaded and ready for search."""
        return self._is_ready

    def build(self, corpus: Corpus | pd.DataFrame) -> None:
        """Fit TF-IDF matrix over corpus sentences."""
        df = corpus.table if hasattr(corpus, "table") else corpus
        col = "text_with_title" if (self.use_title_augmentation and "text_with_title" in df.columns) else "text"
        texts = df[col].astype(str).tolist()
        self.doc_ids = df["doc_id"].astype(str).tolist()

        logger.info("Fitting TF-IDF vectorizer on %d sentences (column: '%s')...", len(texts), col)
        self.vectorizer = TfidfVectorizer(
            token_pattern=r"\b\w+\b",
            lowercase=True,
            stop_words="english",
            norm="l2",
            sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform(texts)
        self._is_ready = True
        logger.info("TF-IDF matrix built with shape: %s", self.matrix.shape)

    def save(self, path: Path | str) -> None:
        """Serialize TF-IDF model, matrix, and doc_ids."""
        if not self._is_ready or self.vectorizer is None or self.matrix is None:
            raise ValueError("Cannot save uninitialized TF-IDF index.")

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as f:
            pickle.dump(
                {
                    "vectorizer": self.vectorizer,
                    "matrix": self.matrix,
                    "doc_ids": self.doc_ids,
                    "use_title_augmentation": self.use_title_augmentation,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        logger.info("Saved TF-IDF index to %s", target)

    def load(self, path: Path | str) -> None:
        """Load serialized TF-IDF model and matrix from disk."""
        target = Path(path)
        if not target.is_file():
            raise FileNotFoundError(f"TF-IDF index file not found: {target}")

        with target.open("rb") as f:
            data = pickle.load(f)

        self.vectorizer = data["vectorizer"]
        self.matrix = data["matrix"]
        self.doc_ids = data["doc_ids"]
        self.use_title_augmentation = data.get("use_title_augmentation", True)
        self._is_ready = True
        logger.info("Loaded TF-IDF index from %s (%d documents)", target, len(self.doc_ids))

    def search(self, query: str, k: int = 100) -> list[Hit]:
        """Compute cosine similarity and return top-k Hits sorted descending."""
        if not self._is_ready or self.vectorizer is None or self.matrix is None:
            raise RuntimeError("TF-IDF retriever is not built or loaded.")

        if not query.strip() or k <= 0:
            return []

        query_vec = self.vectorizer.transform([query])
        # Vectorizer uses L2 norm; dot product equals cosine similarity
        scores = (self.matrix @ query_vec.T).toarray().ravel()

        top_k = min(k, len(scores))
        if top_k == 0:
            return []

        # Find top_k indices efficiently
        candidate_indices = np.argpartition(-scores, top_k - 1)[:top_k]
        sorted_indices = candidate_indices[np.argsort(-scores[candidate_indices])]

        hits: list[Hit] = []
        for idx in sorted_indices:
            score_val = float(scores[idx])
            if score_val <= 0.0 and len(hits) >= 10:
                break
            hits.append(Hit(doc_idx=int(idx), score=score_val))

        return hits
