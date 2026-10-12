"""BM25 (Okapi) Lexical Retriever.

Follows Single Responsibility Principle (SRP) and Liskov Substitution Principle (LSP):
Implements BaseRetriever using rank-bm25 with tunable k1 and b parameters.
Persists and loads indexes via pickle.
"""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from fever_ir.preprocessing.tokenize import tokenize_text
from fever_ir.retrieval.base import BaseRetriever, Hit, Corpus
from fever_ir.utils.logging import get_logger

logger = get_logger("bm25_retriever")


class BM25Retriever(BaseRetriever):
    """BM25 Okapi retriever over tokenized sentences with disk persistence."""

    def __init__(self, k1: float = 1.5, b: float = 0.75, use_title_augmentation: bool = True) -> None:
        self._name = "bm25"
        self.k1 = k1
        self.b = b
        self.use_title_augmentation = use_title_augmentation
        self.bm25: BM25Okapi | None = None
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
        """Tokenize corpus and construct BM25Okapi index."""
        df = corpus.table if hasattr(corpus, "table") else corpus
        col = "text_with_title" if (self.use_title_augmentation and "text_with_title" in df.columns) else "text"
        texts = df[col].astype(str).tolist()
        self.doc_ids = df["doc_id"].astype(str).tolist()

        logger.info("Tokenizing %d corpus sentences for BM25 (column: '%s')...", len(texts), col)
        tokenized_corpus = [tokenize_text(text, remove_stopwords=True) for text in texts]

        logger.info("Building BM25Okapi index (k1=%.2f, b=%.2f)...", self.k1, self.b)
        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        self._is_ready = True
        logger.info("BM25 index built with %d documents.", len(self.doc_ids))

    def save(self, path: Path | str) -> None:
        """Serialize BM25 index and document identifiers."""
        if not self._is_ready or self.bm25 is None:
            raise ValueError("Cannot save uninitialized BM25 index.")

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as f:
            pickle.dump(
                {
                    "bm25": self.bm25,
                    "doc_ids": self.doc_ids,
                    "k1": self.k1,
                    "b": self.b,
                    "use_title_augmentation": self.use_title_augmentation,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        logger.info("Saved BM25 index to %s", target)

    def load(self, path: Path | str) -> None:
        """Load serialized BM25 index from disk."""
        target = Path(path)
        if not target.is_file():
            raise FileNotFoundError(f"BM25 index file not found: {target}")

        with target.open("rb") as f:
            data = pickle.load(f)

        self.bm25 = data["bm25"]
        self.doc_ids = data["doc_ids"]
        self.k1 = data.get("k1", 1.5)
        self.b = data.get("b", 0.75)
        self.use_title_augmentation = data.get("use_title_augmentation", True)
        self._is_ready = True
        logger.info("Loaded BM25 index from %s (%d documents)", target, len(self.doc_ids))

    def search(self, query: str, k: int = 100) -> list[Hit]:
        """Score tokenized query against BM25 index and return top-k Hits."""
        if not self._is_ready or self.bm25 is None:
            raise RuntimeError("BM25 retriever is not built or loaded.")

        tokens = tokenize_text(query, remove_stopwords=True)
        if not tokens or k <= 0:
            return []

        scores = np.asarray(self.bm25.get_scores(tokens))
        top_k = min(k, len(scores))
        if top_k == 0:
            return []

        candidate_indices = np.argpartition(-scores, top_k - 1)[:top_k]
        sorted_indices = candidate_indices[np.argsort(-scores[candidate_indices])]

        hits: list[Hit] = []
        for idx in sorted_indices:
            score_val = float(scores[idx])
            if score_val <= 0.0 and len(hits) >= 10:
                break
            hits.append(Hit(doc_idx=int(idx), score=score_val))

        return hits
