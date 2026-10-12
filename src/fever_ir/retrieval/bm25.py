"""BM25 (Okapi) Lexical Retriever.

Follows Single Responsibility Principle (SRP) and Liskov Substitution Principle (LSP):
Implements BaseRetriever using rank-bm25 with tunable k1 and b parameters.
Phase 4 target module.
"""

from pathlib import Path
import pandas as pd

from fever_ir.retrieval.base import BaseRetriever, Hit, Corpus


class BM25Retriever(BaseRetriever):
    """BM25 Okapi retriever over tokenized sentences."""

    def __init__(self, k1: float = 1.5, b: float = 0.75, use_title_augmentation: bool = True) -> None:
        self._name = "bm25"
        self.k1 = k1
        self.b = b
        self.use_title_augmentation = use_title_augmentation

    @property
    def name(self) -> str:
        return self._name

    def build(self, corpus: Corpus | pd.DataFrame) -> None:
        """Tokenize corpus and construct BM25Okapi index."""
        raise NotImplementedError("BM25Retriever.build will be implemented in Phase 4.")

    def save(self, path: Path | str) -> None:
        """Serialize BM25 index."""
        raise NotImplementedError("BM25Retriever.save will be implemented in Phase 4.")

    def load(self, path: Path | str) -> None:
        """Load serialized BM25 index."""
        raise NotImplementedError("BM25Retriever.load will be implemented in Phase 4.")

    def search(self, query: str, k: int = 100) -> list[Hit]:
        """Score tokenized query against BM25 index and return top-k Hits."""
        raise NotImplementedError("BM25Retriever.search will be implemented in Phase 4.")
