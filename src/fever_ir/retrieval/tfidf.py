"""TF-IDF and Cosine Similarity Retriever.

Follows Single Responsibility Principle (SRP) and Liskov Substitution Principle (LSP):
Implements BaseRetriever using scikit-learn's TfidfVectorizer.
Phase 4 target module.
"""

from pathlib import Path
import pandas as pd

from fever_ir.retrieval.base import BaseRetriever, Hit, Corpus


class TFIDFRetriever(BaseRetriever):
    """TF-IDF sparse vectorizer with cosine similarity scoring."""

    def __init__(self, use_title_augmentation: bool = True) -> None:
        self._name = "tfidf"
        self.use_title_augmentation = use_title_augmentation

    @property
    def name(self) -> str:
        return self._name

    def build(self, corpus: Corpus | pd.DataFrame) -> None:
        """Fit TF-IDF matrix over corpus sentences."""
        raise NotImplementedError("TFIDFRetriever.build will be implemented in Phase 4.")

    def save(self, path: Path | str) -> None:
        """Serialize TF-IDF model and matrix."""
        raise NotImplementedError("TFIDFRetriever.save will be implemented in Phase 4.")

    def load(self, path: Path | str) -> None:
        """Load serialized TF-IDF model and matrix."""
        raise NotImplementedError("TFIDFRetriever.load will be implemented in Phase 4.")

    def search(self, query: str, k: int = 100) -> list[Hit]:
        """Compute cosine similarity and return top-k Hits."""
        raise NotImplementedError("TFIDFRetriever.search will be implemented in Phase 4.")
