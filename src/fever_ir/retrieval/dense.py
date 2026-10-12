"""Dense Semantic Retriever using Sentence-Transformers.

Follows Single Responsibility Principle (SRP) and Liskov Substitution Principle (LSP):
Implements BaseRetriever using bi-encoder dense sentence embeddings and cosine similarity.
Phase 5 target module.
"""

from pathlib import Path
import pandas as pd

from fever_ir.retrieval.base import BaseRetriever, Hit, Corpus


class DenseRetriever(BaseRetriever):
    """Dense retriever using pretrained Sentence-Transformer models."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 64,
        normalize: bool = True,
        use_title_augmentation: bool = True,
    ) -> None:
        self._name = "dense"
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self.use_title_augmentation = use_title_augmentation

    @property
    def name(self) -> str:
        return self._name

    def build(self, corpus: Corpus | pd.DataFrame) -> None:
        """Encode corpus sentences into normalized dense embeddings."""
        raise NotImplementedError("DenseRetriever.build will be implemented in Phase 5.")

    def save(self, path: Path | str) -> None:
        """Save normalized embeddings matrix (numpy npy)."""
        raise NotImplementedError("DenseRetriever.save will be implemented in Phase 5.")

    def load(self, path: Path | str) -> None:
        """Load normalized embeddings matrix from disk."""
        raise NotImplementedError("DenseRetriever.load will be implemented in Phase 5.")

    def search(self, query: str, k: int = 100) -> list[Hit]:
        """Encode query and perform matrix multiplication cosine similarity."""
        raise NotImplementedError("DenseRetriever.search will be implemented in Phase 5.")
