"""Core abstractions for information retrieval.

Follows Open/Closed Principle (OCP) and Liskov Substitution Principle (LSP):
Defines the abstract BaseRetriever interface and immutable Hit model.
All retrieval implementations (TF-IDF, BM25, Dense, RRF) adhere strictly to this contract.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import pandas as pd


@dataclass(frozen=True)
class Hit:
    """Represents a scored document hit in the unified corpus.

    Attributes:
        doc_idx: Integer index corresponding to rows 0..N-1 in the Corpus table.
        score: Relevance score (higher indicates greater relevance).
    """
    doc_idx: int
    score: float


class Corpus(Protocol):
    """Protocol for single-source-of-truth Corpus.

    The underlying DataFrame must contain columns:
    - doc_idx: int (0..N-1, contiguous, no gaps)
    - doc_id: str ("{page_title}::{sent_id}")
    - page_title: str
    - sent_id: int
    - text: str
    - text_with_title: str
    """
    table: pd.DataFrame


class BaseRetriever(ABC):
    """Abstract Base Class for all retrieval methods."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the retriever (e.g. 'tfidf', 'bm25', 'dense', 'rrf')."""
        ...

    @abstractmethod
    def build(self, corpus: Corpus | pd.DataFrame) -> None:
        """Index the provided unified corpus."""
        ...

    @abstractmethod
    def save(self, path: Path | str) -> None:
        """Persist index state and artifacts to disk."""
        ...

    @abstractmethod
    def load(self, path: Path | str) -> None:
        """Load index state and artifacts from disk."""
        ...

    @abstractmethod
    def search(self, query: str, k: int = 100) -> list[Hit]:
        """Search the indexed corpus and return top-k hits sorted by descending score."""
        ...
