"""Reciprocal Rank Fusion (RRF) Hybrid Retriever.

Follows Interface Segregation Principle (ISP) and Liskov Substitution Principle (LSP):
RRF fuses ranked lists of Hits without knowledge of underlying score scales or models.
score(d) = Σ_m 1 / (k + rank_m(d)), where rank is 1-based and ties are broken deterministically by doc_idx (§4.2).
Phase 6 target module.
"""

from collections import defaultdict
from pathlib import Path
import pandas as pd

from fever_ir.retrieval.base import BaseRetriever, Hit, Corpus


def reciprocal_rank_fusion(
    ranked_lists: list[list[Hit]],
    k: int = 60,
    top_n: int = 100,
) -> list[Hit]:
    """Fuse multiple ranked lists of Hit objects via Reciprocal Rank Fusion.

    Args:
        ranked_lists: List of ranked Hit lists from different retrieval systems.
        k: RRF constant (default 60, tuned on tuning split only).
        top_n: Maximum number of fused hits to return.

    Returns:
        List of fused Hit objects sorted descending by RRF score.
    """
    scores: dict[int, float] = defaultdict(float)

    for hit_list in ranked_lists:
        for rank_0, hit in enumerate(hit_list):
            rank_1 = rank_0 + 1
            scores[hit.doc_idx] += 1.0 / (k + rank_1)

    # Sort descending by score, tie-break deterministically by doc_idx
    sorted_items = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [Hit(doc_idx=doc_idx, score=score) for doc_idx, score in sorted_items[:top_n]]


class RRFHybridRetriever(BaseRetriever):
    """Hybrid Retriever combining TF-IDF, BM25, and Dense via RRF."""

    def __init__(self, k: int = 60, retrievers: list[BaseRetriever] | None = None) -> None:
        self._name = "rrf"
        self.k = k
        self.retrievers = retrievers or []

    @property
    def name(self) -> str:
        return self._name

    def build(self, corpus: Corpus | pd.DataFrame) -> None:
        """Trigger build on constituent retrievers."""
        for retriever in self.retrievers:
            retriever.build(corpus)

    def save(self, path: Path | str) -> None:
        """Serialize configuration parameters."""
        raise NotImplementedError("RRFHybridRetriever.save will be implemented in Phase 6.")

    def load(self, path: Path | str) -> None:
        """Load configuration parameters."""
        raise NotImplementedError("RRFHybridRetriever.load will be implemented in Phase 6.")

    def search(self, query: str, k: int = 100) -> list[Hit]:
        """Search all constituent retrievers and fuse candidate rankings."""
        if not self.retrievers:
            raise ValueError("No constituent retrievers registered for RRF fusion.")
        ranked_lists = [r.search(query, k=100) for r in self.retrievers]
        return reciprocal_rank_fusion(ranked_lists, k=self.k, top_n=k)
