"""Reciprocal Rank Fusion (RRF) Hybrid Retriever.

Follows Interface Segregation Principle (ISP) and Liskov Substitution Principle (LSP):
RRF fuses ranked lists of Hits or TREC run files without knowledge of underlying score scales.
score(d) = Σ_m 1 / (k + rank_m(d)), where rank is 1-based and ties are broken deterministically.
"""

from collections import defaultdict
from pathlib import Path
from typing import Sequence
import pandas as pd

from fever_ir.retrieval.base import BaseRetriever, Hit, Corpus
from fever_ir.utils.io import read_jsonl_stream, write_jsonl


def reciprocal_rank_fusion(
    ranked_lists: list[list[Hit]],
    k: int = 60,
    top_n: int = 100,
) -> list[Hit]:
    """Fuse multiple ranked lists of Hit objects via Reciprocal Rank Fusion."""
    scores: dict[int, float] = defaultdict(float)

    for hit_list in ranked_lists:
        for rank_0, hit in enumerate(hit_list):
            rank_1 = rank_0 + 1
            scores[hit.doc_idx] += 1.0 / (k + rank_1)

    sorted_items = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [Hit(doc_idx=doc_idx, score=score) for doc_idx, score in sorted_items[:top_n]]


def fuse_run_files(
    run_paths: Sequence[Path | str],
    output_path: Path | str,
    k: int = 60,
    depth: int = 100,
) -> int:
    """Fuse multiple TREC run JSONL files using RRF.

    Args:
        run_paths: List of run JSONL paths to fuse.
        output_path: Target path for the fused run file.
        k: RRF constant (default 60).
        depth: Max hits per query to keep.

    Returns:
        Number of queries fused.
    """
    runs_by_method: list[dict[int, list[dict]]] = []
    for p in run_paths:
        method_runs: dict[int, list[dict]] = {}
        for row in read_jsonl_stream(p):
            method_runs[int(row["qid"])] = row.get("hits", [])
        runs_by_method.append(method_runs)

    all_qids = sorted(runs_by_method[0].keys())
    fused_records = []

    for qid in all_qids:
        doc_scores: dict[str, float] = defaultdict(float)

        for method_runs in runs_by_method:
            hits = method_runs.get(qid, [])
            for hit in hits:
                doc_id = hit["doc_id"]
                rank = hit["rank"]
                doc_scores[doc_id] += 1.0 / (k + rank)

        sorted_docs = sorted(doc_scores.items(), key=lambda item: (-item[1], item[0]))
        fused_hits = [
            {"doc_id": doc_id, "rank": rank, "score": round(score, 6)}
            for rank, (doc_id, score) in enumerate(sorted_docs[:depth], start=1)
        ]

        fused_records.append({
            "qid": qid,
            "method": "rrf",
            "hits": fused_hits,
        })

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(fused_records, out)
    return len(fused_records)


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
        pass

    def load(self, path: Path | str) -> None:
        pass

    def search(self, query: str, k: int = 100) -> list[Hit]:
        """Search all constituent retrievers and fuse candidate rankings."""
        if not self.retrievers:
            raise ValueError("No constituent retrievers registered for RRF fusion.")
        ranked_lists = [r.search(query, k=100) for r in self.retrievers]
        return reciprocal_rank_fusion(ranked_lists, k=self.k, top_n=k)
