"""Information Retrieval evaluation metrics.

Follows Single Responsibility Principle (SRP):
Pure mathematical functions calculating Precision@K, Recall@K, AP@100 (MAP),
nDCG@K, MRR, and FEVER multi-hop Set Coverage@K (§6).
"""

import math
from typing import Sequence


def precision_at_k(retrieved: Sequence[str], gold: set[str], k: int) -> float:
    """Calculate Precision@K: |relevant ∩ top-K| / K.

    If k <= 0, returns 0.0.
    """
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    rel_count = sum(1 for doc_id in top_k if doc_id in gold)
    return rel_count / float(k)


def recall_at_k(retrieved: Sequence[str], gold: set[str], k: int) -> float:
    """Calculate Recall@K: |relevant ∩ top-K| / |relevant|.

    If gold is empty, returns 0.0.
    """
    if not gold or k <= 0:
        return 0.0
    top_k = retrieved[:k]
    rel_count = sum(1 for doc_id in top_k if doc_id in gold)
    return rel_count / float(len(gold))


def average_precision(retrieved: Sequence[str], gold: set[str], depth: int = 100) -> float:
    """Calculate Average Precision at depth: (1 / |relevant|) * Σ (P@i * rel_i).

    Denominator is total relevant documents |relevant|, not retrieved-relevant (§6).
    """
    if not gold or depth <= 0:
        return 0.0

    hits = retrieved[:depth]
    cumulative_hits = 0
    score_sum = 0.0

    for i, doc_id in enumerate(hits, start=1):
        if doc_id in gold:
            cumulative_hits += 1
            score_sum += cumulative_hits / float(i)

    return score_sum / float(len(gold))


def ndcg_at_k(retrieved: Sequence[str], gold: set[str], k: int) -> float:
    """Calculate normalized Discounted Cumulative Gain at K (nDCG@K) with binary gains.

    DCG = Σ rel_i / log2(i + 1)
    IDCG = ideal DCG with min(|gold|, K) ones.
    """
    if not gold or k <= 0:
        return 0.0

    top_k = retrieved[:k]
    dcg = 0.0
    for i, doc_id in enumerate(top_k, start=1):
        if doc_id in gold:
            dcg += 1.0 / math.log2(i + 1)

    # Ideal DCG
    n_ideal = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_ideal + 1))

    return (dcg / idcg) if idcg > 0.0 else 0.0


def reciprocal_rank(retrieved: Sequence[str], gold: set[str]) -> float:
    """Calculate Reciprocal Rank (MRR component): 1 / rank of first relevant hit."""
    if not gold:
        return 0.0

    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in gold:
            return 1.0 / float(i)
    return 0.0


def set_coverage_at_k(retrieved: Sequence[str], evidence_sets: list[list[str]], k: int) -> float:
    """Calculate FEVER Set Coverage@K.

    Returns 1.0 if at least one complete evidence set is a subset of retrieved[:k],
    0.0 otherwise (§6).
    """
    if not evidence_sets or k <= 0:
        return 0.0

    retrieved_set = set(retrieved[:k])
    for ev_set in evidence_sets:
        if ev_set and set(ev_set).issubset(retrieved_set):
            return 1.0
    return 0.0
