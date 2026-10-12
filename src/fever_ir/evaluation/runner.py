"""Evaluation runner decoupled from retrieval models.

Follows Single Responsibility Principle (SRP) and Dependency Inversion Principle (DIP):
Evaluates TREC-style run files against qrels ground truth without importing or executing
any retrieval models.
"""

from pathlib import Path
from typing import Sequence
import pandas as pd

from fever_ir.data.corpus_builder import QrelsRecord
from fever_ir.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    average_precision,
    ndcg_at_k,
    reciprocal_rank,
    set_coverage_at_k,
)
from fever_ir.utils.io import read_jsonl_stream


def evaluate_run(
    run_path: Path | str,
    qrels_path: Path | str,
    report_k: Sequence[int] = (1, 3, 5, 10, 20),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate a saved run file against qrels.

    Args:
        run_path: Path to run JSONL file (keys: qid, method, hits=[{doc_id, rank, score}, ...]).
        qrels_path: Path to qrels JSONL file.
        report_k: Cutoffs K to evaluate.

    Returns:
        per_query_df: DataFrame of per-query metrics.
        summary_df: DataFrame of macro-averaged metrics.
    """
    qrels_map: dict[int, QrelsRecord] = {}
    for record in read_jsonl_stream(qrels_path):
        qrels_map[int(record["qid"])] = QrelsRecord(
            qid=int(record["qid"]),
            label=str(record["label"]),
            gold_doc_ids=list(record["gold_doc_ids"]),
            evidence_sets=list(record.get("evidence_sets", [])),
        )

    rows = []
    for run_entry in read_jsonl_stream(run_path):
        qid = int(run_entry["qid"])
        if qid not in qrels_map:
            continue

        qrels = qrels_map[qid]
        gold_set = set(qrels.gold_doc_ids)
        hits = [hit["doc_id"] for hit in run_entry.get("hits", [])]

        row: dict[str, int | float | str] = {
            "qid": qid,
            "label": qrels.label,
            "mrr": reciprocal_rank(hits, gold_set),
            "map@100": average_precision(hits, gold_set, depth=100),
        }

        for k in report_k:
            row[f"p@{k}"] = precision_at_k(hits, gold_set, k=k)
            row[f"r@{k}"] = recall_at_k(hits, gold_set, k=k)
            row[f"ndcg@{k}"] = ndcg_at_k(hits, gold_set, k=k)
            row[f"set_cov@{k}"] = set_coverage_at_k(hits, qrels.evidence_sets, k=k)

        rows.append(row)

    per_query_df = pd.DataFrame(rows)
    numeric_cols = [c for c in per_query_df.columns if c not in ("qid", "label")]
    summary_df = pd.DataFrame([per_query_df[numeric_cols].mean().to_dict()])

    return per_query_df, summary_df
