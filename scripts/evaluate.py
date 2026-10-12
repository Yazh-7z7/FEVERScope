#!/usr/bin/env python3
"""Evaluate saved TREC run files against ground-truth qrels.

Follows Section 6 of plan.md.
Computes P@K, Recall@K, MAP@100, nDCG@K, MRR, and Set Coverage@K.
Saves results/metrics/per_query_{method}_{split}.csv and results/metrics/summary.csv.
"""

import argparse
from pathlib import Path
import pandas as pd

from fever_ir.config import load_config
from fever_ir.evaluation.runner import evaluate_run
from fever_ir.utils.logging import get_logger

logger = get_logger("evaluate")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a retrieval run against ground truth qrels.")
    parser.add_argument(
        "--run",
        type=Path,
        required=True,
        help="Path to run file (e.g., results/runs/bm25_eval.jsonl).",
    )
    parser.add_argument(
        "--qrels",
        type=Path,
        default=None,
        help="Path to ground truth qrels file (default inferred from run filename).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to configuration YAML file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    run_path = Path(args.run)
    if not run_path.is_file():
        raise FileNotFoundError(f"Run file not found: {run_path}")

    # Infer qrels if not passed
    if args.qrels is not None:
        qrels_path = Path(args.qrels)
    else:
        # Check split from filename (e.g. bm25_eval.jsonl -> eval)
        stem = run_path.stem
        split = "eval" if "eval" in stem else ("tune" if "tune" in stem else "mini")
        qrels_path = config.paths.processed / f"qrels_{split}.jsonl"
        if not qrels_path.is_file():
            qrels_path = config.paths.interim / f"mini_qrels.jsonl"

    if not qrels_path.is_file():
        raise FileNotFoundError(f"Qrels file not found: {qrels_path}")

    logger.info("Evaluating Run: %s against Qrels: %s...", run_path, qrels_path)
    per_query_df, summary_df = evaluate_run(
        run_path=run_path,
        qrels_path=qrels_path,
        report_k=config.retrieval.report_k,
    )

    # Output directory
    metrics_dir = config.paths.results / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    method_name = run_path.stem
    per_query_out = metrics_dir / f"per_query_{method_name}.csv"
    per_query_df.to_csv(per_query_out, index=False)
    logger.info("Saved per-query metrics to %s", per_query_out)

    summary_out = metrics_dir / f"summary_{method_name}.csv"
    summary_df.to_csv(summary_out, index=False)
    logger.info("Saved summary metrics to %s", summary_out)

    # Print Formatted Table
    print("\n" + "=" * 70)
    print(f"BENCHMARK EVALUATION RESULTS: {method_name.upper()}")
    print("=" * 70)
    print(f"Queries Evaluated: {len(per_query_df):,}")
    print("-" * 70)

    # Key metric displays
    summary_row = summary_df.iloc[0]
    metrics_to_show = [
        ("Precision@1", "p@1"),
        ("Precision@5", "p@5"),
        ("Precision@10", "p@10"),
        ("Recall@5", "r@5"),
        ("Recall@10", "r@10"),
        ("Recall@20", "r@20"),
        ("MAP@100", "map@100"),
        ("nDCG@10", "ndcg@10"),
        ("MRR", "mrr"),
        ("Set Coverage@10", "set_cov@10"),
    ]

    for label, col in metrics_to_show:
        if col in summary_row:
            val = summary_row[col]
            print(f"  {label:<22} : {val:.4f} ({val * 100:.2f}%)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
