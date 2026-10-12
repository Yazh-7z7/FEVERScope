#!/usr/bin/env python3
"""Generate consolidated benchmark comparison table across all evaluated models."""

from pathlib import Path
import pandas as pd

from fever_ir.config import load_config
from fever_ir.utils.logging import get_logger

logger = get_logger("compare_methods")


def main() -> None:
    config = load_config("configs/default.yaml")
    metrics_dir = config.paths.results / "metrics"

    summary_files = {
        "TF-IDF": metrics_dir / "summary_tfidf_eval.csv",
        "BM25": metrics_dir / "summary_bm25_eval.csv",
        "RRF (TF-IDF + BM25)": metrics_dir / "summary_rrf_eval.csv",
    }

    rows = []
    for model_name, path in summary_files.items():
        if path.is_file():
            df = pd.read_csv(path)
            row = df.iloc[0].to_dict()
            row["Model"] = model_name
            rows.append(row)

    if not rows:
        print("No evaluation summary files found in results/metrics/.")
        return

    comp_df = pd.DataFrame(rows)
    cols_order = [
        "Model",
        "p@1",
        "p@5",
        "p@10",
        "r@5",
        "r@10",
        "r@20",
        "map@100",
        "ndcg@10",
        "mrr",
        "set_cov@10",
    ]
    comp_df = comp_df[[c for c in cols_order if c in comp_df.columns]]

    # Save to CSV
    out_csv = metrics_dir / "summary_comparison.csv"
    comp_df.to_csv(out_csv, index=False)
    logger.info("Saved comparison table to %s", out_csv)

    print("\n" + "=" * 90)
    print("FEVERSCOPE BENCHMARK COMPARISON (1,482 HELD-OUT EVALUATION QUERIES)")
    print("=" * 90)
    header = f"{'Model':<22} | {'P@1':<8} | {'P@10':<8} | {'R@10':<8} | {'R@20':<8} | {'MAP@100':<8} | {'nDCG@10':<8} | {'MRR':<8} | {'SetCov@10':<8}"
    print(header)
    print("-" * 90)

    for _, r in comp_df.iterrows():
        line = (
            f"{r['Model']:<22} | "
            f"{r['p@1']:<7.4f} | "
            f"{r['p@10']:<7.4f} | "
            f"{r['r@10']:<7.4f} | "
            f"{r['r@20']:<7.4f} | "
            f"{r['map@100']:<7.4f} | "
            f"{r['ndcg@10']:<7.4f} | "
            f"{r['mrr']:<7.4f} | "
            f"{r['set_cov@10']:<7.4f}"
        )
        print(line)
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
