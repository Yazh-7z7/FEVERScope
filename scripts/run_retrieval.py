#!/usr/bin/env python3
"""CLI entry point for executing retrieval methods and saving TREC run files.

Generates run files under results/runs/{method}_{split}.jsonl.
Decoupled from evaluation per D8.
"""

import argparse
from pathlib import Path

from fever_ir.config import load_config
from fever_ir.utils.logging import get_logger

logger = get_logger("run_retrieval")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval method and produce TREC run output.")
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["tfidf", "bm25", "dense", "rrf"],
        help="Retrieval method to execute.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="eval",
        choices=["tune", "eval", "mini"],
        help="Query split to run retrieval against.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to configuration YAML file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    logger.info("Retriever: %s | Split: %s | Config: %s", args.method, args.split, args.config)


if __name__ == "__main__":
    main()
