#!/usr/bin/env python3
"""Phase 1 Step 5: Build ~5K-sentence mini-corpus from Wikipedia shards and FEVER dev claims.

Follows Section 5 Step 5 of plan.md:
- 60 claims from shared_task_dev (30 SUPPORTS + 30 REFUTES) + 10 NEI claims.
- Single streaming pass over all Wikipedia shards.
- Output: data/interim/mini_corpus.parquet, data/interim/mini_qrels.jsonl, data/interim/mini_nei_claims.jsonl.
"""

import argparse
from pathlib import Path

from fever_ir.config import load_config
from fever_ir.utils.logging import get_logger

logger = get_logger("build_mini_corpus")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build mini-corpus for FEVERScope Phase 1.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to configuration YAML file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    logger.info("Loaded configuration from %s", args.config)
    logger.info("Target interim directory: %s", config.paths.interim)
    print("Mini-corpus builder entry point ready for Phase 1 Step 5 execution.")


if __name__ == "__main__":
    main()
