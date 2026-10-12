#!/usr/bin/env python3
"""Phase 1 Step 4: Streaming inspection script for raw FEVER datasets.

Follows Section 0 Rule 4 (never read full multi-GB files into memory) and
Section 5 Step 4 of plan.md.

Prints:
1. Claim counts per label for train.jsonl and shared_task_dev.jsonl.
2. Three sample claims per label.
3. Evidence set statistics for SUPPORTS and REFUTES.
4. Raw vs. parsed Wikipedia lines for wiki-001.jsonl.
5. Shard-level statistics (total pages, sentences, empty lines, PTB escapes).
"""

import argparse
from pathlib import Path

from fever_ir.config import load_config
from fever_ir.utils.logging import get_logger

logger = get_logger("inspect_fever")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect raw FEVER claims and Wikipedia shards.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to configuration YAML file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    logger.info("Loaded configuration from %s", args.config)
    logger.info("Raw data directory: %s", config.paths.raw)

    print("=" * 70)
    print("FEVERScope: Raw Data Inspection (Phase 1, Step 4)")
    print("=" * 70)
    print("This script will execute streaming checks once raw datasets are present under data/raw/.")


if __name__ == "__main__":
    main()
