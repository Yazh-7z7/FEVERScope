#!/usr/bin/env python3
"""Phase 1 Step 5: Build ~5K-sentence mini-corpus from Wikipedia shards and FEVER dev claims.

Follows Section 5 Step 5 of plan.md:
- 60 claims from shared_task_dev (30 SUPPORTS + 30 REFUTES) + 10 NEI claims.
- Single streaming pass over all Wikipedia shards.
- Output: data/interim/mini_corpus.parquet, data/interim/mini_qrels.jsonl, data/interim/mini_nei_claims.jsonl.
- Automatically fits and persists TF-IDF and BM25 indexes upon completion.
"""

import argparse
from pathlib import Path
from fever_ir.config import load_config
from fever_ir.data.corpus_builder import build_corpus_and_qrels
from fever_ir.data.loaders import load_claims_stream, ClaimRecord
from fever_ir.data.sampling import deterministic_page_bucket, sample_claims
from fever_ir.data.wiki_parser import parse_wiki_page, ParsedWikiPage, normalize_nfc
from fever_ir.retrieval.manager import index_manager
from fever_ir.utils.io import read_jsonl_stream, write_jsonl
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
    logger.info("Loaded configuration (seed=%d)", config.seed)

    dev_path = config.paths.raw / "shared_task_dev.jsonl"
    if not dev_path.is_file():
        raise FileNotFoundError(f"Missing dev claims file: {dev_path}")

    # 1. Sample mini claims (30 SUPPORTS, 30 REFUTES, 10 NEI)
    logger.info("Loading claims from %s...", dev_path)
    all_dev_claims = list(load_claims_stream(dev_path))

    verifiable_claims = [c for c in all_dev_claims if not c.is_nei]
    nei_claims = [c for c in all_dev_claims if c.is_nei][: config.sampling.mini.nei_claims]

    mini_claims = sample_claims(
        verifiable_claims,
        n_supports=config.sampling.mini.claims_per_label,
        n_refutes=config.sampling.mini.claims_per_label,
        seed=config.seed,
    )
    logger.info("Sampled %d verifiable claims and %d NEI claims.", len(mini_claims), len(nei_claims))

    # 2. Collect gold pages
    gold_pages: set[str] = set()
    for claim in mini_claims:
        for page_title, _ in claim.flattened_gold_sentences:
            gold_pages.add(normalize_nfc(page_title))
    logger.info("Identified %d unique gold Wikipedia pages to extract.", len(gold_pages))

    # 3. Stream Wikipedia shards
    wiki_dir = config.paths.raw / "wiki-pages"
    if not wiki_dir.is_dir():
        logger.warning("Wiki directory %s not yet available. Waiting for shards...", wiki_dir)
        return

    shard_files = sorted(wiki_dir.glob("wiki-*.jsonl"))
    if not shard_files:
        logger.warning("No wiki-*.jsonl shards found in %s.", wiki_dir)
        return

    logger.info("Streaming across %d Wikipedia shards...", len(shard_files))
    matched_pages: list[ParsedWikiPage] = []
    distractor_count = 0
    max_distractor_pages = config.sampling.mini.random_pages

    for shard in shard_files:
        for raw_record in read_jsonl_stream(shard):
            raw_title = raw_record.get("id", "")
            norm_title = normalize_nfc(raw_title)

            is_gold = norm_title in gold_pages
            is_distractor = False

            if not is_gold and distractor_count < max_distractor_pages:
                # Deterministic distractor selection
                if deterministic_page_bucket(norm_title, modulus=1000) < 15:
                    is_distractor = True
                    distractor_count += 1

            if is_gold or is_distractor:
                parsed_page = parse_wiki_page(raw_record)
                matched_pages.append(parsed_page)

    logger.info("Extracted %d total pages (%d distractors).", len(matched_pages), distractor_count)

    # 4. Build Corpus & Qrels
    corpus_df, qrels, dropped = build_corpus_and_qrels(
        matched_pages,
        mini_claims,
        use_title_augmentation=config.preprocessing.use_title_augmentation,
    )

    logger.info(
        "Compiled corpus with %d sentence rows and %d valid qrels queries.",
        len(corpus_df),
        len(qrels),
    )
    if dropped:
        logger.warning("Dropped %d claims due to missing evidence in extracted pages.", len(dropped))

    # 5. Save artifacts to data/interim/
    config.paths.interim.mkdir(parents=True, exist_ok=True)
    corpus_out = config.paths.interim / "mini_corpus.parquet"
    corpus_df.to_parquet(corpus_out, index=False)
    logger.info("Saved corpus to %s", corpus_out)

    qrels_out = config.paths.interim / "mini_qrels.jsonl"
    write_jsonl([q.to_dict() for q in qrels], qrels_out)
    logger.info("Saved qrels to %s", qrels_out)

    nei_out = config.paths.interim / "mini_nei_claims.jsonl"
    write_jsonl(
        [{"id": c.id, "claim": c.claim, "label": c.label} for c in nei_claims],
        nei_out,
    )
    logger.info("Saved NEI claims to %s", nei_out)

    # 6. Fit and persist TF-IDF and BM25 immediately
    logger.info("Triggering automatic index construction and persistence...")
    index_manager.rebuild_indexes(corpus_df)
    logger.info("Indexes built and persisted successfully! Ready for search.")


if __name__ == "__main__":
    main()
