#!/usr/bin/env python3
"""Phase 2: Build full-scale ~150K-200K sentence benchmark corpus for FEVER.

Follows Section 2 (D1, D2, D4) and Section 4.4 of plan.md:
- 1,500 evaluation claims from shared_task_dev.jsonl (750 SUPPORTS, 750 REFUTES)
- 500 tuning claims from train.jsonl (250 SUPPORTS, 250 REFUTES)
- 5,000 distractor train claims from train.jsonl (gold pages only)
- Single streaming pass over all 109 Wikipedia shards
- Compiles data/processed/corpus.parquet (~150K-200K sentences)
- Compiles data/processed/qrels_eval.jsonl and data/processed/qrels_tune.jsonl
- Immediately fits and persists local TF-IDF and BM25 indexes
"""

import json
from pathlib import Path
import pandas as pd

from fever_ir.config import load_config
from fever_ir.data.corpus_builder import build_corpus_and_qrels, format_doc_id
from fever_ir.data.loaders import load_claims_stream, ClaimRecord
from fever_ir.data.sampling import deterministic_page_bucket, sample_claims
from fever_ir.data.wiki_parser import parse_wiki_page, ParsedWikiPage, normalize_nfc
from fever_ir.retrieval.manager import index_manager
from fever_ir.utils.io import read_jsonl_stream, write_jsonl
from fever_ir.utils.logging import get_logger

logger = get_logger("build_full_corpus")


def main() -> None:
    config = load_config("configs/default.yaml")
    seed = config.seed
    logger.info("Initializing Full Corpus Construction (seed=%d)...", seed)

    raw_dir = config.paths.raw
    processed_dir = config.paths.processed
    processed_dir.mkdir(parents=True, exist_ok=True)

    dev_file = raw_dir / "shared_task_dev.jsonl"
    train_file = raw_dir / "train.jsonl"
    wiki_dir = raw_dir / "wiki-pages"

    if not dev_file.is_file() or not train_file.is_file():
        raise FileNotFoundError("Raw claim files missing in data/raw/")
    if not wiki_dir.is_dir():
        raise FileNotFoundError("Wikipedia shards missing in data/raw/wiki-pages/")

    # 1. Sample Evaluation Claims from shared_task_dev.jsonl (1,500 claims: 750 SUPPORTS / 750 REFUTES)
    logger.info("Loading dev claims from %s...", dev_file)
    dev_claims_all = list(load_claims_stream(dev_file))
    dev_verifiable = [c for c in dev_claims_all if not c.is_nei]
    dev_nei = [c for c in dev_claims_all if c.is_nei][: config.sampling.nei_claims]

    eval_claims = sample_claims(
        dev_verifiable,
        n_supports=config.sampling.eval_claims // 2,
        n_refutes=config.sampling.eval_claims // 2,
        seed=seed,
    )
    logger.info("Selected %d eval claims (held-out test queries).", len(eval_claims))

    # 2. Sample Tuning Claims (500) and Distractor Train Claims (5,000) from train.jsonl
    logger.info("Loading train claims from %s...", train_file)
    train_claims_all = list(load_claims_stream(train_file))
    train_verifiable = [c for c in train_claims_all if not c.is_nei]

    tune_claims = sample_claims(
        train_verifiable,
        n_supports=config.sampling.tune_claims // 2,
        n_refutes=config.sampling.tune_claims // 2,
        seed=seed,
    )
    logger.info("Selected %d tuning claims (hyperparameter optimization queries).", len(tune_claims))

    # Remaining train claims for topical distractors
    tune_ids = {c.id for c in tune_claims}
    remaining_train = [c for c in train_verifiable if c.id not in tune_ids]
    distractor_claims = sample_claims(
        remaining_train,
        n_supports=config.sampling.distractor_train_claims // 2,
        n_refutes=config.sampling.distractor_train_claims // 2,
        seed=seed + 1,
    )
    logger.info("Selected %d train claims for topical distractors.", len(distractor_claims))

    # 3. Collect Target Gold Wikipedia Pages
    target_pages: set[str] = set()
    for claim in eval_claims:
        for title, _ in claim.flattened_gold_sentences:
            target_pages.add(normalize_nfc(title))

    for claim in tune_claims:
        for title, _ in claim.flattened_gold_sentences:
            target_pages.add(normalize_nfc(title))

    distractor_pages: set[str] = set()
    for claim in distractor_claims:
        for title, _ in claim.flattened_gold_sentences:
            distractor_pages.add(normalize_nfc(title))

    wanted_pages = target_pages | distractor_pages
    logger.info(
        "Total target Wikipedia pages: %d (Eval/Tune Gold: %d, Train Distractors: %d)",
        len(wanted_pages),
        len(target_pages),
        len(distractor_pages),
    )

    # 4. Stream 109 Wikipedia Shards
    shard_files = sorted(wiki_dir.glob("wiki-*.jsonl"))
    logger.info("Streaming across %d Wikipedia shards to collect target pages + random distractors...", len(shard_files))

    matched_pages: list[ParsedWikiPage] = []
    random_distractor_count = 0
    max_random_distractors = 1500  # Extra random Wikipedia pages to widen candidate pool

    for shard in shard_files:
        for raw_record in read_jsonl_stream(shard):
            raw_title = raw_record.get("id", "")
            norm_title = normalize_nfc(raw_title)

            is_wanted = norm_title in wanted_pages
            is_random = False

            if not is_wanted and random_distractor_count < max_random_distractors:
                # Deterministic random page bucket
                if deterministic_page_bucket(norm_title, modulus=1000) < 5:
                    is_random = True
                    random_distractor_count += 1

            if is_wanted or is_random:
                parsed = parse_wiki_page(raw_record)
                matched_pages.append(parsed)

    logger.info("Extracted %d total Wikipedia pages (%d random distractors).", len(matched_pages), random_distractor_count)

    # 5. Build Corpus DataFrame and Qrels
    all_active_claims = eval_claims + tune_claims
    corpus_df, all_qrels, dropped = build_corpus_and_qrels(
        matched_pages,
        all_active_claims,
        use_title_augmentation=config.preprocessing.use_title_augmentation,
    )

    logger.info("Compiled full benchmark corpus with %d sentence rows!", len(corpus_df))
    if dropped:
        logger.info("%d claims dropped due to unresolved evidence.", len(dropped))

    # Split Qrels into Eval and Tune
    eval_ids = {c.id for c in eval_claims}
    qrels_eval = [q for q in all_qrels if q.qid in eval_ids]
    qrels_tune = [q for q in all_qrels if q.qid not in eval_ids]

    # 6. Save Artifacts
    corpus_path = processed_dir / "corpus.parquet"
    corpus_df.to_parquet(corpus_path, index=False)
    logger.info("Saved full corpus to %s (file size: ~%.1f MB)", corpus_path, corpus_path.stat().st_size / (1024 * 1024))

    qrels_eval_path = processed_dir / "qrels_eval.jsonl"
    write_jsonl([q.to_dict() for q in qrels_eval], qrels_eval_path)
    logger.info("Saved %d eval qrels to %s", len(qrels_eval), qrels_eval_path)

    qrels_tune_path = processed_dir / "qrels_tune.jsonl"
    write_jsonl([q.to_dict() for q in qrels_tune], qrels_tune_path)
    logger.info("Saved %d tune qrels to %s", len(qrels_tune), qrels_tune_path)

    # Save manifest documenting seeds, sizes, filters
    manifest = {
        "seed": seed,
        "corpus_sentences": len(corpus_df),
        "corpus_pages": len(matched_pages),
        "eval_claims": len(qrels_eval),
        "tune_claims": len(qrels_tune),
        "dropped_claims": len(dropped),
    }
    with (processed_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Saved experiment manifest to %s", processed_dir / "manifest.json")

    # 7. Rebuild and Persist Local TF-IDF & BM25 Indexes
    logger.info("Rebuilding persistent local TF-IDF and BM25 indexes on the full corpus...")
    index_manager.rebuild_indexes(corpus_df)
    logger.info("Full benchmark indexing complete! System ready for local & cloud search.")


if __name__ == "__main__":
    main()
