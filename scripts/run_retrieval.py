#!/usr/bin/env python3
"""Run retrieval methods and produce TREC-style run files.

Generates results/runs/{method}_{split}.jsonl.
Decoupled from evaluation per D8 of plan.md.
"""

import argparse
from pathlib import Path
import time
from tqdm import tqdm

from fever_ir.config import load_config
from fever_ir.retrieval.base import BaseRetriever
from fever_ir.retrieval.bm25 import BM25Retriever
from fever_ir.retrieval.dense import DenseRetriever
from fever_ir.retrieval.rrf import reciprocal_rank_fusion
from fever_ir.retrieval.tfidf import TFIDFRetriever
from fever_ir.utils.io import read_jsonl_stream, write_jsonl
from fever_ir.utils.logging import get_logger

logger = get_logger("run_retrieval")


def load_retriever(method: str, config_path: Path = Path("configs/default.yaml")) -> BaseRetriever:
    """Load and initialize requested retrieval model from disk."""
    config = load_config(config_path)

    if method == "tfidf":
        retriever = TFIDFRetriever(use_title_augmentation=config.preprocessing.use_title_augmentation)
        index_file = config.paths.indexes / "tfidf.pkl"
        logger.info("Loading TF-IDF index from %s...", index_file)
        retriever.load(index_file)
        return retriever

    elif method == "bm25":
        retriever = BM25Retriever(
            k1=config.bm25.k1,
            b=config.bm25.b,
            use_title_augmentation=config.preprocessing.use_title_augmentation,
        )
        index_file = config.paths.indexes / "bm25.pkl"
        logger.info("Loading BM25 index from %s...", index_file)
        retriever.load(index_file)
        return retriever

    elif method == "dense":
        retriever = DenseRetriever(model_name=config.dense.model_name)
        if not retriever.is_ready:
            raise RuntimeError("Dense retriever is not ready. Ensure PINECONE_API_KEY is configured.")
        return retriever

    else:
        raise ValueError(f"Unknown retriever method: {method}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute retrieval method and generate TREC run file.")
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["tfidf", "bm25", "dense"],
        help="Retrieval method to execute.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="eval",
        choices=["eval", "tune"],
        help="Query split to run retrieval against.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=100,
        help="Top-K retrieval depth per query (default: 100 per §2 D7).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to configuration YAML file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    query_file = config.paths.processed / f"queries_{args.split}.jsonl"
    if not query_file.is_file():
        raise FileNotFoundError(f"Queries file not found: {query_file}")

    # Load retriever
    retriever = load_retriever(args.method, args.config)
    doc_ids = retriever.doc_ids

    queries = list(read_jsonl_stream(query_file))
    logger.info("Loaded %d queries from %s", len(queries), query_file)

    run_records = []
    start_time = time.perf_counter()

    logger.info("Running retrieval for method='%s' across %d queries (depth=%d)...", args.method, len(queries), args.depth)
    for q_item in tqdm(queries, desc=f"Searching ({args.method})"):
        qid = q_item["qid"]
        query_text = q_item["query"]

        hits = retriever.search(query_text, k=args.depth)

        hits_formatted = []
        for rank, hit in enumerate(hits, start=1):
            doc_id = doc_ids[hit.doc_idx] if hit.doc_idx < len(doc_ids) else f"doc::{hit.doc_idx}"
            hits_formatted.append({
                "doc_id": doc_id,
                "rank": rank,
                "score": round(float(hit.score), 5),
            })

        run_records.append({
            "qid": qid,
            "method": args.method,
            "hits": hits_formatted,
        })

    elapsed = time.perf_counter() - start_time
    qps = len(queries) / elapsed if elapsed > 0 else 0
    avg_latency_ms = (elapsed / len(queries)) * 1000 if len(queries) > 0 else 0

    logger.info(
        "Completed %d queries in %.2fs (%.1f QPS, avg latency: %.2f ms/query)",
        len(queries),
        elapsed,
        qps,
        avg_latency_ms,
    )

    # Save run file
    output_dir = config.paths.results / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_file = output_dir / f"{args.method}_{args.split}.jsonl"

    write_jsonl(run_records, run_file)
    logger.info("Saved TREC run file to %s (%d records).", run_file, len(run_records))


if __name__ == "__main__":
    main()
