"""Retrieval and Index Manager.

Follows Single Responsibility Principle (SRP):
Orchestrates loading existing indexes from disk (indexes/tfidf.pkl, indexes/bm25.pkl),
or building and persisting them if they do not yet exist.
Provides document metadata lookup for frontend ranking cards.
"""

from pathlib import Path
from typing import Any
import pandas as pd

from fever_ir.config import AppConfig, load_config
from fever_ir.retrieval.base import Hit
from fever_ir.retrieval.bm25 import BM25Retriever
from fever_ir.retrieval.dense import DenseRetriever
from fever_ir.retrieval.rrf import RRFHybridRetriever, reciprocal_rank_fusion
from fever_ir.retrieval.tfidf import TFIDFRetriever
from fever_ir.utils.logging import get_logger

logger = get_logger("index_manager")


class IndexManager:
    """Singleton/service manager for persistent indexes and retrieval engines."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or load_config()
        self.tfidf = TFIDFRetriever(use_title_augmentation=self.config.preprocessing.use_title_augmentation)
        self.bm25 = BM25Retriever(
            k1=self.config.bm25.k1,
            b=self.config.bm25.b,
            use_title_augmentation=self.config.preprocessing.use_title_augmentation,
        )
        self.dense = DenseRetriever(model_name=self.config.dense.model_name)
        self.rrf = RRFHybridRetriever(k=self.config.rrf.k)

        self.corpus_df: pd.DataFrame | None = None
        self._doc_map: dict[str, dict[str, Any]] = {}

        self.tfidf_path = self.config.paths.indexes / "tfidf.pkl"
        self.bm25_path = self.config.paths.indexes / "bm25.pkl"

    def ensure_indexes_loaded(self) -> dict[str, bool]:
        """Ensure TF-IDF and BM25 indexes are loaded. If not found on disk, build and persist."""
        status = {
            "tfidf": False,
            "bm25": False,
            "dense": self.dense.is_ready,
        }

        # Check TF-IDF
        if self.tfidf_path.is_file():
            try:
                self.tfidf.load(self.tfidf_path)
                status["tfidf"] = True
            except Exception as exc:
                logger.warning("Error loading TF-IDF from %s: %s", self.tfidf_path, exc)

        # Check BM25
        if self.bm25_path.is_file():
            try:
                self.bm25.load(self.bm25_path)
                status["bm25"] = True
            except Exception as exc:
                logger.warning("Error loading BM25 from %s: %s", self.bm25_path, exc)

        # If either is not loaded, we need corpus to build
        if not (status["tfidf"] and status["bm25"]):
            logger.info("One or more indexes missing. Building from available corpus data...")
            self._build_missing_indexes(status)

        self._ensure_corpus_metadata()
        return self.get_status()

    def _ensure_corpus_metadata(self) -> None:
        """Load corpus parquet if available for rich document details in search results."""
        corpus_paths = [
            self.config.paths.processed / "corpus.parquet",
            self.config.paths.interim / "mini_corpus.parquet",
        ]
        for path in corpus_paths:
            if path.is_file() and self.corpus_df is None:
                try:
                    self.corpus_df = pd.read_parquet(path)
                    logger.info("Loaded corpus metadata from %s (%d rows)", path, len(self.corpus_df))
                    self._doc_map = {
                        str(row["doc_id"]): {
                            "doc_id": str(row["doc_id"]),
                            "page_title": str(row["page_title"]),
                            "sent_id": int(row["sent_id"]),
                            "text": str(row["text"]),
                        }
                        for _, row in self.corpus_df.iterrows()
                    }
                    break
                except Exception as exc:
                    logger.warning("Could not read corpus metadata from %s: %s", path, exc)

    def _build_missing_indexes(self, current_status: dict[str, bool]) -> None:
        """Build missing TF-IDF or BM25 index and save to disk."""
        # Find corpus file
        corpus_path = None
        for candidate in [
            self.config.paths.interim / "mini_corpus.parquet",
            self.config.paths.processed / "corpus.parquet",
        ]:
            if candidate.is_file():
                corpus_path = candidate
                break

        if corpus_path is None:
            logger.warning("No compiled corpus found at data/interim/ or data/processed/. Cannot build indexes yet.")
            return

        df = pd.read_parquet(corpus_path)
        self.corpus_df = df

        if not current_status["tfidf"]:
            self.tfidf.build(df)
            self.tfidf.save(self.tfidf_path)

        if not current_status["bm25"]:
            self.bm25.build(df)
            self.bm25.save(self.bm25_path)

    def rebuild_indexes(self, corpus_df: pd.DataFrame | None = None) -> None:
        """Force rebuild of TF-IDF and BM25 indexes and save to disk."""
        if corpus_df is not None:
            self.corpus_df = corpus_df
        else:
            self._ensure_corpus_metadata()

        if self.corpus_df is None or len(self.corpus_df) == 0:
            raise ValueError("No corpus available to rebuild indexes.")

        logger.info("Rebuilding TF-IDF index on %d documents...", len(self.corpus_df))
        self.tfidf.build(self.corpus_df)
        self.tfidf.save(self.tfidf_path)

        logger.info("Rebuilding BM25 index on %d documents...", len(self.corpus_df))
        self.bm25.build(self.corpus_df)
        self.bm25.save(self.bm25_path)

        self._ensure_corpus_metadata()
        logger.info("Rebuild complete. Both indexes persisted.")

    def get_status(self) -> dict[str, Any]:
        """Return status dictionary of all retrievers."""
        return {
            "tfidf": {
                "ready": self.tfidf.is_ready,
                "path": str(self.tfidf_path),
                "num_docs": len(self.tfidf.doc_ids) if self.tfidf.is_ready else 0,
            },
            "bm25": {
                "ready": self.bm25.is_ready,
                "path": str(self.bm25_path),
                "num_docs": len(self.bm25.doc_ids) if self.bm25.is_ready else 0,
            },
            "dense": {
                "ready": self.dense.is_ready,
                "type": "pinecone",
                "index_name": self.dense.index_name,
                "has_api_key": bool(self.dense.api_key),
            },
            "rrf": {
                "ready": self.tfidf.is_ready and self.bm25.is_ready,
                "k": self.rrf.k,
            },
        }

    def search_all(self, query: str, k: int = 10) -> dict[str, list[dict[str, Any]]]:
        """Search query across all active retrievers and RRF hybrid."""
        results: dict[str, list[dict[str, Any]]] = {
            "tfidf": [],
            "bm25": [],
            "dense": [],
            "rrf": [],
        }

        active_hit_lists: list[list[Hit]] = []

        # 1. TF-IDF
        if self.tfidf.is_ready:
            tfidf_hits = self.tfidf.search(query, k=k)
            results["tfidf"] = self._enrich_hits(tfidf_hits, self.tfidf.doc_ids)
            if tfidf_hits:
                active_hit_lists.append(tfidf_hits)

        # 2. BM25
        if self.bm25.is_ready:
            bm25_hits = self.bm25.search(query, k=k)
            results["bm25"] = self._enrich_hits(bm25_hits, self.bm25.doc_ids)
            if bm25_hits:
                active_hit_lists.append(bm25_hits)

        # 3. Dense (Pinecone)
        if self.dense.is_ready:
            dense_hits = self.dense.search(query, k=k)
            results["dense"] = self._enrich_hits(dense_hits, [])
            if dense_hits:
                active_hit_lists.append(dense_hits)

        # 4. RRF Hybrid
        if active_hit_lists:
            fused_hits = reciprocal_rank_fusion(active_hit_lists, k=self.rrf.k, top_n=k)
            doc_id_ref = self.tfidf.doc_ids if self.tfidf.is_ready else (self.bm25.doc_ids if self.bm25.is_ready else [])
            results["rrf"] = self._enrich_hits(fused_hits, doc_id_ref)

        return results

    def _enrich_hits(self, hits: list[Hit], doc_ids_list: list[str]) -> list[dict[str, Any]]:
        """Attach document metadata (page_title, sent_id, text) to hits."""
        enriched = []
        for rank, hit in enumerate(hits, start=1):
            doc_id = doc_ids_list[hit.doc_idx] if (doc_ids_list and hit.doc_idx < len(doc_ids_list)) else f"doc::{hit.doc_idx}"

            meta = self._doc_map.get(doc_id, {})
            enriched.append({
                "rank": rank,
                "doc_idx": hit.doc_idx,
                "doc_id": doc_id,
                "score": round(hit.score, 4),
                "page_title": meta.get("page_title", doc_id.split("::")[0] if "::" in doc_id else "Wikipedia"),
                "sent_id": meta.get("sent_id", int(doc_id.split("::")[1]) if "::" in doc_id else 0),
                "text": meta.get("text", "Text indexed in corpus."),
            })
        return enriched


# Global default instance
index_manager = IndexManager()
