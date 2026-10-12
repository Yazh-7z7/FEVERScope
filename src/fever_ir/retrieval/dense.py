"""Dense Semantic Retriever interfacing with Pinecone Vector Database.

Follows Single Responsibility Principle (SRP) and Liskov Substitution Principle (LSP):
Queries the serverless Pinecone index built using Sentence-Transformers embeddings.
Falls back gracefully when credentials are not yet configured.
"""

import os
from pathlib import Path
from typing import Any
import pandas as pd

from fever_ir.retrieval.base import BaseRetriever, Hit, Corpus
from fever_ir.utils.logging import get_logger

logger = get_logger("dense_retriever")


class DenseRetriever(BaseRetriever):
    """Dense Semantic Retriever connecting to Pinecone."""

    def __init__(
        self,
        api_key: str | None = None,
        index_name: str | None = None,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self._name = "dense"
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.index_name = index_name or os.getenv("PINECONE_INDEX_NAME", "fever-evidence-index")
        self.model_name = model_name
        self.pc = None
        self.index = None
        self.encoder = None
        self._is_ready: bool = False
        self._init_client()

    def _init_client(self) -> None:
        """Attempt to initialize connection to Pinecone."""
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        if not self.api_key:
            self.api_key = os.getenv("PINECONE_API_KEY")
        if not self.index_name:
            self.index_name = os.getenv("PINECONE_INDEX_NAME", "fever-evidence-index")

        if not self.api_key:
            logger.info("Pinecone API key not provided. Dense retriever is inactive until key is supplied.")
            return

        try:
            from pinecone import Pinecone
            from sentence_transformers import SentenceTransformer

            self.pc = Pinecone(api_key=self.api_key)
            self.index = self.pc.Index(self.index_name)
            self.encoder = SentenceTransformer(self.model_name)
            self._is_ready = True
            logger.info("Successfully connected to Pinecone index: %s", self.index_name)
        except Exception as exc:
            logger.warning("Could not connect to Pinecone: %s. Dense retriever disabled.", exc)
            self._is_ready = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_ready(self) -> bool:
        """Return True if Pinecone index and query encoder are ready."""
        return self._is_ready

    def build(self, corpus: Corpus | pd.DataFrame) -> None:
        """Embedding and upserting is delegated to Colab (notebooks/fever_pinecone_colab.ipynb)."""
        logger.info("Dense embedding generation runs on GPU via Colab script.")

    def save(self, path: Path | str) -> None:
        """Index is remotely hosted on Pinecone serverless infrastructure."""
        pass

    def load(self, path: Path | str = "") -> None:
        """Reload connection to Pinecone."""
        self.api_key = os.getenv("PINECONE_API_KEY") or self.api_key
        self.index_name = os.getenv("PINECONE_INDEX_NAME") or self.index_name
        self._init_client()

    def search(self, query: str, k: int = 100) -> list[Hit]:
        """Encode query and search Pinecone vector index."""
        if not self._is_ready or self.index is None or self.encoder is None:
            return []

        query_vec = self.encoder.encode(query, normalize_embeddings=True).tolist()
        response = self.index.query(vector=query_vec, top_k=k, include_metadata=True)

        hits: list[Hit] = []
        for match in response.get("matches", []):
            score = float(match.get("score", 0.0))
            metadata = match.get("metadata", {})
            doc_idx = int(metadata.get("doc_idx", len(hits)))
            hits.append(Hit(doc_idx=doc_idx, score=score))

        return hits
