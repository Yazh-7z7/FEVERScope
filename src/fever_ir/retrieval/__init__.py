"""Retrieval module for FEVERScope: Base interface, TF-IDF, BM25, Dense, and RRF."""

from fever_ir.retrieval.base import Hit, BaseRetriever, Corpus
from fever_ir.retrieval.tfidf import TFIDFRetriever
from fever_ir.retrieval.bm25 import BM25Retriever
from fever_ir.retrieval.dense import DenseRetriever
from fever_ir.retrieval.rrf import RRFHybridRetriever

__all__ = [
    "Hit",
    "BaseRetriever",
    "Corpus",
    "TFIDFRetriever",
    "BM25Retriever",
    "DenseRetriever",
    "RRFHybridRetriever",
]
