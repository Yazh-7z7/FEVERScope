"""FastAPI Backend for FEVERScope.

Provides REST API endpoints for:
- System status and retriever readiness
- Sample claim retrieval for one-click testing
- Multi-method evidence retrieval (TF-IDF, BM25, Pinecone Dense, RRF Hybrid)
- Pinecone credential updates from UI
"""

import os
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

# Load .env variables before importing application modules
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from fever_ir.config import load_config
from fever_ir.data.loaders import load_claims_stream
from fever_ir.retrieval.manager import index_manager
from fever_ir.utils.logging import get_logger

logger = get_logger("fastapi_backend")

app = FastAPI(
    title="FEVERScope API",
    description="Backend service for FEVER Evidence Retrieval (TF-IDF vs BM25 vs Dense vs RRF)",
    version="0.1.0",
)

# Enable CORS for React frontend (default Vite dev port 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event() -> None:
    """Load persistent indexes on server boot if available."""
    logger.info("Initializing FEVERScope backend service...")
    try:
        index_manager.dense.load("")
        status = index_manager.ensure_indexes_loaded()
        logger.info("Retriever index status on boot: %s", status)
    except Exception as exc:
        logger.error("Error during startup index check: %s", exc)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Factual claim or search query")
    k: int = Field(10, ge=1, le=100, description="Top-k evidence sentences to retrieve")


class PineconeConfigRequest(BaseModel):
    api_key: str = Field(..., description="Pinecone API key")
    index_name: str = Field("fever-evidence-index", description="Pinecone index name")


@app.get("/api/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "service": "FEVERScope API"}


@app.get("/api/status")
def get_status() -> dict[str, Any]:
    """Get readiness status of all retrieval engines."""
    return index_manager.get_status()


@app.get("/api/sample-claims")
def get_sample_claims(limit: int = 8) -> list[dict[str, Any]]:
    """Return diverse sample claims from shared_task_dev.jsonl for UI quick-select."""
    dev_path = Path("data/raw/shared_task_dev.jsonl")
    if not dev_path.is_file():
        # Fallback default claims if raw dataset not yet downloaded
        return [
            {
                "id": 1,
                "label": "SUPPORTS",
                "claim": "Roman Empire was one of the largest empires in history.",
                "gold_doc_ids": ["Roman_Empire::0"],
            },
            {
                "id": 2,
                "label": "REFUTES",
                "claim": "Albert Einstein was an award-winning chef.",
                "gold_doc_ids": ["Albert_Einstein::0"],
            },
            {
                "id": 3,
                "label": "SUPPORTS",
                "claim": "The Beatles were an English rock band formed in Liverpool.",
                "gold_doc_ids": ["The_Beatles::0"],
            },
            {
                "id": 4,
                "label": "REFUTES",
                "claim": "The moon is made of cheddar cheese.",
                "gold_doc_ids": ["Moon::0"],
            },
        ]

    samples: list[dict[str, Any]] = []
    supports_count = 0
    refutes_count = 0
    target_per_label = max(1, limit // 2)

    for claim in load_claims_stream(dev_path):
        if claim.is_nei:
            continue
        if claim.label == "SUPPORTS" and supports_count < target_per_label:
            samples.append({
                "id": claim.id,
                "label": claim.label,
                "claim": claim.claim,
                "gold_doc_ids": [f"{p}::{s}" for p, s in claim.flattened_gold_sentences],
            })
            supports_count += 1
        elif claim.label == "REFUTES" and refutes_count < target_per_label:
            samples.append({
                "id": claim.id,
                "label": claim.label,
                "claim": claim.claim,
                "gold_doc_ids": [f"{p}::{s}" for p, s in claim.flattened_gold_sentences],
            })
            refutes_count += 1

        if supports_count >= target_per_label and refutes_count >= target_per_label:
            break

    return samples


@app.post("/api/search")
def search_evidence(req: SearchRequest) -> dict[str, Any]:
    """Execute evidence retrieval across all enabled methods for the given query."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be empty.")

    results = index_manager.search_all(req.query, k=req.k)
    status = index_manager.get_status()

    return {
        "query": req.query,
        "k": req.k,
        "status": status,
        "results": results,
    }


@app.post("/api/pinecone-config")
def update_pinecone_config(req: PineconeConfigRequest) -> dict[str, Any]:
    """Dynamically configure Pinecone credentials and reconnect dense retriever."""
    os.environ["PINECONE_API_KEY"] = req.api_key
    os.environ["PINECONE_INDEX_NAME"] = req.index_name

    index_manager.dense.api_key = req.api_key
    index_manager.dense.index_name = req.index_name
    index_manager.dense.load("")

@app.post("/api/rebuild-indexes")
def trigger_rebuild_indexes() -> dict[str, Any]:
    """Trigger rebuilding TF-IDF and BM25 indexes from current corpus parquet."""
    try:
        index_manager.rebuild_indexes()
        return {"message": "Indexes rebuilt successfully.", "status": index_manager.get_status()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

