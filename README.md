# FEVERScope: FEVER Evidence Retrieval Benchmark

Benchmarking and evaluating evidence retrieval for factual claims across the FEVER Wikipedia corpus:
- `tfidf`: TF-IDF + Cosine Similarity (scikit-learn)
- `bm25`: BM25 Okapi (rank-bm25)
- `dense`: Bi-encoder dense sentence embeddings + Cosine Similarity (sentence-transformers)
- `rrf`: Reciprocal Rank Fusion (hybrid fusion across lexical and dense models)

## Architecture & SOLID Principles

This codebase is organized around strict object-oriented and software engineering principles:
- **Single Responsibility Principle (SRP)**: Distinct modules handle data streaming, Wiki parsing, sampling, indexing, evaluation metrics, and reporting.
- **Open/Closed Principle (OCP)**: The retrieval architecture builds on `BaseRetriever`. New retrieval algorithms or re-rankers can be introduced without altering existing retrievers or evaluation pipelines.
- **Liskov Substitution Principle (LSP)**: All retrievers accept the single-source-of-truth `Corpus` and return ranked `Hit` objects with identical semantic contracts.
- **Interface Segregation Principle (ISP)**: Hybrid fusion (`RRF`) interacts exclusively with ranked candidate lists, independent of underlying scoring schemes or embedding vectors.
- **Dependency Inversion Principle (DIP)**: High-level CLI scripts and the evaluation runner interact with abstract interfaces and centralized configuration schemas (`configs/default.yaml`).

## Project Layout

```
fever-hybrid-ir/
├── plan.md                          # Master project specification
├── configs/default.yaml             # Hyperparameters, paths, and seeds
├── data/                            # Git-ignored (raw, interim, processed)
├── indexes/                         # Git-ignored serialized indexes
├── results/                         # TREC runs, metrics CSVs, figures
├── notes/data_findings.md           # Dataset inspection notes
├── src/fever_ir/
│   ├── config.py                    # Typed configuration dataclasses
│   ├── data/                        # Loaders, parser, sampling, corpus builder
│   ├── preprocessing/               # Text normalization and tokenization
│   ├── retrieval/                   # BaseRetriever, TF-IDF, BM25, Dense, RRF
│   ├── evaluation/                  # IR metrics, runner, significance, plots
│   └── utils/                       # Stream I/O and logging
├── scripts/                         # CLI execution scripts
├── app/streamlit_app.py             # Interactive demonstration app
└── tests/                           # Unit and acceptance tests
```

## Setup & Environment

Requires Python 3.11+.

```bash
# 1. Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run initial tests
pytest tests/
```
