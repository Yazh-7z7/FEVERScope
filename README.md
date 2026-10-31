# FEVERScope — FEVER Evidence Retrieval Benchmark

> An **Information Retrieval** system that retrieves and ranks evidence sentences for factual claims using the [FEVER](https://fever.ai/) Wikipedia corpus — benchmarking four retrieval strategies through an interactive **FastAPI + React** web application.

FEVERScope compares four retrieval approaches head-to-head on the same corpus and evaluation queries:

| Method | Algorithm | Library |
|--------|-----------|---------|
| `tfidf` | TF-IDF + Cosine Similarity | scikit-learn |
| `bm25`  | BM25 Okapi | rank-bm25 |
| `dense` | Bi-encoder Sentence Embeddings + Cosine Similarity (via Pinecone) | sentence-transformers, pinecone-client |
| `rrf`   | Reciprocal Rank Fusion — hybrid fusion of the three above | custom implementation |

**Research question:** *Can hybrid (RRF) retrieval find relevant evidence more effectively than any individual method?*

**Evaluation metrics:** Precision@K · Recall@K · MAP · nDCG · MRR · Set Coverage@K · Precision–Recall curves

---

## Table of Contents

- [Architecture](#architecture)
- [Project Layout](#project-layout)
- [Requirements](#requirements)
- [Setup & Environment](#setup--environment)
- [Data Acquisition](#data-acquisition)
- [Building Indexes](#building-indexes)
- [Running the App](#running-the-app)
- [API Reference](#api-reference)
- [CLI Scripts](#cli-scripts)
- [Evaluation & Metrics](#evaluation--metrics)
- [Configuration](#configuration)
- [Design Decisions](#key-design-decisions)
- [Known Pitfalls](#known-pitfalls)
- [Roadmap](#roadmap)

---

## Architecture

FEVERScope is a full-stack application with a Python backend and a React frontend:

```
┌──────────────────────────────────┐
│       React Frontend (Vite)      │  http://localhost:5173
│  frontend/src/App.jsx            │
│  React 19 · Vanilla CSS          │
└─────────────────┬────────────────┘
                  │  REST API (JSON)
                  ▼
┌──────────────────────────────────┐
│      FastAPI Backend             │  http://localhost:8000
│  backend/main.py                 │
│                                  │
│  ┌──────────────────────────┐    │
│  │  IndexManager            │    │
│  │  ├─ TFIDFRetriever       │    │
│  │  ├─ BM25Retriever        │    │
│  │  ├─ DenseRetriever       │◄───┼── Pinecone (cloud)
│  │  └─ RRFHybridRetriever   │    │
│  └──────────────────────────┘    │
└─────────────────┬────────────────┘
                  │
    ┌─────────────▼─────────────┐
    │  src/fever_ir/ (core IR)  │
    │  configs/default.yaml     │
    │  data/  indexes/          │
    └───────────────────────────┘
```

### SOLID Principles

- **SRP** — Distinct modules handle data streaming, Wiki parsing, sampling, indexing, metrics, and reporting.
- **OCP** — New retrieval algorithms plug into `BaseRetriever` without touching existing code.
- **LSP** — All retrievers accept a unified `Corpus` and return ranked `Hit` objects with identical contracts.
- **ISP** — `RRFHybridRetriever` consumes only ranked lists, independent of underlying scoring.
- **DIP** — CLI scripts and the evaluation runner depend on abstract interfaces and `configs/default.yaml`.

---

## Project Layout

```
FEVERScope/
├── .env.example                     # Template for environment variables
├── configs/
│   └── default.yaml                 # All hyperparameters, paths, and seeds
├── data/                            # Git-ignored
│   ├── raw/                         # train.jsonl, shared_task_dev.jsonl, wiki-pages/
│   ├── interim/                     # mini_corpus.parquet, mini_qrels.jsonl
│   └── processed/                   # corpus.parquet, qrels_eval.jsonl, queries_eval.jsonl
├── indexes/                         # Git-ignored — tfidf.pkl, bm25.pkl
├── results/
│   ├── runs/                        # One JSONL run file per method
│   ├── metrics/                     # per_query_{method}.csv, summary.csv, summary_comparison.csv
│   └── figures/                     # PR-curve plots
├── notes/
│   └── data_findings.md             # FEVER dataset inspection notes
├── notebooks/                       # Exploration only — no logic lives here
├── backend/
│   └── main.py                      # FastAPI application — all REST endpoints
├── frontend/                        # React 19 + Vite 8 application
│   ├── src/
│   │   ├── App.jsx                  # Main UI component
│   │   ├── App.css                  # Full application styles
│   │   ├── index.css                # Global CSS reset & variables
│   │   └── main.jsx                 # React entry point
│   ├── package.json
│   └── vite.config.js
├── src/fever_ir/                    # Core IR library (installable package)
│   ├── config.py                    # Typed configuration dataclasses + YAML loader
│   ├── data/
│   │   ├── loaders.py               # Claim and JSONL streaming loaders
│   │   ├── wiki_parser.py           # FEVER Wikipedia page parser
│   │   ├── sampling.py              # Deterministic stratified sampling
│   │   └── corpus_builder.py       # Build sentence-level Corpus DataFrame
│   ├── preprocessing/
│   │   ├── text_normalize.py        # Unicode normalization, PTB unescape
│   │   └── tokenize.py              # Lowercase, stopword removal, tokenization
│   ├── retrieval/
│   │   ├── base.py                  # Hit dataclass + BaseRetriever ABC + Corpus Protocol
│   │   ├── tfidf.py                 # TF-IDF + cosine similarity
│   │   ├── bm25.py                  # BM25 Okapi retriever
│   │   ├── dense.py                 # Pinecone vector retriever
│   │   ├── rrf.py                   # Reciprocal Rank Fusion
│   │   └── manager.py              # IndexManager — loads/builds/serves all retrievers
│   ├── evaluation/
│   │   ├── metrics.py               # P@K, R@K, AP, nDCG, MRR, Set Coverage
│   │   ├── runner.py                # Batch evaluation over all methods
│   │   ├── significance.py          # Paired bootstrap + Wilcoxon significance tests
│   │   └── plots.py                 # PR-curve generation
│   └── utils/
│       ├── io.py                    # JSONL streaming read/write, parquet helpers
│       └── logging.py               # Structured logger factory
├── scripts/
│   ├── inspect_fever.py             # Print FEVER dataset statistics (streaming)
│   ├── build_mini_corpus.py         # Build ~5K-sentence mini corpus
│   ├── build_full_corpus.py         # Build full ~150–300K-sentence corpus
│   ├── run_retrieval.py             # Run all four retrievers and save TREC runs
│   ├── evaluate.py                  # Compute metrics from run + qrels files
│   └── compare_methods.py           # Generate comparison table and PR curves
├── tests/                           # Unit and acceptance tests
├── plan.md                          # Master project specification
├── pyproject.toml                   # Build system and project metadata
├── requirements.txt                 # Python dependencies
├── start.sh                         # Unified launcher — macOS/Linux
└── start.bat                        # Unified launcher — Windows
```

---

## Requirements

- **Python** 3.11+
- **Node.js** 18+ and **npm** (for the React frontend)
- A **Pinecone** account with an API key (required only for dense vector retrieval)
- ≥ 15 GB free disk space (wiki-pages ≈ 1.7 GB zipped, several GB unzipped)
- ≥ 8 GB RAM (16 GB recommended for dense embedding)

---

## Setup & Environment

### 1. Clone the repository

```bash
git clone https://github.com/Yazh-7z7/FEVERScope.git
cd FEVERScope
```

### 2. Create and activate a virtual environment

```bash
# macOS / Linux
python3.11 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt

# — or as an editable package (recommended for development) —
pip install -e .
```

### 4. Install frontend dependencies

```bash
npm --prefix frontend install
```

### 5. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` with your [Pinecone](https://www.pinecone.io/) credentials:

```env
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=fever-evidence-index
```

> **Note:** TF-IDF and BM25 retrievers work entirely locally without Pinecone.
> The `dense` retriever and `rrf` hybrid (when dense is included) require a Pinecone
> index pre-populated with FEVER sentence embeddings.

### 6. Run tests

```bash
pytest tests/
```

---

## Data Acquisition

The FEVER dataset must be downloaded manually. From the [FEVER download page](https://fever.ai/dataset/fever.html):

```bash
mkdir -p data/raw

# Claim files
curl -L -o data/raw/train.jsonl           https://fever.ai/download/fever/train.jsonl
curl -L -o data/raw/shared_task_dev.jsonl https://fever.ai/download/fever/shared_task_dev.jsonl

# Wikipedia evidence corpus (~1.7 GB compressed)
curl -L -o data/raw/wiki-pages.zip        https://fever.ai/download/fever/wiki-pages.zip

# Unzip so shards are at data/raw/wiki-pages/wiki-001.jsonl, …
unzip data/raw/wiki-pages.zip -d data/raw/
```

> **Windows:** replace `curl` with `curl.exe`.
> If any link fails, visit https://fever.ai/dataset/fever.html and download manually.

**Dataset splits used:**

| File | Role |
|------|------|
| `train.jsonl` | Hyperparameter tuning + distractor page selection |
| `shared_task_dev.jsonl` | Held-out evaluation (SUPPORTS / REFUTES / NOT ENOUGH INFO) |
| `wiki-pages/*.jsonl` | Wikipedia evidence corpus (~109 shards) |

> `NOT ENOUGH INFO` (NEI) claims have no gold evidence and are excluded from P/R/MAP/nDCG.
> They are stored in a separate file and receive a dedicated analysis.

---

## Building Indexes

### Mini corpus (fast — for development/testing)

Builds a ~5K-sentence corpus from 60 held-out claims + ~400 random Wikipedia pages:

```bash
python scripts/build_mini_corpus.py
```

Outputs: `data/interim/mini_corpus.parquet`, `data/interim/mini_qrels.jsonl`

### Full corpus (for real experiments)

Builds the full ~150–300K-sentence corpus from the evaluation + tuning claim sets:

```bash
python scripts/build_full_corpus.py
```

Outputs: `data/processed/corpus.parquet`, `data/processed/qrels_eval.jsonl`, `data/processed/queries_eval.jsonl`

### Inspect the dataset (optional)

Print dataset statistics without loading large files into memory:

```bash
python scripts/inspect_fever.py
```

### Run retrieval experiments

```bash
# Generate TREC-style run files for all four methods
python scripts/run_retrieval.py

# Compute metrics from run + qrels files
python scripts/evaluate.py

# Generate comparison table and PR-curve plots
python scripts/compare_methods.py
```

TF-IDF and BM25 indexes are automatically persisted to `indexes/tfidf.pkl` and `indexes/bm25.pkl`
and reloaded on subsequent runs. The FastAPI backend also auto-builds them on startup if not found.

---

## Running the App

### Quick start (recommended)

The launcher scripts start both backend and frontend simultaneously:

**macOS / Linux:**
```bash
chmod +x start.sh
./start.sh
```

**Windows:**
```bat
start.bat
```

| Service | URL |
|---------|-----|
| React frontend | http://localhost:5173 |
| FastAPI backend | http://localhost:8000 |
| Swagger UI (API docs) | http://localhost:8000/docs |
| ReDoc | http://localhost:8000/redoc |

Press `Ctrl+C` to stop all services.

### Manual start

```bash
# Terminal 1 — Backend
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — Frontend
npm --prefix frontend run dev
```

---

## API Reference

All endpoints are also documented interactively at **http://localhost:8000/docs**.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/api/health` | Health check — `{"status": "ok", "service": "FEVERScope API"}` |
| `GET`  | `/api/status` | Readiness of all retrieval engines (TF-IDF, BM25, dense, RRF) |
| `GET`  | `/api/benchmarks` | Global benchmark summary table across all methods |
| `GET`  | `/api/sample-claims` | Pre-loaded SUPPORTS/REFUTES claims for UI quick-select |
| `POST` | `/api/search` | Multi-method retrieval + real-time per-claim IR metrics |
| `POST` | `/api/pinecone-config` | Dynamically update Pinecone credentials |
| `POST` | `/api/rebuild-indexes` | Force rebuild of TF-IDF and BM25 indexes from corpus |

### `POST /api/search`

**Request body:**

```json
{
  "query": "The Eiffel Tower is located in Paris.",
  "k": 10,
  "qid": null,
  "gold_doc_ids": null,
  "label": null
}
```

| Field | Type | Description |
|-------|------|-------------|
| `query` | `string` | Factual claim or search query (required) |
| `k` | `int` | Number of top evidence sentences to retrieve (1–100, default 10) |
| `qid` | `int \| null` | Optional: evaluation query ID for ground-truth lookup |
| `gold_doc_ids` | `list[str] \| null` | Optional: known gold `doc_id` list for custom benchmarking |
| `label` | `str \| null` | Optional: ground-truth label (`SUPPORTS`/`REFUTES`) |

**Response structure:**

```json
{
  "query": "The Eiffel Tower is located in Paris.",
  "k": 10,
  "status": {
    "tfidf": { "ready": true, "num_docs": 154320 },
    "bm25":  { "ready": true, "num_docs": 154320 },
    "dense": { "ready": true, "type": "pinecone" },
    "rrf":   { "ready": true, "k": 60 }
  },
  "claim_benchmark": {
    "is_known_claim": true,
    "label": "SUPPORTS",
    "gold_doc_ids": ["Eiffel_Tower::0"],
    "metrics_per_model": {
      "tfidf":  { "precision": 0.3, "recall": 0.9, "ndcg": 0.72, "average_precision": 0.85, "reciprocal_rank": 1.0, "set_coverage": 1.0 },
      "bm25":   { "..." : "..." },
      "dense":  { "..." : "..." },
      "rrf":    { "..." : "..." }
    }
  },
  "global_benchmarks": [ { "Model": "TF-IDF", "p@1": 0.5, "..." : "..." } ],
  "results": {
    "tfidf": [ { "rank": 1, "doc_id": "Eiffel_Tower::0", "score": 3.21, "page_title": "Eiffel Tower", "sent_id": 0, "text": "The Eiffel Tower is a..." } ],
    "bm25":  [ "..." ],
    "dense": [ "..." ],
    "rrf":   [ "..." ]
  }
}
```

---

## CLI Scripts

| Script | Description |
|--------|-------------|
| `scripts/inspect_fever.py` | Print claim counts, evidence statistics, and sample records (streaming — safe for large files) |
| `scripts/build_mini_corpus.py` | Build a small ~5K-sentence corpus for development and testing |
| `scripts/build_full_corpus.py` | Build the full ~150–300K-sentence evaluation corpus |
| `scripts/run_retrieval.py` | Run all four retrievers and save TREC-style run files to `results/runs/` |
| `scripts/evaluate.py` | Compute P@K, R@K, MAP, nDCG metrics from saved run and qrels files |
| `scripts/compare_methods.py` | Generate the 4-method comparison table and PR-curve figures |

---

## Evaluation & Metrics

All retrievers are evaluated on the **same held-out evaluation set** (stratified sample from `shared_task_dev.jsonl`, 50% SUPPORTS / 50% REFUTES).

| Metric | Definition |
|--------|------------|
| **P@K** | `\|rel ∩ top-K\| / K` |
| **R@K** | `\|rel ∩ top-K\| / \|rel\|` |
| **AP@100** | `(1/\|rel\|) · Σᵢ P@i · relᵢ` — MAP = mean AP over queries |
| **nDCG@K** | Binary gain; `DCG = Σ relᵢ / log₂(i+1)` |
| **MRR** | Mean Reciprocal Rank of first gold hit |
| **Set Coverage@K** | 1 if at least one complete evidence set ⊆ top-K (multi-hop FEVER metric) |
| **PR curve** | Macro-average `(R@K, P@K)` for K = 1..100 |

Reported at K ∈ {1, 3, 5, 10, 20}.

**Experimental integrity:**

- BM25 `k1`, `b` and RRF `k` are tuned **only on the tuning set** (from `train.jsonl`).
- Final metrics are computed **only on the eval set** (from `shared_task_dev.jsonl`), which remains untouched until the final frozen run.
- Statistical significance: paired bootstrap (≥ 1000 resamples, 95% CI) and paired Wilcoxon signed-rank test.

**Output files:**

| File | Contents |
|------|----------|
| `results/runs/{method}_eval.jsonl` | Top-100 hits per query in TREC format |
| `results/metrics/per_query_{method}.csv` | Per-query metric values |
| `results/metrics/summary.csv` | Aggregate metrics per method |
| `results/metrics/summary_comparison.csv` | Side-by-side 4-method comparison |
| `results/figures/pr_curve.png` | Precision–Recall curves |

---

## Configuration

All hyperparameters, paths, and seeds live in [`configs/default.yaml`](configs/default.yaml). Nothing is hardcoded in the source.

```yaml
seed: 42

paths:
  raw:       data/raw
  interim:   data/interim
  processed: data/processed
  indexes:   indexes
  results:   results

sampling:
  tune_claims: 500           # from train.jsonl, SUPPORTS/REFUTES 50/50
  eval_claims: 1500          # from shared_task_dev.jsonl, SUPPORTS/REFUTES 50/50
  nei_claims: 200            # analysis only — not in metrics
  distractor_train_claims: 5000
  target_corpus_sentences: 200000
  mini:
    claims_per_label: 30
    nei_claims: 10
    random_pages: 400

preprocessing:
  use_title_augmentation: true   # prepends page title to each sentence
  lowercase: true
  remove_stopwords: true
  stemming: false

retrieval:
  depth: 100          # top-100 retrieved internally; headlines at K ≤ 20
  report_k: [1, 3, 5, 10, 20]

bm25:
  k1: 1.5
  b:  0.75

dense:
  model_name: all-MiniLM-L6-v2
  batch_size: 64
  normalize: true

rrf:
  k: 60    # RRF denominator; tuned on tuning set only
```

---

## Core Abstractions

```python
@dataclass(frozen=True)
class Hit:
    doc_idx: int    # row index 0..N-1 in the Corpus DataFrame
    score: float    # higher = more relevant

class BaseRetriever(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...          # "tfidf" | "bm25" | "dense" | "rrf"

    @abstractmethod
    def build(self, corpus: Corpus | pd.DataFrame) -> None: ...
    def save(self, path: Path | str) -> None: ...
    def load(self, path: Path | str) -> None: ...
    def search(self, query: str, k: int = 100) -> list[Hit]: ...  # sorted descending
```

`Corpus` is a DataFrame with columns: `doc_idx · doc_id · page_title · sent_id · text · text_with_title`.
`doc_id` format: `"{page_title}::{sent_id}"`.

---

## Key Design Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | **Closed-world sub-corpus of ~150–300K sentences**, not full Wikipedia | Full-scale embedding and BM25 scoring is impractical on a student machine |
| D2 | **Corpus = gold pages ∪ distractor pages ∪ random pages** — no retriever used to select distractors | Gold evidence is guaranteed searchable; comparison is unbiased |
| D3 | **`doc_id = "{page_title}::{sent_id}"`** as the universal document identifier | Stable, human-readable, matches FEVER evidence format |
| D4 | **Held-out evaluation** — hyperparameters tuned only on tuning set; final results only on eval set | Prevents test-set leakage |
| D5 | **NEI claims excluded from P/R/MAP/nDCG** | No gold evidence → undefined metrics |
| D6 | **Title augmentation is a config flag and an ablation** | FEVER sentences often lack their subject ("He was born…"); title adds context |
| D7 | **Each method retrieves top-100 internally; RRF fuses full lists** | Fusing only top-10 lists artificially caps recall |
| D8 | **Retrieval and evaluation decoupled via TREC run/qrels files** | Metrics can be recomputed without re-running retrieval; results are auditable |
| D9 | **Per-query metrics saved; significance via bootstrap + Wilcoxon** | "Better" must be statistically better |
| D10 | **Lexical methods use normalized/tokenized text; dense uses raw (title-augmented) text** | Each method gets the input it works best with; fairness comes from identical documents |

---

## Known Pitfalls

- **Positional index vs `sent_id` mismatch:** FEVER `lines` fields contain empty lines — always read `sent_id` from the line content, never from list position.
- **Non-deterministic `hash()`:** Python's built-in `hash()` is salted per process. Use `hashlib` (MD5/SHA) for reproducible sampling.
- **Title matching:** apply Unicode NFC normalization; handle PTB escapes (`-LRB-`, `-RRB-`, `-COLON-`, `-LSB-`, `-RSB-`).
- **Empty gold sentences:** a gold `(page, sent_id)` that exists but has empty text — drop the claim and log it.
- **Tuning on eval queries** is forbidden (D4). Never touch `shared_task_dev.jsonl` until final frozen run.
- **Choosing distractors with a retriever** is forbidden (D2). Use only random sampling or gold page membership.
- **Low P@10 is expected** when a claim has 1–2 gold sentences (ceiling = min(g, 10)/10). Always interpret precision relative to that ceiling, and lean on R@K, nDCG, and MAP.
- **`NOT ENOUGH INFO` claims** look like `[annotation_id, null, null, null]` in evidence — parse defensively.

---

## Roadmap

- [x] **Phase 1** — Environment, data acquisition, inspection, mini-corpus scaffolding
- [ ] **Phase 2** — Full corpus + evaluation sets; 100% gold coverage; `manifest.json`
- [ ] **Phase 3** — Evaluation core: metrics, runner, PR-curve, random + oracle baselines
- [ ] **Phase 4** — Preprocessing pipeline + TF-IDF + BM25 retrievers with timing
- [ ] **Phase 5** — Dense retriever: embed corpus with `all-MiniLM-L6-v2`, store in Pinecone
- [ ] **Phase 6** — RRF hybrid fusion; tune `k` on tuning set only
- [ ] **Phase 7** — Full experiments + analysis: 4-method comparison, slices, ablations, PR graphs
- [ ] **Phase 8** — FastAPI + React interactive app *(in progress)*
- [ ] **Phase 9** — Report and presentation

---

## Contributing

This is an academic project. If you spot a bug or have a suggestion, feel free to open an issue or pull request at [github.com/Yazh-7z7/FEVERScope](https://github.com/Yazh-7z7/FEVERScope).

---

## License

Built for academic research and learning using the [FEVER dataset](https://fever.ai/).

---

*FEVERScope — Python 3.11 · FastAPI · React 19 · Vite 8 · scikit-learn · rank-bm25 · sentence-transformers · Pinecone*
