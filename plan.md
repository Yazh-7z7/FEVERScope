# plan.md — FEVER Evidence Retrieval: TF-IDF vs BM25 vs Sentence-Transformers vs RRF Hybrid

> **Audience:** the AI coding agent working in this workspace (Google Antigravity) and the human student who is *learning* while building.
> **Read this file fully before every task.** It is the source of truth for scope, design decisions, interfaces, and acceptance criteria.
> If a request conflicts with this file, stop and ask the human.

**Current phase:** `PHASE 1`  *(human updates this line when a phase is approved as complete)*

---

## 0. Agent operating protocol (non-negotiable)

1. **Work on exactly one component per task, within the current phase only.** Never start the next phase without explicit human approval.
2. **Planning first.** For every task, produce an implementation plan artifact and wait for human review before writing code.
3. **Never fabricate** data, example outputs, or metrics. Every number reported anywhere (README, app, report) must come from a file in `results/` that was produced by code in this repo.
4. **Never open, `cat`, or print whole files under `data/`** (multi-GB). Use streaming iteration, `head -n`, or sampling.
5. **The human runs the large downloads** (Section 5, Step 3). Do not download `wiki-pages.zip` yourself.
6. **Learning mode:** after each task, append a 5-line plain-English explanation: what was built, why this design, what to look at in the output, one thing that could go wrong, one question the human should be able to answer.
7. **Tests are part of the task.** Each module ships with pytest tests using tiny, hand-verifiable fixtures. Report the test output.
8. **Small diffs.** Prefer small, reviewable changes. Do not refactor unrelated code.
9. **Determinism.** Fixed seeds from config. Use `hashlib` (not Python's built-in `hash()`, which is salted per process) wherever a stable hash is needed.
10. **No hidden constants.** All paths, seeds, sample sizes, and hyperparameters come from `configs/default.yaml`.

---

## 1. Project goal

Build a working system that retrieves and ranks **evidence sentences** for factual claims using the **FEVER** dataset and its Wikipedia corpus.

Four retrieval approaches, all searching the **same sentence-level corpus** with the **same evaluation queries**:

| ID | Method | Library |
|---|---|---|
| `tfidf` | TF-IDF + cosine similarity | scikit-learn |
| `bm25` | BM25 (Okapi) | rank-bm25 |
| `dense` | Sentence-Transformer embeddings + cosine similarity | sentence-transformers |
| `rrf` | Reciprocal Rank Fusion of the three above | own implementation |

Evaluate with **Precision@K, Recall@K, MAP, nDCG**, plus a **Precision–Recall graph**, using real experimental results. Research question: *Can hybrid (RRF) retrieval find relevant evidence more effectively than any individual method?* Expose results through an interactive **FastAPI + React** web application.

**Tech stack:** Python 3.11, pandas, NumPy, scikit-learn, rank-bm25, sentence-transformers, Matplotlib, FastAPI, uvicorn, React 19 (Vite 8), Pinecone.
Supporting: scipy (stats), pyarrow (parquet), pyyaml, tqdm, pytest, python-dotenv.

---

## 2. Key design decisions (do not change without human approval)

| # | Decision | Rationale |
|---|---|---|
| D1 | **Closed-world sub-corpus of ~150–300K sentences**, not full Wikipedia (~5.4M pages) | Full-scale embedding and `rank-bm25` scoring is impractical on a student machine. |
| D2 | **Corpus = gold pages of eval claims ∪ gold pages of tuning claims ∪ gold pages of extra *train* claims (topical distractors) ∪ randomly sampled pages** | Gold evidence is guaranteed searchable; distractors are realistic. **No retriever may be used to choose distractors** (would bias the comparison). |
| D3 | **Relevant set for a claim = union of all gold sentences across all its evidence sets**, identified as `doc_id = "{page_title}::{sent_id}"` | Standard IR framing. Supplementary FEVER-style metric: *set coverage@K* (see §6). |
| D4 | **Held-out evaluation:** hyperparameters (BM25 `k1`,`b`; RRF `k`; preprocessing choices) are tuned only on the **tuning set** (sampled from `train.jsonl`). Final numbers are reported only on the **eval set** (sampled from `shared_task_dev.jsonl`), which is untouched until the final frozen run. | Avoids tuning on the test queries. |
| D5 | **NOT ENOUGH INFO (NEI) claims are excluded** from P/R/MAP/nDCG (no gold evidence → undefined). They live in a separate file and get a separate analysis. | Metric validity. |
| D6 | **Title-augmented sentence text is a config flag** (`text_with_title = "{title with spaces}: {sentence}"`) and an **ablation**. | FEVER sentences often lack the subject ("He was born…"). |
| D7 | **Every method retrieves top-100 internally; headline metrics are reported at K ≤ 10** (also K=20). RRF fuses the three top-100 lists. A "fuse top-10 only" variant is an ablation. | Fusing only top-10 lists artificially caps recall. |
| D8 | **Retrieval and evaluation are decoupled** via TREC-style *run* and *qrels* files (§4.3). | Metrics can be recomputed without re-running retrieval; results are auditable. |
| D9 | **Per-query metrics are saved**; significance via paired tests + bootstrap CIs. | "Better" must mean statistically better. |
| D10 | **Different text pipelines per method, same corpus rows.** Lexical methods use normalized/tokenized text; the dense model receives raw (title-augmented if flag on) text. All share `doc_idx` ordering. | Each method gets the input it works best with; fairness comes from identical documents and queries. |

---

## 3. FEVER data format (verify every point in Phase 1 — do not assume)

Source: `https://fever.ai/download/fever/` — files: `train.jsonl`, `shared_task_dev.jsonl`, `wiki-pages.zip`.
Use `train.jsonl` (tuning + distractors) and `shared_task_dev.jsonl` (held-out eval). **Do not use** `shared_task_test.jsonl` (no labels/evidence).
**Do not use `datasets.load_dataset("fever/fever")`** — it relies on a loading script that recent `datasets` versions no longer support. Use the raw files.

### 3.1 Claim record (JSONL, one object per line)
```json
{"id": 123, "verifiable": "VERIFIABLE", "label": "SUPPORTS",
 "claim": "...",
 "evidence": [ [[annotation_id, evidence_id, "Page_Title", 0]],
               [[annotation_id, evidence_id, "Page_Title", 3], [annotation_id, evidence_id, "Other_Page", 1]] ]}
```
- `label` ∈ {`SUPPORTS`, `REFUTES`, `NOT ENOUGH INFO`}.
- `evidence` = list of **evidence sets** (alternative justifications). Each set = list of `[annotation_id, evidence_id, page_title, sentence_id]`.
- A set with >1 entry is **multi-hop** (sentences needed together, possibly across pages).
- For NEI: entries look like `[annotation_id, null, null, null]` → **no gold evidence**.

### 3.2 Wikipedia page record (`wiki-pages/wiki-XXX.jsonl`, expected ~109 shards — verify)
```json
{"id": "Page_Title", "text": "...", "lines": "0\tFirst sentence.\tlink_text\tlink_target\n1\tSecond sentence...."}
```
- Use the `lines` field. Split on `\n`, then each line on `\t`: `[sent_id, text, *hyperlink fields]`. Ignore hyperlink fields.
- **Take `sent_id` from the line itself**, never from list position (empty lines exist).
- Text/titles contain PTB escapes (`-LRB-`, `-RRB-`, `-LSB-`, `-RSB-`, `-COLON-`, …). **Match titles on the raw string**; normalize only for display/indexing (`_` → space, un-escape brackets).
- Normalize both sides with Unicode NFC before matching titles (known mismatch source). Log unresolved evidence.

---

## 4. Architecture and interfaces

### 4.1 Project structure
```
FEVERScope/
├── plan.md                          # this file
├── .agent/
│   ├── rules/project_rules.md       # short rules (copy of §0), optional
│   └── workflows/                   # saved prompts (e.g. new-component, run-tests)
├── configs/default.yaml
├── .env.example                     # Pinecone API key template
├── data/                            # git-ignored
│   ├── raw/                         # train.jsonl, shared_task_dev.jsonl, wiki-pages/
│   ├── interim/                     # extracted pages, mini corpus
│   └── processed/                   # corpus.parquet, qrels.jsonl, queries files, manifest.json
├── indexes/                         # tfidf.pkl, bm25.pkl (git-ignored)
├── results/
│   ├── runs/                        # one run file per method
│   ├── metrics/                     # per-query and aggregate CSVs
│   └── figures/
├── notes/data_findings.md           # human-written findings
├── backend/
│   └── main.py                      # FastAPI application — all REST endpoints
├── frontend/                        # React 19 + Vite 8 application
│   ├── src/
│   │   ├── App.jsx                  # Main UI component
│   │   └── App.css / index.css      # Styling
│   └── package.json
├── src/fever_ir/
│   ├── config.py
│   ├── data/            # loaders.py, wiki_parser.py, corpus_builder.py, sampling.py
│   ├── preprocessing/   # text_normalize.py, tokenize.py
│   ├── retrieval/       # base.py, tfidf.py, bm25.py, dense.py, rrf.py, manager.py
│   ├── evaluation/      # metrics.py, runner.py, significance.py, plots.py
│   └── utils/           # io.py, logging.py
├── scripts/             # thin CLI entry points only
├── tests/
├── notebooks/           # exploration only; NO logic lives here
├── start.sh             # Unified launcher (macOS/Linux)
├── start.bat            # Unified launcher (Windows)
├── pyproject.toml
├── requirements.txt
└── README.md
```

### 4.2 Core abstractions
```python
@dataclass(frozen=True)
class Hit:
    doc_idx: int
    score: float

class BaseRetriever(ABC):
    name: str                                   # "tfidf" | "bm25" | "dense" | "rrf"
    def build(self, corpus: Corpus) -> None: ...
    def save(self, path: Path) -> None: ...
    def load(self, path: Path) -> None: ...
    def search(self, query: str, k: int) -> list[Hit]: ...   # sorted by descending score
```
- `Corpus` is the **single source of truth**: a DataFrame with columns
  `doc_idx` (0..N-1, no gaps), `doc_id`, `page_title`, `sent_id`, `text`, `text_with_title`.
- All retrievers index the same rows in the same order.
- **RRF consumes only ranked lists** (it must not know which method produced them):
  `score(d) = Σ_m 1 / (k + rank_m(d))`, ranks 1-based, default `k = 60`, tuned on the tuning set only. Ties broken deterministically (by `doc_idx`).

### 4.3 File formats
**Qrels** (`data/processed/qrels_{split}.jsonl`, one line per claim):
```json
{"qid": 123, "label": "SUPPORTS", "gold_doc_ids": ["Page::0", "Page::3"], "evidence_sets": [["Page::0"], ["Page::3", "Other::1"]]}
```
**Run** (`results/runs/{method}_{split}.jsonl`, one line per claim):
```json
{"qid": 123, "method": "bm25", "hits": [{"doc_id": "Page::0", "rank": 1, "score": 12.3}, ...]}
```
Each run stores top-100 hits per query. Evaluation reads only qrels + runs.

### 4.4 Config sketch (`configs/default.yaml`) — extend as needed, never hard-code
```yaml
seed: 42
paths: {raw: data/raw, interim: data/interim, processed: data/processed, indexes: indexes, results: results}
sampling:
  tune_claims: 500          # from train.jsonl, SUPPORTS/REFUTES stratified 50/50
  eval_claims: 1500         # from shared_task_dev.jsonl, SUPPORTS/REFUTES stratified 50/50
  nei_claims: 200           # from shared_task_dev.jsonl, analysis only
  distractor_train_claims: 5000   # gold pages only
  target_corpus_sentences: 200000 # random pages added until this is approximately reached
  mini: {claims_per_label: 30, nei_claims: 10, random_pages: 400}
preprocessing: {use_title_augmentation: true, lowercase: true, remove_stopwords: true, stemming: false}
retrieval: {depth: 100, report_k: [1, 3, 5, 10, 20]}
bm25: {k1: 1.5, b: 0.75}
dense: {model_name: all-MiniLM-L6-v2, batch_size: 64, normalize: true}
rrf: {k: 60}
```

---

## 5. PHASE 1 — Environment, data acquisition, inspection, mini-corpus  ← CURRENT

**Goal:** environment works, data downloaded and *understood*, and a ~5K-sentence mini-corpus resolves 100% of gold evidence for its claims.

### Step 1 — Prerequisites (human)
Python 3.11; ≥15 GB free disk (zip ≈ 1.7 GB, several GB unzipped); 8 GB RAM minimum (16 GB comfortable).

### Step 2 — Environment (human + agent)
```bash
# Windows PowerShell
py -3.11 -m venv .venv ; .\.venv\Scripts\Activate.ps1
# macOS / Linux
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Agent: create `.gitignore` (`data/`, `indexes/`, `.venv/`, `results/runs/`, `__pycache__/`), `requirements.txt`, `configs/default.yaml`, and the folder skeleton from §4.1 (empty modules with docstrings).

### Step 3 — Download (human runs these; Windows: use `curl.exe`)
```bash
curl -L -o data/raw/train.jsonl           https://fever.ai/download/fever/train.jsonl
curl -L -o data/raw/shared_task_dev.jsonl https://fever.ai/download/fever/shared_task_dev.jsonl
curl -L -o data/raw/wiki-pages.zip        https://fever.ai/download/fever/wiki-pages.zip
```
Unzip so shards are at `data/raw/wiki-pages/wiki-001.jsonl`, … . If a link fails, use `https://fever.ai/dataset/fever.html`.

### Step 4 — Inspection (agent writes `scripts/inspect_fever.py`; streaming only)
The script must **print, not assume**:
1. Claim counts per label for `train.jsonl` and `shared_task_dev.jsonl`; whether dev is balanced.
2. Three raw claims per label (`json.dumps(indent=2)`).
3. Evidence statistics for SUPPORTS/REFUTES: distribution of #evidence sets per claim, #sentences per set, #distinct pages per set, mean #gold sentences per claim (flattened, deduplicated).
4. The first raw page record from `wiki-001.jsonl` next to its parsed `lines`.
5. For one shard: pages, total sentences, empty sentences, count of titles containing PTB escapes.

Human then writes `notes/data_findings.md` answering: fraction of claims needing multi-sentence/multi-page evidence; mean gold sentences per claim; implications for the **P@10 ceiling** (if a claim has g gold sentences, P@10 ≤ min(g,10)/10).

### Step 5 — Mini-corpus (agent implements; modules under `src/fever_ir/data/`)
- **Claims (mini):** 60 from `shared_task_dev` (30 SUPPORTS + 30 REFUTES, seeded) + 10 NEI (separate file).
- **Pages:** gold pages of the 60 claims + ~400 random pages.
- **Extraction:** a **single streaming pass** over all shards. Keep a page if its title ∈ wanted set, or `int(md5(title).hexdigest(), 16) % 1000 < threshold` for random distractors.
- **Explode to sentence rows** with columns defined in §4.2. Drop empty-text sentences only after verifying no gold sentence is empty.
- **Save:** `data/interim/mini_corpus.parquet`, `data/interim/mini_qrels.jsonl`, `data/interim/mini_nei_claims.jsonl`, plus a log of dropped claims with reasons.

### Step 6 — Acceptance tests (`tests/test_data.py`)
1. `doc_id` unique; `doc_idx` is exactly 0..N-1.
2. For every retained claim, **every gold `(page, sent_id)` exists in the corpus with non-empty text**. Failing claims are dropped and logged (count + reason).
3. No NEI claim appears in qrels.
4. Two runs with the same seed produce byte-identical outputs (compare hashes).
5. The parser reproduces the exact sentence text seen in the raw record from Step 4.
6. Title matching works with NFC normalization (include a fixture with an accented title).

### Phase 1 deliverables
`requirements.txt`, `configs/default.yaml`, `.gitignore`, folder skeleton, `scripts/inspect_fever.py`, `src/fever_ir/data/{loaders,wiki_parser,corpus_builder,sampling}.py`, `scripts/build_mini_corpus.py`, `tests/test_data.py`, `notes/data_findings.md`, first git commit.

**Phase 1 is done when:** all Step 6 tests pass, inspection output has been read by the human, and `notes/data_findings.md` exists.

---

## 6. Evaluation specification (used from Phase 3 onward)

**Eligible queries:** SUPPORTS and REFUTES claims whose gold sentences are all present in the corpus. NEI excluded (D5).

| Metric | Definition (per query; macro-average over queries) |
|---|---|
| **P@K** | `|rel ∩ top-K| / K` |
| **R@K** | `|rel ∩ top-K| / |rel|` |
| **AP@100** | `(1/|rel|) · Σ_{i=1..100} P@i · rel_i` (denominator = |rel|, not #retrieved-relevant); **MAP** = mean AP |
| **nDCG@K** | binary gain; `DCG = Σ rel_i / log2(i+1)`; ideal DCG has `min(|rel|, K)` ones |
| **MRR**, **Hit@K** | supplementary |
| **Set coverage@K** | 1 if at least one *complete* evidence set ⊆ top-K (FEVER-style; matters for multi-hop) |
| **PR curve** | macro-average of (R@K, P@K) for K = 1..100 (K-sweep); optionally also 11-point interpolated precision |

Report K ∈ {1, 3, 5, 10, 20}. Save **per-query** metrics to `results/metrics/per_query_{method}.csv` and aggregates to `results/metrics/summary.csv`.

**Validation of the yardstick (Phase 3, before any real retriever):**
- Unit tests on toy rankings with hand-computed expected values for every metric.
- **Oracle baseline** (returns gold first) → P/R/nDCG ≈ their theoretical maxima.
- **Random baseline** → recall@K ≈ `K·|rel|/N` in expectation; verify empirically.
- Optional dev-only cross-check against an independent library (e.g. `ranx` or `pytrec_eval`) in a test, *not* a runtime dependency.

**Statistics:** paired bootstrap (seeded, ≥1000 resamples) 95% CIs on per-query metrics; paired Wilcoxon (scipy) for hybrid vs best single method. Report effect sizes, not just p-values.

**Analyses (Phase 7)** — each must be produced by code and saved:
- 4-method comparison table + PR graph (headline).
- Slices: SUPPORTS vs REFUTES; single- vs multi-sentence evidence; claim length bins.
- Top-10 overlap (Jaccard) between methods; counts of queries where *only* one method finds gold (complementarity).
- Ablations: title augmentation on/off; stopwords on/off; RRF depth 10 vs 100; RRF `k` sensitivity (tuning set).
- Corpus-size scaling (e.g. 50K / 100K / 200K sentences).
- Per-query hybrid wins/losses with 5–10 concrete examples.
- NEI analysis: distribution of top-1 scores for NEI vs SUPPORTS/REFUTES claims.

**Honesty rules:** state hypotheses, never numbers, before running. If hybrid does not beat the best single method on a metric, report that. Document limitations: sub-corpus results are not comparable with full-Wikipedia FEVER numbers; FEVER annotations are incomplete (unannotated valid evidence lowers measured precision).

---

## 7. Roadmap and phase gates

- [ ] **Phase 1 — Setup, acquisition, inspection, mini-corpus** *(§5)*
- [ ] **Phase 2 — Full corpus + evaluation sets.** Tuning (≈500 train), eval (≈1,500 dev), NEI (≈200 dev), qrels, corpus per D2, `manifest.json` documenting seeds, sizes, filters, dropped claims. *Gate:* 100% of eval gold sentences in the corpus; eval ∩ tuning claim IDs = ∅.
- [ ] **Phase 3 — Evaluation core.** `metrics.py`, `runner.py`, PR-curve code, random + oracle baselines. *Gate:* all metric tests pass (§6).
- [ ] **Phase 4 — Preprocessing + TF-IDF + BM25.** Shared tokenizer; saved indexes; both behind `BaseRetriever`; mini-corpus sanity checks, then full corpus with timing logged. Note: `rank-bm25` scores in Python loops (~O(N·|query|)); log per-query latency and cap corpus size accordingly.
- [ ] **Phase 5 — Dense retriever.** Embed corpus with `all-MiniLM-L6-v2`, upload to Pinecone; query via Pinecone SDK. If local embedding is too slow, embed on Colab/Kaggle and upload.
- [ ] **Phase 6 — RRF.** Fusion of three top-100 lists; tune `k` on tuning set only. *Gate:* fusion unit test with hand-computed scores.
- [ ] **Phase 7 — Experiments + analysis.** Tune on tuning set → **freeze** config → one run on eval set → all analyses in §6. One command regenerates every table/figure.
- [ ] **Phase 8 — FastAPI + React app.** FastAPI backend (`backend/main.py`) exposes REST endpoints; React frontend (`frontend/`) shows side-by-side top-10 per method; gold evidence highlighted for known eval claims; benchmark metrics loaded from `results/` (never hard-coded numbers); NEI demo.
- [ ] **Phase 9 — Report + presentation.** Methodology, results, limitations, reproducibility notes; every number cites a file in `results/`.

---

## 8. Known pitfalls checklist

- Positional index vs `sent_id` mismatch when parsing `lines`.
- Python's `hash()` is salted → non-reproducible sampling. Use `hashlib`.
- Title matching: raw string vs normalized; NFC normalization; PTB escapes.
- Gold sentence exists ǐ̧n the page but is empty → drop the claim and log.
- Tuning on eval queries (leakage) → forbidden by D4.
- Choosing distractors with a retriever → forbidden by D2.
- Different corpora or query sets per method → forbidden; one `Corpus`, one qrels file per split.
- Opening multi-GB files in the editor or printing them → freezes the IDE and bloats agent context.
- Low P@10 is expected when claims have 1–2 gold sentences (ceiling effect); always interpret precision against that ceiling, and lean on Recall@K, nDCG, and MAP.

---

## 9. Requirements (`requirements.txt`)
```
pandas
numpy
scipy
scikit-learn
rank-bm25
sentence-transformers
matplotlib
fastapi
uvicorn[standard]
pinecone-client
python-dotenv
pyarrow
pyyaml
tqdm
pytest
```
Frontend dependencies are managed by `npm` via `frontend/package.json`.
After everything works: `pip freeze > requirements.lock.txt`.

---

## 10. First prompt to give the agent (Planning mode)

> Read `plan.md`. We are in Phase 1. Task 1 only: scaffold the project structure from §4.1 (folders, empty modules with docstrings, `configs/default.yaml` per §4.4, `.gitignore`, `requirements.txt`, README stub). Do not download data and do not implement anything beyond scaffolding. Show me the implementation plan first.

Then, one task per prompt: (2) `scripts/inspect_fever.py` per Step 4 → (3) `wiki_parser.py` + tests → (4) `sampling.py` + `corpus_builder.py` + `build_mini_corpus.py` → (5) `tests/test_data.py` per Step 6.