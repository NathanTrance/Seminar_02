# Seminar 02 — RAG for Malicious PyPI Package Detection

Course project: **does retrieval strategy matter for LLM-based detection of
malicious PyPI packages?** We run a controlled comparison of seven pipelines
(E0–E6) on the EASE 2025 dataset with a fixed generator
(Llama-3.1-8B-Instruct), a fixed code-only knowledge base, and a fixed
prompt — only the retrieval strategy varies.

This repository has two parts:

| Path | Contents |
|------|----------|
| `code/` | The experiment codebase (`ragsec` package) + all raw results |
| `report/` | The course report in the EASE (acmart) LaTeX template |

## Headline results

| Method | Mal. F1 | Mal. Recall | FPR |
|--------|---------|-------------|-----|
| E0 — No RAG (baseline) | 0.888 | 0.798 | 0.0% |
| E1 — Dense RAG (FAISS + Qwen3-Embedding-4B) | **0.972** | **0.972** | 0.9% |
| E2 — BM25 RAG | 0.968 | 0.972 | 1.2% |
| E3 — Hybrid RAG (RRF) | 0.970 | 0.964 | 0.8% |
| E4 — Behavior RAG (AST summaries) | 0.878 | 0.782 | 0.0% |
| E5 — Contrastive RAG (mal + ben pools) | 0.971 | 0.948 | **0.1%** |
| E6 — Behavior + Contrastive | 0.881 | 0.788 | 0.0% |

Takeaways:

- Any code-level retrieval is a big win (recall +17 pts over no RAG).
- Plain BM25 ≈ dense retrieval (within 0.4 F1) — sparse is the cheap default.
- Contrastive retrieval cuts false positives to 0.1% at 0.948 recall.
- AST behavior-summary queries actively hurt (F1 0.878) — lossy representation.
- 5 typosquats (e.g., `colorama` metadata clones) evade *all* methods;
  `setup.py`-only evidence can never catch them.

---

## 1. Experiments (`code/`)

### Setup

```bash
cd code
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Environment

Copy `.env.example` to `.env` (or export before running):

| Variable | Example | Required |
|----------|---------|----------|
| `OPENAI_BASE_URL` | `http://localhost:8000/v1` | Yes |
| `OPENAI_API_KEY` | `dummy` | Yes |
| `EMBEDDING_BASE_URL` | `http://localhost:8001/v1` | dense/hybrid/behavior/contrastive |
| `EMBEDDING_API_KEY` | `dummy` | dense/hybrid/behavior/contrastive |
| `EMBEDDING_MODEL` | `Qwen3-Embedding-4B` | dense/hybrid/behavior/contrastive |
| `SSL_VERIFY` | `false` | self-signed certs |

### Dataset

The dataset lives in `code/mal-LLM/RQ_experiments/data/` (replication package
of the EASE 2025 paper, built on Datadog's malicious-software-packages
dataset). The `EaseRagDataset` adapter reads the four pre-split JSON files:

- `train_malicious_packages_final.json` (1,158)
- `train_benign_packages_final.json` (3,005) → knowledge base (4,163)
- `test_malicious_packages_final.json` (276)
- `test_benign_packages_final.json` (753) → test set (1,029; 1,020 usable
  after removing 9 samples with `"Not Available"` code)

### Run experiments

One script per method (also `run_scripts/run_all.sh` for everything):

```bash
# E0 — zero-shot baseline (no retrieval)
bash run_scripts/run_e0_no_rag.sh

# E1 — dense retrieval (needs embedding endpoint)
bash run_scripts/run_e1_dense.sh

# E2 — BM25 (local, no embeddings)
bash run_scripts/run_e2_bm25.sh

# E3 — hybrid (dense + BM25, reciprocal rank fusion)
bash run_scripts/run_e3_hybrid.sh

# E4 — behavior-summary retrieval
bash run_scripts/run_e4_behavior.sh

# E5 — contrastive (malicious + benign pools)
bash run_scripts/run_e5_contrastive.sh

# E6 — behavior + contrastive
bash run_scripts/run_e6_behavior_contrastive.sh
```

Each script sets its own env vars (endpoints, `SSL_VERIFY=false`, `PYTHONPATH=code`)
— edit the URLs for your serving setup.

Alternatively, direct CLI:

```bash
python scripts/run_experiment.py --dataset ease_rag \
  --model Llama-3.1-8B-Instruct --method dense --split test \
  --max-tokens 1024 --top-k 3
```

### Evaluate & inspect

```bash
# Recompute metrics for one result file (or pass --output to save JSON)
python scripts/evaluate.py results/raw/dense_Llama-3.1-8B-Instruct_ease_rag.jsonl

# Rebuild the error-analysis dashboard (HTML, clickable per method)
python scripts/dashboard.py   # → dashboard.html
```

### Outputs

- `results/raw/<method>_<model>_ease_rag.jsonl` — one JSON row per sample
  (verdict, confidence, cited behaviors, retrieved IDs/scores, raw response,
  latency, tokens, parse status)
- `results/metrics/…json` — aggregated metrics (per-class P/R/F1, macro F1,
  balanced accuracy, FPR/FNR, coverage, faithfulness)
- `data/indexes/` — persisted BM25 / FAISS indexes (auto-built on first run)
- `data/cache/cache.db` — SQLite cache of LLM responses, embeddings,
  retrieval results

### Repository layout (code/)

```
code/
├── configs/                 # models / datasets / experiments YAML
├── run_scripts/             # one-click run scripts (E0–E6, run_all)
├── scripts/                 # run_experiment, evaluate, dashboard, ...
├── src/ragsec/
│   ├── clients/             # OpenAI-compatible LLM + embedding clients
│   ├── datasets/            # unified Sample schema + EaseRagDataset adapter
│   ├── static/              # AST feature extraction, behavior flags
│   ├── retrieval/           # dense (FAISS), bm25, hybrid (RRF), contrastive
│   ├── prompts/             # prompt templates + JSON schema
│   ├── evaluation/          # classification, faithfulness, retrieval metrics
│   └── experiments/         # runner with caching + resume support
├── results/                 # raw JSONL + metrics
└── mal-LLM/                 # upstream replication repo (data only, read-only)
```

---

## 2. Report (`report/`)

EASE (acmart sigconf) LaTeX template with the full draft:

```text
report/
├── scripts/recompute_metrics.py   # stdlib-only, recomputes every metric
│                                  # from results/raw (no deps needed)
├── metrics.json                   # authoritative per-method numbers
├── tables.tex                     # generated LaTeX table body
└── ease-template/
    ├── ease-submission.tex        # anonymous review version (compiles)
    ├── ease-camera-ready.tex      # author version (fill in your name)
    ├── Chapters/                  # 01-introduction … 06-conclusion
    ├── references.bib             # citations (2 entries need author fill-in)
    └── Figures/                   # TikZ pipeline figure (inline in 03)
```

Compile on Windows (TeX Live):

```bash
cd report/ease-template
pdflatex -interaction=nonstopmode ease-submission.tex
bibtex ease-submission
pdflatex -interaction=nonstopmode ease-submission.tex
pdflatex -interaction=nonstopmode ease-submission.tex
```

(Or use `compile.sh` on Linux/macOS, or Overleaf.)

Regenerating report numbers after new runs:

```bash
cd report
python scripts/recompute_metrics.py   # updates metrics.json + tables.tex
```

---

## License & safety notes

- Treat all malicious-package samples as hostile; never execute, install,
  or import them. Analysis is static only (`ast` + text).
- The dataset is the EASE 2025 replication package / Datadog dataset; check
  their licenses before redistributing raw data.
- This is course work. Two bibliography entries (`ease2025rag`, `lamps2025`)
  have placeholder author fields — fill them from the published versions
  before any external submission.
