# RAGsec — Training-Free RAG for Malicious PyPI Package Detection

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Environment

Copy and fill:

```bash
cp .env.example .env
```

Key vars:

| Variable | Example | Required |
|----------|---------|----------|
| `OPENAI_BASE_URL` | `http://localhost:8000/v1` | Yes |
| `OPENAI_API_KEY` | `dummy` | Yes |
| `EMBEDDING_BASE_URL` | `http://localhost:8001/v1` | Only for dense/behavior/contrastive |
| `EMBEDDING_API_KEY` | `dummy` | Only for dense etc. |
| `EMBEDDING_MODEL` | `Qwen3-4B-Embedding` | Only for dense etc. |
| `SSL_VERIFY` | `false` | If using self-signed certs |

Quick setup with your serving endpoints:

```bash
export OPENAI_BASE_URL='<your-chat-endpoint>/v1'
export OPENAI_API_KEY='dummy'
export EMBEDDING_BASE_URL='<your-embedding-endpoint>/v1'
export EMBEDDING_API_KEY='dummy'
export EMBEDDING_MODEL='Qwen3-4B-Embedding'
export SSL_VERIFY='false'
```

## Model config

Set your model IDs in `configs/models.yaml`:

```yaml
models:
  - id: llama31_8b
    api_model: meta-llama/Llama-3.1-8B-Instruct
  - id: qwen3_vl_4b
    api_model: Qwen3-VL-4B-Instruct
```

Use the `id` (e.g. `llama31_8b`) as `--model` in commands; the script sends `api_model` to the endpoint.

## Quick start — smoke test

```bash
python scripts/run_experiment.py \
  --dataset ease_rag \
  --model llama31_8b \
  --method no_rag \
  --split test \
  --limit 5
```

## Direct JEV decision demo

The JEV demo sends source text directly to Jev through OpenRouter without
retrieval or another language model. Validate the two synthetic fixtures without
network access:

```bash
python scripts/run_jev_demo.py --dry-run
```

For a bounded two-request live demo:

```bash
export OPENROUTER_API_KEY='sk-or-<personal-key>'
python scripts/run_jev_demo.py --live
```

Or use the bash launcher. With no arguments it securely prompts for a key when
needed and runs one live synthetic sample:

```bash
./run_scripts/run_jev_demo.sh
```

It can also load `OPENROUTER_API_KEY` from the ignored `code/.env` file.
Arguments are forwarded to the Python CLI, for example:

```bash
./run_scripts/run_jev_demo.sh --dry-run --limit 1
```

See `docs/jev_demo.md` for the request contract, replay command, and safety
limitations.

## Experiments

### Full run (all 1029 test samples)

```bash
# E0 — No RAG baseline
python scripts/run_experiment.py --dataset ease_rag --model llama31_8b --method no_rag --split test

# E2 — BM25 RAG (no embeddings needed)
python scripts/run_experiment.py --dataset ease_rag --model llama31_8b --method bm25 --split test

# E1 — Dense RAG (needs embedding endpoint)
python scripts/run_experiment.py --dataset ease_rag --model llama31_8b --method dense --split test

# E3 — Hybrid RAG
python scripts/run_experiment.py --dataset ease_rag --model llama31_8b --method hybrid --split test

# E4 — Behavior RAG
python scripts/run_experiment.py --dataset ease_rag --model llama31_8b --method behavior --split test

# E5 — Contrastive RAG
python scripts/run_experiment.py --dataset ease_rag --model llama31_8b --method contrastive --split test

# E6 — Behavior + Contrastive RAG
python scripts/run_experiment.py --dataset ease_rag --model llama31_8b --method behavior_contrastive --split test
```

### Top-k ablation (Stage A)

```bash
python scripts/run_experiment.py --dataset ease_rag --model llama31_8b --method dense --top-k 1
python scripts/run_experiment.py --dataset ease_rag --model llama31_8b --method dense --top-k 3
python scripts/run_experiment.py --dataset ease_rag --model llama31_8b --method dense --top-k 5
```

## Build indexes (optional, auto-built on first run)

```bash
python scripts/build_indexes.py --dataset ease_rag
```

## Evaluate results

```bash
python scripts/evaluate.py results/raw/no_rag_llama31_8b_ease_rag.jsonl
```

## Outputs

| Path | Contents |
|------|----------|
| `results/raw/*.jsonl` | Per-sample results |
| `results/metrics/*.json` | Aggregated metrics |
| `data/cache/cache.db` | Cached LLM responses |
| `data/indexes/` | BM25/dense indexes |
