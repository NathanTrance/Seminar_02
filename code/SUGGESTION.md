# AGENT.md — Training-Free RAG for Malicious PyPI Package Detection

## 0. Project Goal

Build a modular, reproducible, **training-free** research prototype that tests whether better retrieval makes RAG useful for malicious PyPI package detection.

The project must be doable mainly through:
- Python coding
- dataset adapters
- static/AST analysis
- BM25 / dense retrieval / hybrid retrieval
- calls to LLMs through OpenAI-compatible APIs
- evaluation and plotting

Do **not** add model fine-tuning, LoRA, classifier training, embedding training, or dataset-cleaning pipelines unless explicitly requested later.

Primary research question:

> When does RAG help LLMs detect malicious PyPI package code, and can security-aware / contrastive retrieval improve both classification and explanation faithfulness?

Course-project scope:
- 3 LLMs by default, configurable to 2–4
- 2–3 ready-made evaluation datasets
- no model training
- no manual data-cleaning project
- all model inference behind one OpenAI-compatible client
- deterministic/reproducible experiment runner
- classification + retrieval + explanation-faithfulness evaluation

---

## 1. Hard Constraints

1. **No training.**
   - No fine-tuning.
   - No LoRA.
   - No learned reranker.
   - No learned classifier.
   - No embedding-model training.

2. **Never execute suspicious package code.**
   - Treat all malicious-package samples as hostile.
   - Static analysis only.
   - Do not `pip install`, import, run, build, or execute samples.
   - Do not evaluate setup scripts.
   - Extraction must be done as plain files only.

3. **API-first model access.**
   - All generation models must be called through an OpenAI-compatible interface.
   - The same code must work with vLLM, SGLang, Ollama-compatible gateways, OpenRouter-like providers, self-hosted endpoints, or other OpenAI-compatible APIs by changing environment variables only.
   - Do not hard-code provider-specific SDK logic in experiment code.

4. **Dataset-specific logic must live in adapters.**
   - Core experiment code must not know dataset file formats.
   - Each dataset adapter returns a common `Sample` schema.

5. **Cache every expensive result.**
   - LLM responses
   - embeddings
   - retrieval results
   - parsed static features

6. **Do not silently drop failed samples.**
   - Record parse/API failures explicitly.
   - Evaluation must report coverage.

---

## 2. Recommended Models

Default three-model panel:

1. `gpt-oss-20b`
2. `Qwen3-Coder-30B-A3B-Instruct`
3. `Llama-3.3-70B-Instruct`

Optional fourth family:
4. a Mistral/Codestral-class instruct model available on the configured endpoint

Exact API model IDs must be configured in `configs/models.yaml`; never assume the provider uses Hugging Face names.

We want models from different families because a retrieval effect that appears across several generators is more credible than a result on one LLM.

Example:

```yaml
models:
  - id: gpt_oss_20b
    api_model: gpt-oss-20b
  - id: qwen3_coder_30b
    api_model: Qwen/Qwen3-Coder-30B-A3B-Instruct
  - id: llama33_70b
    api_model: meta-llama/Llama-3.3-70B-Instruct
```

---

## 3. API Configuration

Use environment variables:

```bash
OPENAI_BASE_URL=http://localhost:8000/v1
OPENAI_API_KEY=dummy
```

Optional independent embedding endpoint:

```bash
EMBEDDING_BASE_URL=http://localhost:8001/v1
EMBEDDING_API_KEY=dummy
EMBEDDING_MODEL=text-embedding-model
```

If embedding variables are absent, allow embeddings to use the primary OpenAI-compatible endpoint.

Implement one wrapper:

```python
class LLMClient:
    def complete(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 512,
        response_format: dict | None = None,
    ) -> dict:
        ...
```

Use the `openai` Python package with configurable `base_url` and `api_key`.

The rest of the project must never instantiate provider-specific clients.

---

## 4. Dataset Plan

### Dataset A — EASE 2025 RAG paper replication dataset

Use the replication package from:

**Detecting Malicious Source Code in PyPI Packages with LLMs: Does RAG Come in Handy?**

Prefer the already-processed representations supplied by the authors.

The paper reports:
- 1,242 successfully processed malicious packages
- 3,752 successfully processed benign packages
- AST-derived behavioral descriptions

This is the **primary dataset** because it directly reproduces/extends the assigned paper.

Adapter name:

```text
EaseRagDataset
```

---

### Dataset B — LAMPS D1

From the replication package for:

**Many Hands Make Light Work: An LLM-based Multi-Agent System for Detecting Malicious PyPI Packages**

D1 is a balanced `setup.py` classification dataset:
- 3,000 malicious
- 3,000 benign

Use the repository-provided files/labels directly.

Adapter:

```text
LampsD1Dataset
```

This tests whether the retrieval findings hold on raw setup-script source rather than only pre-extracted behavior descriptions.

---

### Dataset C — LAMPS D2 (recommended third evaluation set if readily available)

D2 is a realistic multi-file PyPI evaluation dataset with natural class imbalance.

Adapter:

```text
LampsD2Dataset
```

Use it only if the replication package exposes all required samples and labels without requiring a data-curation project.

If D2 cannot be used cleanly, **do not spend the holiday reconstructing it**. Run Dataset A + Dataset B and mark Dataset C as an optional robustness experiment.

---

## 5. Unified Sample Schema

```python
from dataclasses import dataclass

@dataclass
class Sample:
    sample_id: str
    dataset: str
    label: int                 # 0 benign, 1 malicious
    raw_code: str | None
    behavior_text: str | None
    package_name: str | None
    file_path: str | None
    metadata: dict
```

Every adapter implements:

```python
class DatasetAdapter(Protocol):
    def load(self, split: str) -> list[Sample]:
        ...
```

Do not modify raw datasets in-place.

---

## 6. Experimental Design

There are two stages:

### Stage A — Retrieval-design ablation

Use:
- Dataset A development subset
- one default model, preferably Qwen3-Coder-30B-A3B-Instruct
- fixed random seed

Purpose: choose retrieval settings without repeatedly testing every hyperparameter across every model/dataset.

### Stage B — Frozen main evaluation

Freeze all retrieval settings chosen in Stage A.

Evaluate the selected systems on:
- 3 LLMs
- Dataset A
- Dataset B
- Dataset C if available

Do not retune per model or per dataset.

---

# 7. Full Course-Project Experiment List

## E0 — Zero-shot No-RAG baseline

Input:
- sample only

Output:
- malicious/benign
- confidence
- behavior claims
- evidence references

Purpose:
- establish base LLM performance
- measure whether RAG actually helps

Run across every model × dataset.

---

## E1 — Vanilla Dense RAG

Query representation:
- raw code when available
- supplied behavior text for behavior-description datasets

Knowledge base:
- malicious examples from the training/reference partition only

Retriever:
- dense embedding cosine similarity

Retrieve:
- top-k

Purpose:
- closest reproduction of ordinary code-example RAG
- baseline against the negative RAG finding in the reference paper

Run across every model × dataset after k is frozen.

---

## E2 — BM25 RAG

Same knowledge base as E1.

Retriever:
- BM25 lexical retrieval

Purpose:
- test whether security-relevant lexical/code tokens outperform generic dense semantic similarity

Run across every model × dataset.

---

## E3 — Hybrid RAG

Combine BM25 and dense retrieval with Reciprocal Rank Fusion (RRF).

Do not train fusion weights.

Example:

```python
score(d) = sum(1 / (60 + rank_r(d)) for r in retrievers)
```

Purpose:
- test whether lexical + semantic retrieval is more robust

Run across every model × dataset.

---

## E4 — Security-Behavior RAG

Represent code as a static behavior summary before retrieval.

Extract features such as:
- imports
- function/API calls
- subprocess/process creation
- shell execution
- `eval` / `exec`
- network-access calls
- filesystem writes/deletion
- environment-variable access
- base64/decode behavior
- suspicious command strings
- persistence-related APIs when statically visible

Example representation:

```text
imports=os,subprocess,requests;
behaviors=shell_execution,network_access,remote_download;
calls=os.system,requests.get
```

Retrieve against behavior representations of reference examples.

No learned extractor.

Purpose:
- test the central hypothesis that retrieval representation matters more than simply adding RAG

Run across every model × dataset where raw code exists.
For a dataset already containing behavior descriptions, use its provided behavior text.

---

## E5 — Contrastive RAG

Build two retrieval pools:
- malicious reference examples
- benign reference examples

For each query retrieve:
- top-k malicious
- top-k benign

Prompt the generator to compare the target with both groups.

Purpose:
- test whether malicious-only RAG biases the generator toward false positives
- provide explicit counterevidence

Run across every model × dataset.

Recommended final version:
- use the best retriever from E1–E4
- contrastive evidence composition

---

## E6 — Behavior + Contrastive RAG

This is the likely proposed method.

Pipeline:

```text
target source
    ↓
static behavior representation
    ↓
retrieve similar malicious examples
+
retrieve similar benign examples
    ↓
LLM classification + evidence
```

Purpose:
- test the combined hypothesis

This should be treated as the main proposed system.

Run across every model × dataset.

---

# 8. Small Ablations — Run Only During Stage A

Do NOT multiply these across the entire benchmark.

## A1 — top-k

Try:

```text
k ∈ {1, 3, 5}
```

Freeze the best/default k afterwards.

---

## A2 — Query representation

Compare:

```text
raw code
AST/static behavior description
raw code + behavior description
```

Only where available.

---

## A3 — Retrieval pool composition

Compare:

```text
malicious-only
benign-only
malicious + benign
```

Main interest is malicious-only vs contrastive.

---

## A4 — Prompt evidence ordering

Compare at most:

```text
retrieved evidence before target
target before retrieved evidence
```

Only do this if results look unusually prompt-sensitive.

---

# 9. Main Experiment Matrix

With 3 models and 3 datasets:

```text
E0 No RAG
E1 Dense
E2 BM25
E3 Hybrid
E4 Behavior RAG
E5 Contrastive RAG
E6 Behavior + Contrastive RAG
```

Maximum main cells:

```text
7 methods × 3 models × 3 datasets = 63 experiment cells
```

If this is too expensive, use the **core four**:

```text
E0 No RAG
E1 Dense RAG
E4 Behavior RAG
E6 Behavior + Contrastive RAG
```

Then:

```text
4 × 3 × 3 = 36 cells
```

This 36-cell version is the recommended holiday scope.

---

# 10. Metrics

For classification report:

- malicious-class precision
- malicious-class recall
- malicious-class F1
- macro F1
- balanced accuracy
- confusion matrix
- false-positive rate
- false-negative rate
- parse/API failure rate

Accuracy alone is insufficient because datasets may be imbalanced.

---

## Retrieval metrics

Where labels/categories make evaluation meaningful:

### Label Precision@k

For a malicious query:

```text
# retrieved malicious examples / k
```

For contrastive retrieval, report each pool independently.

### Same-behavior Hit@k

If behavioral labels can be automatically derived:

```text
1 if at least one retrieved item shares ≥1 target behavior else 0
```

### Retrieval similarity statistics

Save:
- ranks
- scores
- retrieved IDs

This enables later analysis of when retrieval helps/hurts.

---

# 11. What “Hallucination” Means Here

Do **not** define hallucination as “wrong malicious/benign prediction.”

Classification error and hallucination are different.

Example target:

```python
import requests
x = requests.get(url).text
```

Suppose the LLM says:

```text
The code downloads a payload, writes it to disk,
changes file permissions, and executes it through a shell.
```

Only network access/download is supported by the shown source.
The claims about:
- writing to disk
- chmod
- shell execution

are unsupported.

Those unsupported claims are **hallucinated security behaviors**.

The project should call this more precisely:

> **unsupported behavior claim rate** / **explanation faithfulness**

rather than pretending to solve hallucination in every possible sense.

---

## Structured output

Require:

```json
{
  "verdict": "malicious",
  "confidence": 0.84,
  "behaviors": [
    {
      "type": "network_access",
      "evidence": ["L12-L12"]
    }
  ],
  "rationale": "..."
}
```

The source supplied to the model must be line-numbered.

---

## Automated faithfulness check

Build a lightweight AST/static checker producing behavior flags:

```json
{
  "shell_execution": false,
  "process_creation": false,
  "network_access": true,
  "file_write": false,
  "dynamic_execution": false,
  "environment_access": false,
  "base64_decode": false
}
```

Normalize model claims into the same behavior vocabulary.

Then:

```text
unsupported_claim =
    model_claim == True
    AND static_checker_support == False
```

Primary metric:

```text
Unsupported Behavior Claim Rate
= unsupported behavior claims / all model behavior claims
```

Also report:

```text
Behavior Precision
= supported claimed behaviors / all claimed behaviors

Behavior Recall
= correctly claimed observed behaviors / statically observed behaviors
```

Important limitation:
absence from static analysis does not prove the behavior is impossible.
Therefore call this an **automated static-grounding proxy**, not perfect hallucination ground truth.

---

## Optional manual validation

Randomly sample ~100 explanations.

Manually label each claimed behavior:
- supported
- unsupported
- ambiguous

Use this only to validate that the automated metric is reasonable.

Do not manually annotate the whole dataset.

---

# 12. Prompting Rules

Use a single prompt template across methods except for the evidence block.

Temperature:

```text
0.0
```

If provider behavior makes true deterministic decoding impossible, store all generation parameters and seeds.

Require strict JSON.

System prompt should say:

```text
You are performing static source-code analysis.
Do not assume behavior that is not supported by the provided source or retrieved evidence.
Retrieved examples are references, not proof that the target is malicious.
Classify the TARGET only.
Every claimed target behavior must cite target source lines.
```

For RAG, retrieved documents must have stable IDs such as:

```text
MAL_000123
BEN_000847
```

Never let the model confuse retrieved-example behavior with target behavior.

---

# 13. Repository Structure

```text
.
├── AGENT.md
├── README.md
├── pyproject.toml
├── .env.example
├── configs/
│   ├── models.yaml
│   ├── datasets.yaml
│   ├── experiments.yaml
│   └── behaviors.yaml
├── data/
│   ├── raw/              # gitignored when licensing requires
│   ├── indexes/
│   └── cache/
├── src/
│   └── ragsec/
│       ├── clients/
│       │   ├── llm.py
│       │   └── embeddings.py
│       ├── datasets/
│       │   ├── base.py
│       │   ├── ease_rag.py
│       │   ├── lamps_d1.py
│       │   └── lamps_d2.py
│       ├── static/
│       │   ├── ast_features.py
│       │   ├── behaviors.py
│       │   └── line_numbering.py
│       ├── retrieval/
│       │   ├── base.py
│       │   ├── bm25.py
│       │   ├── dense.py
│       │   ├── hybrid.py
│       │   └── contrastive.py
│       ├── prompts/
│       │   ├── classifier.py
│       │   └── schemas.py
│       ├── evaluation/
│       │   ├── classification.py
│       │   ├── retrieval.py
│       │   ├── faithfulness.py
│       │   └── statistics.py
│       ├── experiments/
│       │   ├── runner.py
│       │   └── registry.py
│       └── utils/
│           ├── cache.py
│           ├── logging.py
│           └── hashing.py
├── scripts/
│   ├── build_indexes.py
│   ├── run_experiment.py
│   ├── run_matrix.py
│   ├── evaluate.py
│   └── make_tables.py
├── results/
│   ├── raw/
│   ├── metrics/
│   ├── tables/
│   └── figures/
└── tests/
```

---

# 14. Experiment Configuration

Example:

```yaml
experiment_id: behavior_contrastive_qwen_lamps_d1

dataset:
  name: lamps_d1
  split: test

model:
  id: qwen3_coder_30b

retrieval:
  enabled: true
  type: hybrid
  query_representation: behavior
  contrastive: true
  top_k_malicious: 3
  top_k_benign: 3

generation:
  temperature: 0.0
  max_tokens: 512

seed: 42
```

Every run must save the fully resolved config next to its outputs.

---

# 15. Results Schema

Store one JSONL row per target sample:

```json
{
  "experiment_id": "...",
  "dataset": "...",
  "model": "...",
  "sample_id": "...",
  "gold_label": 1,
  "predicted_label": 1,
  "confidence": 0.86,
  "behaviors": [],
  "retrieved_ids": [],
  "retrieval_scores": [],
  "raw_response": "...",
  "latency_ms": 1234,
  "input_tokens": 1200,
  "output_tokens": 180,
  "parse_ok": true,
  "error": null
}
```

Never overwrite existing raw runs.

---

# 16. Caching

Cache key must include:

```text
model
prompt hash
generation parameters
sample content hash
retrieved evidence IDs
```

For embeddings:

```text
embedding model
document content hash
```

Use SQLite, DuckDB, or filesystem JSON/Parquet.

Do not add Redis or external infrastructure unless necessary.

---

# 17. Reproducibility

For every run record:

- git commit hash
- timestamp
- dataset version/hash
- model API ID
- base URL hostname or anonymized provider label
- prompt version/hash
- retrieval config
- seed
- generation parameters

Create:

```bash
python scripts/run_matrix.py --config configs/experiments.yaml
```

It should resume safely after interruption.

---

# 18. Minimum Deliverable

The project is course-complete once the following work:

1. Dataset A + Dataset B
2. 2 or more LLMs
3. E0 No-RAG
4. E1 Dense RAG
5. E4 Behavior RAG
6. E6 Behavior + Contrastive RAG
7. classification metrics
8. retrieval logs
9. explanation-faithfulness metric
10. result tables/figures
11. modular memory/file persistence where required by coursework
12. README + report material

Dataset C and the third/fourth model are robustness additions, not blockers.

---

# 19. What NOT to Build Yet

Do not add:

- fine-tuning
- learned rerankers
- graph neural networks
- agent frameworks
- LangChain unless it becomes genuinely useful
- vector databases requiring servers
- dynamic malware execution
- sandbox infrastructure
- gigantic custom datasets
- heavy manual annotation
- deployment UI

Prefer plain Python abstractions.

FAISS can be used for dense indexes; BM25 can use `rank-bm25`.
A small local persistent store is enough.

---

# 20. EASE Extension — NOT Required for Course Version

These are later research extensions.

## Cross-dataset evaluation

Instead of merely testing each dataset independently:

```text
retrieval/reference knowledge from Dataset A
→ test on Dataset B
```

and vice versa.

Question:

> Does the method generalize to malware originating from a different dataset/source, or does it rely on dataset-specific patterns?

This is stronger than ordinary within-dataset testing.

---

## Temporal split

Split examples by discovery/publication time:

```text
reference KB = older malware
test = newer malware
```

Question:

> Can RAG detect attacks that appeared after the knowledge base was constructed?

This better represents the real security setting.

Do not implement until reliable timestamps exist.

---

## Attack-family evaluation

Group malicious samples by behavior/family/campaign, e.g.:

```text
credential theft
dropper
remote-code execution
exfiltration
reverse shell
obfuscation
dependency confusion
typosquatting
```

Then report per-family recall/F1.

Question:

> Which attack types benefit from retrieval, and which remain hard?

Requires trustworthy family labels, so leave it for later if labels are messy.

---

## Trained reranker

Course version:

```text
BM25/dense retrieval → top-k
```

Paper extension:

```text
BM25/dense retrieval → top-20 → learned reranker → top-k
```

A reranker is trained or fine-tuned to decide which retrieved evidence is actually useful for malicious-code reasoning.

This violates the course project's “chill/no-training” rule, so do it only later.

---

## Selective RAG

Do not retrieve for every sample.

Example:

```text
LLM first pass
    ↓
if uncertain / conflicting / low confidence
    → retrieve
else
    → keep prediction
```

Question:

> Can we get the benefit of RAG on hard samples while avoiding harmful retrieval on easy samples?

Metrics:
- F1
- percentage of samples using retrieval
- latency
- token usage

Training is not necessarily required; the trigger can be rule-based.

---

## Statistical testing

For the course version, bootstrap confidence intervals are optional.

For a paper:
- paired bootstrap confidence intervals
- McNemar's test for paired classification predictions
- multiple-comparison correction when testing many methods
- effect sizes

Question:

> Is the improvement likely real rather than random variation?

---

## Error taxonomy

Manually inspect a stratified sample of failures and categorize them:

```text
retrieval failure
irrelevant evidence
retrieved near-duplicate
LLM ignored correct evidence
LLM copied malicious behavior from retrieved example
obfuscation
long-context truncation
benign admin behavior mistaken for malware
unsupported explanation
```

Question:

> Why does RAG help or hurt?

This is often where an empirical SE paper gets much of its explanatory value.

---

# 21. Implementation Order

Implement in this order:

1. project skeleton
2. OpenAI-compatible client
3. unified dataset schema + Dataset A adapter
4. zero-shot classifier
5. result cache
6. evaluation metrics
7. dense retriever
8. BM25 retriever
9. static behavior extractor
10. behavior retrieval
11. benign/malicious dual indexes
12. contrastive prompt
13. Dataset B adapter
14. main experiment runner
15. plots/tables
16. optional Dataset C
17. optional third/fourth model

At every stage keep the code runnable.

---

# 22. Coding Style

- Python 3.11+
- use type hints
- use dataclasses or Pydantic for configs/results
- small modules
- deterministic functions where possible
- pathlib instead of manual path strings
- logging instead of print for experiment infrastructure
- pytest for retrieval/static-analysis/evaluation logic
- Ruff for linting
- avoid unnecessary abstraction/frameworks

---

# 23. Agent Working Rules

When implementing autonomously:

1. Read this file before making architectural changes.
2. Prefer the simplest implementation satisfying the experiment.
3. Do not introduce training.
4. Do not execute dataset samples.
5. Do not add a dependency without a concrete use.
6. Do not refactor unrelated working code during experiment implementation.
7. Preserve raw experiment outputs.
8. Add tests for bug fixes involving metrics, parsing, caching, or retrieval.
9. If an external dataset format differs from expectations, adapt through its dataset adapter rather than modifying core logic.
10. If exact dataset availability blocks an optional experiment, skip it cleanly and document the limitation instead of building a new data-curation pipeline.
11. Any prompt change that could alter results must receive a new prompt version/hash.
12. Before a large run, execute a smoke test on 5–10 samples.
13. Never log API keys.
14. Never commit malicious extracted archives unless the source license and repository policy explicitly permit it.

---

# 24. First Coding Target

The first milestone should support this command:

```bash
python scripts/run_experiment.py \
  --dataset ease_rag \
  --model qwen3_coder_30b \
  --method no_rag \
  --limit 20
```

Then:

```bash
python scripts/run_experiment.py \
  --dataset ease_rag \
  --model qwen3_coder_30b \
  --method dense \
  --limit 20
```

Only after those work reliably should behavior/contrastive retrieval be added.
