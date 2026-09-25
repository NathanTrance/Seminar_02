# Direct JEV Decision Demo

This demo sends source text directly to Jev's OpenRouter Decisions endpoint. It
does not use retrieval, embeddings, RAG, training, or another language model.

The exact binary request template used by the full EASE test experiment is
saved at `configs/jev_direct_binary_request.json`. At runtime, the
`<setup.py source text>` placeholder is replaced with each target source file;
the label, package name, and any retrieval context are not sent.

JEV documentation: <https://www.jevai.org/docs>

## Safety and scope

Samples are read as text and are never imported or executed. The bundled inputs
are synthetic integration fixtures, not paper test data and not an accuracy
benchmark. A decision applies only to the supplied source; package behavior may
exist elsewhere.

`confidence` and `probabilities` are signals returned by JEV. They are not
treated as calibrated malware probabilities.

## Dry run

Run from `Seminar_02/code`:

```bash
python scripts/run_jev_demo.py --dry-run
```

Dry-run validates both fixtures, builds the exact request payloads, writes a
run record under `results/jev_demo/`, and makes no network requests.

## Live demo

Set the key in the environment, never in a tracked file:

```bash
export OPENROUTER_API_KEY='sk-or-<personal-key>'
python scripts/run_jev_demo.py --live
```

The convenience launcher defaults to one live sample and prompts without
echoing the key if `JEV_API_KEY` is unavailable:

```bash
./run_scripts/run_jev_demo.sh
```

It optionally loads the ignored `code/.env` file. Its defaults can be changed
with `JEV_LIMIT`, `JEV_MODEL`, and `JEV_TIMEOUT`, or by passing Python CLI
arguments directly.

The default live demo makes exactly two requests, one per fixture. It calls:

```text
POST https://www.jevai.org/api/v1/decisions
```

Use `--limit 1` for a one-fixture smoke test.

The default Jev model identifier is `typesafe/jev-1.13`. Override it only with
a JEV identifier allowed by OpenRouter:

```bash
python scripts/run_jev_demo.py --live --model typesafe/jev-1.13
```

Analyze one local text file instead of the fixtures:

```bash
python scripts/run_jev_demo.py --live \
  --source-file path/to/sample.py \
  --sample-id example
```

## Replay

Display an existing result without a key or network access:

```bash
python scripts/run_jev_demo.py --replay results/jev_demo/<run-id>/run.json
```

Each run records source provenance and SHA-256, the request payload without
authentication, JEV's sanitized response, timing, and explicit errors. An API
failure is never represented as a benign result.

## Tests

The tests use mocked responses and require no key:

```bash
python -m unittest discover -s tests -v
```

## Later paper-data demo

The synthetic fixtures can later be replaced by a few usable records from the
original paper's Git LFS test files. Labels must stay local and must not be sent
to JEV. That experiment remains direct classification without retrieval.
