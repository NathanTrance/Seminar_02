#!/bin/bash
cd /home/nhatth/Desktop/Mystuff/Seminar_2
source .venv/bin/activate
export OPENAI_BASE_URL='https://nhatth2-llama-3-8b-instruct-3-runai-voice-test.runai-inference.cyberspace.vn/v1'
export OPENAI_API_KEY='dummy'
export SSL_VERIFY='false'
export PYTHONPATH=.

python3 scripts/run_experiment.py \
  --dataset ease_rag \
  --model Llama-3.1-8B-Instruct \
  --method no_rag \
  --split test \
  --max-tokens 1024

echo "Done. Check results/raw/*.jsonl for output."
