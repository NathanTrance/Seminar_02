#!/bin/bash
cd /home/nhatth/Desktop/Mystuff/Seminar_2
source .venv/bin/activate
export OPENAI_BASE_URL='https://nhatth2-llama-3-8b-instruct-3-runai-voice-test.runai-inference.cyberspace.vn/v1'
export OPENAI_API_KEY='dummy'
export EMBEDDING_BASE_URL='https://nhatth2-qwen3-4b-embedding-runai-voice-test.runai-inference.cyberspace.vn/v1'
export EMBEDDING_API_KEY='dummy'
export EMBEDDING_MODEL='Qwen3-Embedding-4B'
export SSL_VERIFY='false'
export PYTHONPATH=.

python3 scripts/run_experiment.py \
  --dataset ease_rag \
  --model Llama-3.1-8B-Instruct \
  --method dense \
  --split test \
  --max-tokens 1024 \
  --top-k 3

echo "Done. results/raw/dense_Llama-3.1-8B-Instruct_ease_rag.jsonl"
