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

echo "Starting all experiments..."
echo ""

# E1 — Dense RAG
# echo "=== E1: Dense RAG ==="
# python3 scripts/run_experiment.py --dataset ease_rag --model Llama-3.1-8B-Instruct --method dense --split test --max-tokens 1024 --top-k 3

# E3 — Hybrid RAG
# echo "=== E3: Hybrid RAG ==="
# python3 scripts/run_experiment.py --dataset ease_rag --model Llama-3.1-8B-Instruct --method hybrid --split test --max-tokens 1024 --top-k 3

# # E4 — Behavior RAG
# echo "=== E4: Behavior RAG ==="
# python3 scripts/run_experiment.py --dataset ease_rag --model Llama-3.1-8B-Instruct --method behavior --split test --max-tokens 1024 --top-k 3

# # E5 — Contrastive RAG
# echo "=== E5: Contrastive RAG ==="
# python3 scripts/run_experiment.py --dataset ease_rag --model Llama-3.1-8B-Instruct --method contrastive --split test --max-tokens 1024 --top-k 3

# # E6 — Behavior + Contrastive RAG
# echo "=== E6: Behavior + Contrastive RAG ==="
# python3 scripts/run_experiment.py --dataset ease_rag --model Llama-3.1-8B-Instruct --method behavior_contrastive --split test --max-tokens 1024 --top-k 3

echo ""
echo "All done! Running eval..."
for method in dense hybrid behavior contrastive behavior_contrastive; do
    echo ""
    echo "=== $method ==="
    python3 scripts/evaluate.py results/raw/${method}_Llama-3.1-8B-Instruct_ease_rag.jsonl 2>&1 | grep -E "malicious_f1|malicious_recall|balanced_accuracy|coverage"
done
