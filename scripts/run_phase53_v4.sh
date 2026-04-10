#!/bin/bash
# Phase 5.3 perturbation experiments (run after Phase 5.2 completes)
set -e

cd /root/autodl-tmp/MovieRecommendationRAG
export OMP_NUM_THREADS=16
export PYTHONUNBUFFERED=1

LOG_FILE="logs/phase53_v4.log"
mkdir -p logs

# Verify vLLM is running
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "ERROR: vLLM not running."
    exit 1
fi

echo "Starting Phase 5.3 at $(date)"

PYTHONUNBUFFERED=1 /root/miniconda3/bin/python -u -m rag.pipeline \
    --phase 5.3 \
    --llm-backend api \
    --model /root/autodl-tmp/models/Qwen/Qwen2.5-7B-Instruct \
    --api-url http://localhost:8000/v1 \
    --num-samples 200 \
    --retrieval-k 8 \
    --output-dir "results/RAG_Phase_5.3" \
    2>&1 | tee "${LOG_FILE}"

echo "Phase 5.3 completed at $(date)"
