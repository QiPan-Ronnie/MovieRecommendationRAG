#!/bin/bash
# Phase 5.2 — V4 LambdaMART + KG paths → explanation generation
set -e

cd /root/autodl-tmp/MovieRecommendationRAG
source /etc/network_turbo 2>/dev/null || true

export OMP_NUM_THREADS=16
export PYTHONUNBUFFERED=1

OUTPUT_DIR="/root/autodl-tmp/MovieRecommendationRAG/results/phase5_with_recommendations_v4"
LOG_DIR="/root/autodl-tmp/MovieRecommendationRAG/logs/phase5_with_recommendations_v4"
LOG_FILE="$LOG_DIR/phase52.log"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# Preflight checks
if [ ! -f "results/results_from_kg/recommendations_v4.csv" ]; then
    echo "ERROR: recommendations_v4.csv not found!"; exit 1
fi
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "ERROR: vLLM not running. Start with: bash scripts/start_vllm.sh"; exit 1
fi

echo "Output dir : $OUTPUT_DIR"
echo "Log file   : $LOG_FILE"
echo "vLLM       : OK"
echo "Starting Phase 5.2 at $(date)"

PYTHONUNBUFFERED=1 /root/miniconda3/bin/python -u -m rag.pipeline \
    --phase 5.2 \
    --llm-backend api \
    --model /root/autodl-tmp/models/Qwen/Qwen2.5-7B-Instruct \
    --api-url http://localhost:8000/v1 \
    --top-k 10 \
    --retrieval-k 8 \
    --alpha 0.6 \
    --concurrency 16 \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee "$LOG_FILE"

echo "Phase 5.2 completed at $(date)"
