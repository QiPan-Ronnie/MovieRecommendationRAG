#!/bin/bash
set -euo pipefail

cd /root/autodl-tmp/MovieRecommendationRAG

export OMP_NUM_THREADS=16
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

MODEL_PATH="/root/autodl-tmp/models/Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR="/root/autodl-tmp/MovieRecommendationRAG/results/phase5_with_recommendation_Hybrid_p500"
LOG_DIR="/root/autodl-tmp/MovieRecommendationRAG/logs/phase5_with_recommendation_Hybrid_p500"
LOG_FILE="$LOG_DIR/phase53.log"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

/root/miniconda3/bin/python -u -m rag.pipeline \
  --phase 5.3 \
  --llm-backend huggingface \
  --model "$MODEL_PATH" \
  --num-samples 500 \
  --top-k 10 \
  --retrieval-k 8 \
  --evidence-mode hybrid \
  --output-dir "$OUTPUT_DIR" \
  2>&1 | tee "$LOG_FILE"
