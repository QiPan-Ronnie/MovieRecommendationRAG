#!/bin/bash
set -euo pipefail

cd /root/autodl-tmp/MovieRecommendationRAG

export OMP_NUM_THREADS=16
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

OUTPUT_DIR="/root/autodl-tmp/MovieRecommendationRAG/results/phase5_with_recommendation_Hybrid_p500"
LOG_DIR="/root/autodl-tmp/MovieRecommendationRAG/logs/phase5_with_recommendation_Hybrid_p500"
LOG_FILE="$LOG_DIR/phase54.log"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

/root/miniconda3/bin/python -u -m rag.pipeline \
  --phase 5.4 \
  --output-dir "$OUTPUT_DIR" \
  --bertscore-rescale-with-baseline false \
  2>&1 | tee "$LOG_FILE"
