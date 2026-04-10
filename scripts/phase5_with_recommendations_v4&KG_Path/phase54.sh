#!/bin/bash
# Phase 5.4 (corrected): faithfulness evaluation for recommendations_v4 + KG path
set -euo pipefail

cd /root/autodl-tmp/MovieRecommendationRAG

export OMP_NUM_THREADS=16
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

OUTPUT_DIR="/root/autodl-tmp/MovieRecommendationRAG/results/phase5_with_recommendations_v4&KG_Path"
LOG_DIR="/root/autodl-tmp/MovieRecommendationRAG/logs/phase5_with_recommendations_v4&KG_Path"
LOG_FILE="$LOG_DIR/phase54.log"

mkdir -p "$LOG_DIR"

if [ ! -f "$OUTPUT_DIR/explanations_rag.jsonl" ]; then
  echo "ERROR: Missing $OUTPUT_DIR/explanations_rag.jsonl"
  echo "Run Phase 5.2 first."
  exit 1
fi

echo "============================================================"
echo "  Phase 5.4 (corrected): recommendations_v4 + KG path"
echo "  Start:  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Output: $OUTPUT_DIR"
echo "  Log:    $LOG_FILE"
echo "============================================================"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader || true

/root/miniconda3/bin/python -u -m rag.pipeline \
  --phase 5.4 \
  --output-dir "$OUTPUT_DIR" \
  2>&1 | tee "$LOG_FILE"

