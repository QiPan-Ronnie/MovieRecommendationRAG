#!/bin/bash
# Phase 5.4 (KG-only): faithfulness evaluation for KG-only explanations
set -euo pipefail

cd /root/autodl-tmp/MovieRecommendationRAG

export OMP_NUM_THREADS=16
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

OUTPUT_DIR="/root/autodl-tmp/MovieRecommendationRAG/results/phase5_with_recommendation_KG_Only"
LOG_DIR="/root/autodl-tmp/MovieRecommendationRAG/logs/phase5_with_recommendation_KG_Only"
LOG_FILE="$LOG_DIR/phase54.log"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

if [ ! -f "$OUTPUT_DIR/explanations_kg_only.jsonl" ]; then
  echo "ERROR: Missing $OUTPUT_DIR/explanations_kg_only.jsonl"
  echo "Run Phase 5.2 first."
  exit 1
fi

echo "============================================================"
echo "  Phase 5.4 (KG-only): faithfulness evaluation"
echo "  Start:  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Output: $OUTPUT_DIR"
echo "  Log:    $LOG_FILE"
echo "============================================================"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader || true

/root/miniconda3/bin/python -u -m rag.pipeline \
  --phase 5.4 \
  --output-dir "$OUTPUT_DIR" \
  2>&1 | tee "$LOG_FILE"
