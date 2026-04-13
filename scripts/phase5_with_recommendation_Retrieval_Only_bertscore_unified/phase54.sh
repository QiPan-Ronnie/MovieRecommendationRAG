#!/bin/bash
set -euo pipefail

cd /root/autodl-tmp/MovieRecommendationRAG

export OMP_NUM_THREADS=16
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

SOURCE_DIR="/root/autodl-tmp/MovieRecommendationRAG/results/backup_pre_kg_fix_20260403_152531/phase5_with_recommendations_v4"
OUTPUT_DIR="/root/autodl-tmp/MovieRecommendationRAG/results/phase5_with_recommendation_Retrieval_Only_bertscore_unified"
LOG_DIR="/root/autodl-tmp/MovieRecommendationRAG/logs/phase5_with_recommendation_Retrieval_Only_bertscore_unified"
LOG_FILE="$LOG_DIR/phase54.log"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

link_if_exists() {
  local src="$1"
  local dst="$2"
  if [ ! -e "$dst" ] && [ -f "$src" ]; then
    ln -s "$src" "$dst"
  fi
}

link_if_exists "$SOURCE_DIR/explanations_rag.jsonl" "$OUTPUT_DIR/explanations_rag.jsonl"
link_if_exists "$SOURCE_DIR/explanations_prompt_only.jsonl" "$OUTPUT_DIR/explanations_prompt_only.jsonl"
link_if_exists "$SOURCE_DIR/perturbation_results.jsonl" "$OUTPUT_DIR/perturbation_results.jsonl"

/root/miniconda3/bin/python -u -m rag.pipeline \
  --phase 5.4 \
  --output-dir "$OUTPUT_DIR" \
  --bertscore-rescale-with-baseline false \
  2>&1 | tee "$LOG_FILE"
