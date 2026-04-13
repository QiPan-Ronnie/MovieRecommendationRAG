#!/bin/bash
set -euo pipefail

cd /root/autodl-tmp/MovieRecommendationRAG

export OMP_NUM_THREADS=16
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

SOURCE_DIR="/root/autodl-tmp/MovieRecommendationRAG/results/phase5_with_recommendation_KG_Only"
OUTPUT_DIR="/root/autodl-tmp/MovieRecommendationRAG/results/phase5_with_recommendation_KG_Only_bertscore_unified"
LOG_DIR="/root/autodl-tmp/MovieRecommendationRAG/logs/phase5_with_recommendation_KG_Only_bertscore_unified"
LOG_FILE="$LOG_DIR/phase54.log"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

link_if_exists() {
  local src="$1"
  local dst="$2"
  if [ ! -e "$dst" ] && [ -f "$src" ]; then
    ln -s "$src" "$dst"
  fi
}

link_if_exists "$SOURCE_DIR/explanations_kg_only.jsonl" "$OUTPUT_DIR/explanations_kg_only.jsonl"
link_if_exists "$SOURCE_DIR/explanations_prompt_only.jsonl" "$OUTPUT_DIR/explanations_prompt_only.jsonl"
link_if_exists "$SOURCE_DIR/perturbation_results.jsonl" "$OUTPUT_DIR/perturbation_results.jsonl"

/root/miniconda3/bin/python -u -m rag.pipeline \
  --phase 5.4 \
  --output-dir "$OUTPUT_DIR" \
  --bertscore-rescale-with-baseline false \
  2>&1 | tee "$LOG_FILE"
