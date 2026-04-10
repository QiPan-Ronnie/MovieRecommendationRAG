#!/bin/bash
# Phase 5.3 (KG-only): perturbation experiments on KG-only explanations
set -euo pipefail

cd /root/autodl-tmp/MovieRecommendationRAG

export OMP_NUM_THREADS=16
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1

MODEL_PATH="/root/autodl-tmp/models/Qwen/Qwen2.5-7B-Instruct"
API_URL="http://localhost:8000/v1"
OUTPUT_DIR="/root/autodl-tmp/MovieRecommendationRAG/results/phase5_with_recommendation_KG_Only"
LOG_DIR="/root/autodl-tmp/MovieRecommendationRAG/logs/phase5_with_recommendation_KG_Only"
LOG_FILE="$LOG_DIR/phase53.log"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

if [ ! -f "data/kg/kg_paths_for_recommendations.json" ]; then
  echo "ERROR: Missing data/kg/kg_paths_for_recommendations.json"
  exit 1
fi

if ! curl -s "$API_URL/models" > /dev/null 2>&1; then
  echo "ERROR: vLLM is not running at $API_URL"
  echo "Start it first with: bash scripts/start_vllm.sh"
  exit 1
fi

echo "============================================================"
echo "  Phase 5.3 (KG-only): perturbation experiments"
echo "  Start:  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Output: $OUTPUT_DIR"
echo "  Log:    $LOG_FILE"
echo "============================================================"

/root/miniconda3/bin/python -u -m rag.pipeline \
  --phase 5.3 \
  --llm-backend api \
  --model "$MODEL_PATH" \
  --api-url "$API_URL" \
  --num-samples 200 \
  --top-k 10 \
  --retrieval-k 8 \
  --evidence-mode kg_only \
  --output-dir "$OUTPUT_DIR" \
  2>&1 | tee "$LOG_FILE"
