#!/bin/bash
# ============================================================
# Phase 5 full run (corrected): recommendations_v4 + KG path
#
# This script reproduces the current corrected full pipeline:
#   Phase 5.2 -> Phase 5.3 -> Phase 5.4
#
# Output:
#   /root/autodl-tmp/MovieRecommendationRAG/results/phase5_with_recommendations_v4&KG_Path
# Logs:
#   /root/autodl-tmp/MovieRecommendationRAG/logs/phase5_with_recommendations_v4&KG_Path
#
# Usage:
#   cd /root/autodl-tmp/MovieRecommendationRAG
#   bash 'scripts/phase5_with_recommendations_v4&KG_Path/run_full.sh'
#
# Background:
#   nohup bash -lc "cd /root/autodl-tmp/MovieRecommendationRAG && \
#     bash 'scripts/phase5_with_recommendations_v4&KG_Path/run_full.sh'" \
#     > '/root/autodl-tmp/MovieRecommendationRAG/logs/phase5_with_recommendations_v4&KG_Path/nohup.out' 2>&1 &
# ============================================================

set -euo pipefail

cd /root/autodl-tmp/MovieRecommendationRAG

export OMP_NUM_THREADS=16
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1

MODEL_PATH="/root/autodl-tmp/models/Qwen/Qwen2.5-7B-Instruct"
API_URL="http://localhost:8000/v1"
OUTPUT_DIR="/root/autodl-tmp/MovieRecommendationRAG/results/phase5_with_recommendations_v4&KG_Path"
LOG_DIR="/root/autodl-tmp/MovieRecommendationRAG/logs/phase5_with_recommendations_v4&KG_Path"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

{
  echo "============================================================"
  echo "  Phase 5 full run (recommendations_v4 + KG_Path)"
  echo "  Start:  $(date '+%Y-%m-%d %H:%M:%S')"
  echo "  Output: $OUTPUT_DIR"
  echo "  Logs:   $LOG_DIR"
  echo "============================================================"
} | tee -a "$LOG_DIR/master.log"

if [ ! -f "results/results_from_kg/recommendations_v4.csv" ]; then
  echo "ERROR: Missing results/results_from_kg/recommendations_v4.csv" | tee -a "$LOG_DIR/master.log"
  exit 1
fi

if [ ! -f "data/kg/kg_paths_for_recommendations.json" ]; then
  echo "ERROR: Missing data/kg/kg_paths_for_recommendations.json" | tee -a "$LOG_DIR/master.log"
  exit 1
fi

if ! curl -s "$API_URL/models" > /dev/null 2>&1; then
  echo "ERROR: vLLM is not running at $API_URL" | tee -a "$LOG_DIR/master.log"
  echo "Start it first with: bash scripts/start_vllm.sh" | tee -a "$LOG_DIR/master.log"
  exit 1
fi

echo "[1/3] Phase 5.2 start $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_DIR/master.log"
/root/miniconda3/bin/python -u -m rag.pipeline \
  --phase 5.2 \
  --llm-backend api \
  --model "$MODEL_PATH" \
  --api-url "$API_URL" \
  --top-k 10 \
  --retrieval-k 8 \
  --alpha 0.6 \
  --concurrency 16 \
  --output-dir "$OUTPUT_DIR" \
  2>&1 | tee "$LOG_DIR/phase52.log"
echo "[1/3] Phase 5.2 done $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_DIR/master.log"

echo "[2/3] Phase 5.3 start $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_DIR/master.log"
/root/miniconda3/bin/python -u -m rag.pipeline \
  --phase 5.3 \
  --llm-backend api \
  --model "$MODEL_PATH" \
  --api-url "$API_URL" \
  --num-samples 200 \
  --top-k 10 \
  --retrieval-k 8 \
  --output-dir "$OUTPUT_DIR" \
  2>&1 | tee "$LOG_DIR/phase53.log"
echo "[2/3] Phase 5.3 done $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_DIR/master.log"

echo "[3/3] Phase 5.4 start $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_DIR/master.log"
/root/miniconda3/bin/python -u -m rag.pipeline \
  --phase 5.4 \
  --output-dir "$OUTPUT_DIR" \
  2>&1 | tee "$LOG_DIR/phase54.log"
echo "[3/3] Phase 5.4 done $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_DIR/master.log"

echo "[DONE] Full pipeline completed at $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_DIR/master.log"
