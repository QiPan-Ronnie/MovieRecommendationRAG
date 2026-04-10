#!/bin/bash
# Phase 5.4 — faithfulness evaluation (GPU, no vLLM needed)
set -e
cd /root/autodl-tmp/MovieRecommendationRAG
export OMP_NUM_THREADS=16
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

OUTPUT_DIR="/root/autodl-tmp/MovieRecommendationRAG/results/phase5_with_recommendations_v4"
LOG_DIR="/root/autodl-tmp/MovieRecommendationRAG/logs/phase5_with_recommendations_v4"
LOG_FILE="$LOG_DIR/phase54.log"
mkdir -p "$LOG_DIR"

# Verify explanation files
if [ ! -f "$OUTPUT_DIR/explanations_rag.jsonl" ]; then
    echo "ERROR: explanations_rag.jsonl not found in $OUTPUT_DIR"; exit 1
fi

echo "Output dir : $OUTPUT_DIR"
echo "Log file   : $LOG_FILE"
echo "Start      : $(date)"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader

PYTHONUNBUFFERED=1 /root/miniconda3/bin/python -u -m rag.pipeline \
    --phase 5.4 \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee "$LOG_FILE"

echo "Done: $(date)"
