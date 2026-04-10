#!/bin/bash
# Phase 5.4 Faithfulness Evaluation (v3) - 实时进度可视化版
# 用法: nohup bash scripts/run_phase54_v3.sh >> logs/phase54.log 2>&1 &
# 监控: tail -f logs/phase54.log

set -euo pipefail
cd /root/autodl-tmp/MovieRecommendationRAG

OUTPUT_DIR="results/RAG_Phase_5.4"
mkdir -p "${OUTPUT_DIR}" logs

echo ""
echo "============================================================"
echo "  Phase 5.4 Faithfulness Evaluation (v3)"
echo "  时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  输出: ${OUTPUT_DIR}/"
echo "============================================================"
echo ""

echo "[检查] 前置文件..."
for f in \
    "results/explanations_rag.jsonl" \
    "results/explanations_prompt_only.jsonl" \
    "results/RAG_Phase_5.3/perturbation_results.jsonl"; do
    if [ -f "$f" ]; then
        COUNT=$(wc -l < "$f")
        echo "  [OK]  $f  (${COUNT} 条)"
    else
        echo "  [WARN] 未找到: $f"
    fi
done
echo ""

# OMP_NUM_THREADS: 避免 libgomp 警告
# PYTHONUNBUFFERED=1 + python -u: 实时输出到日志
export OMP_NUM_THREADS=16
export PYTHONUNBUFFERED=1

echo "[启动] 忠实度评估 (实时进度模式)..."
echo "       监控: tail -f logs/phase54.log"
echo ""

PYTHONUNBUFFERED=1 /root/miniconda3/bin/python -u -m rag.pipeline \
    --phase 5.4 \
    --output-dir "${OUTPUT_DIR}"

echo ""
echo "============================================================"
echo "  Phase 5.4 完成! $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""
echo "结果文件:"
find "${OUTPUT_DIR}" \( -name "*.json" -o -name "*.jsonl" \) 2>/dev/null | sort | while read f; do
    SIZE=$(du -sh "$f" 2>/dev/null | cut -f1)
    echo "  $f  [$SIZE]"
done
