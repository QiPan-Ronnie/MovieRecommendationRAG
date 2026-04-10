#!/bin/bash
# ============================================================
# Phase 5.4 忠实度评估 (v2) - 独立日志 + 独立结果目录
# 用法 (从项目根目录):
#   nohup bash scripts/run_phase54_v2.sh >> logs/phase54.log 2>&1 &
# 依赖: Phase 5.2 + 5.3 完成后运行（纯 CPU，无需 vLLM）
# ============================================================

set -e
cd /root/autodl-tmp/MovieRecommendationRAG

OUTPUT_DIR="results/RAG_Phase_5.4"
LOG_DIR="logs"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo "============================================================"
echo "  Phase 5.4 忠实度评估 (v2)"
echo "  目录: $(pwd)"
echo "  时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  输出: ${OUTPUT_DIR}/"
echo "============================================================"

# ── 前置文件检查 ──────────────────────────────────────────────
MISSING=0
echo ""
echo "[检查] 前置文件..."

for f in \
    "results/explanations_rag.jsonl" \
    "results/explanations_prompt_only.jsonl" \
    "results/RAG_Phase_5.3/perturbation_results.jsonl"; do
    if [ ! -f "$f" ]; then
        echo "  [警告] 缺失: $f"
        MISSING=1
    else
        COUNT=$(wc -l < "$f")
        echo "  [OK]   $f (${COUNT} 条)"
    fi
done

echo ""
if [ $MISSING -eq 1 ]; then
    echo "[提示] 部分文件缺失，将跳过对应评估项，继续执行"
fi

# ── 显示已有输出 ──────────────────────────────────────────────
if [ -d "${OUTPUT_DIR}" ]; then
    EXISTING=$(find "${OUTPUT_DIR}" -name "*.jsonl" -o -name "*.json" 2>/dev/null | wc -l)
    if [ "$EXISTING" -gt 0 ]; then
        echo "[续算] ${OUTPUT_DIR}/ 中已有 ${EXISTING} 个结果文件"
    fi
fi

# ── 启动评估 ──────────────────────────────────────────────────
echo "[启动] 忠实度评估..."
echo "       指标: Evidence Overlap / ROUGE-L / Semantic Similarity / BERTScore"
echo "       预计耗时: 20-40 分钟 (纯 CPU)"
echo ""

HF_HUB_OFFLINE=1 /root/miniconda3/bin/python -m rag.pipeline \
    --phase 5.4 \
    --output-dir "${OUTPUT_DIR}"

echo ""
echo "============================================================"
echo "  Phase 5.4 完成！时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "  输出文件:"
echo "    ${OUTPUT_DIR}/faithfulness_rag/"
echo "      ├── faithfulness_detailed.jsonl"
echo "      └── faithfulness_summary.json"
echo "    ${OUTPUT_DIR}/faithfulness_prompt_only/"
echo "      ├── faithfulness_detailed.jsonl"
echo "      └── faithfulness_summary.json"
echo "    ${OUTPUT_DIR}/faithfulness_perturbation/"
echo "      ├── faithfulness_detailed.jsonl"
echo "      └── faithfulness_summary.json"
echo "============================================================"
