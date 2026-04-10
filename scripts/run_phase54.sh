#!/bin/bash
# ============================================================
# Phase 5.4 忠实度评估 + 可视化
# 用法 (从项目根目录): bash scripts/run_phase54.sh
# 依赖: Phase 5.2 + 5.3 完成后运行（纯 CPU，无需 vLLM）
# ============================================================

set -e
cd "$(dirname "$0")/.."

LOG_DIR="logs"
LOG_FILE="$LOG_DIR/phase54_evaluation.log"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "  Phase 5.4 忠实度评估"
echo "  目录: $(pwd)"
echo "  时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  日志: $LOG_FILE"
echo "============================================================"

# 检查前置输出
MISSING=0
for f in "results/explanations_rag.jsonl" \
         "results/explanations_prompt_only.jsonl" \
         "results/perturbation_results.jsonl"; do
    if [ ! -f "$f" ]; then
        echo "[警告] 未找到: $f"
        MISSING=1
    else
        COUNT=$(wc -l < "$f")
        echo "[OK] $f ($COUNT 条)"
    fi
done

if [ $MISSING -eq 1 ]; then
    echo ""
    echo "[提示] 部分文件缺失，将跳过对应评估项，继续执行"
fi

echo ""
echo "[启动] 忠实度评估 (BERTScore + ROUGE-L + 语义相似度，约 20-40 分钟)"
echo ""

HF_HUB_OFFLINE=1 python -m rag.pipeline \
    --phase 5.4 \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "============================================================"
echo "  Phase 5.4 完成！时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  RAG 评估:    results/faithfulness_rag/"
echo "  Prompt 评估: results/faithfulness_prompt_only/"
echo "  扰动评估:    results/faithfulness_perturbation/"
echo "============================================================"
