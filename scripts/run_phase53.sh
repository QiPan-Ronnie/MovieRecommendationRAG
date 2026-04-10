#!/bin/bash
# ============================================================
# Phase 5.3 扰动实验 E1-E4
# 用法 (从项目根目录): bash scripts/run_phase53.sh
# 依赖: Phase 5.2 完成后运行
# ============================================================

set -e
cd "$(dirname "$0")/.."

LOG_DIR="logs"
LOG_FILE="$LOG_DIR/phase53_perturbation.log"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "  Phase 5.3 扰动实验 E1-E4"
echo "  目录: $(pwd)"
echo "  时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  日志: $LOG_FILE"
echo "============================================================"

# 检查 Phase 5.2 输出是否存在
if [ ! -f "results/explanations_rag.jsonl" ]; then
    echo "[错误] 未找到 results/explanations_rag.jsonl"
    echo "  请先完成 Phase 5.2: bash scripts/run_phase52.sh"
    exit 1
fi
RAG_COUNT=$(wc -l < results/explanations_rag.jsonl)
echo "[检查] Phase 5.2 输出: $RAG_COUNT 条 RAG 解释"

# 确认 vLLM 服务在线
echo "[检查] vLLM 服务..."
if ! curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo "[错误] vLLM 服务未运行！请先启动："
    echo "  bash scripts/start_vllm.sh"
    exit 1
fi
echo "[OK] vLLM 服务在线"

echo "[启动] 扰动实验 (200 样本 × 4 条件 = 800 次 LLM 调用，约 1-2 小时)"
echo ""

HF_HUB_OFFLINE=1 python -m rag.pipeline \
    --phase 5.3 \
    --llm-backend api \
    --api-url http://localhost:8000/v1 \
    --model /root/autodl-tmp/models/Qwen/Qwen2.5-7B-Instruct \
    --num-samples 200 \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "============================================================"
echo "  Phase 5.3 完成！时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  输出: results/perturbation_results.jsonl"
echo "============================================================"
