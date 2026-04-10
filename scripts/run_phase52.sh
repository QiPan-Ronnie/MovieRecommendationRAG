#!/bin/bash
# ============================================================
# Phase 5.2 全量生成 (nohup 模式)
#
# 用法:
#   cd /root/autodl-tmp/MovieRecommendationRAG
#   nohup bash scripts/run_phase52.sh > logs/phase52_full.log 2>&1 &
#
# 特性:
#   - 自动检测/启动 vLLM 服务
#   - vLLM 崩溃后自动重启并续跑（最多重试 5 次）
#   - 断点续传，不删除已有数据
# ============================================================

set -e
cd "$(dirname "$0")/.."

PROJECT_DIR="$(pwd)"
LOG_DIR="logs"
mkdir -p "$LOG_DIR" results

VLLM_PYTHON="/root/miniconda3/envs/vllm/bin/python"
MODEL_PATH="/root/autodl-tmp/models/Qwen/Qwen2.5-7B-Instruct"
PORT=8000
MAX_RETRIES=5

echo "============================================================"
echo "  Phase 5.2 全量生成"
echo "  目录: $PROJECT_DIR"
echo "  时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# 记录已有进度
RAG_DONE=$(wc -l < results/explanations_rag.jsonl 2>/dev/null || echo 0)
PO_DONE=$(wc -l < results/explanations_prompt_only.jsonl 2>/dev/null || echo 0)
echo "[断点续传] RAG: ${RAG_DONE} 条 / Prompt-only: ${PO_DONE} 条"

start_vllm() {
    echo "[vLLM] 检查服务..."
    if curl -s http://localhost:$PORT/v1/models > /dev/null 2>&1; then
        echo "[vLLM] 已在运行"
        return 0
    fi

    echo "[vLLM] 未运行，正在启动..."
    $VLLM_PYTHON -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --port $PORT \
        --gpu-memory-utilization 0.90 \
        --max-model-len 4096 \
        --dtype auto > "$LOG_DIR/vllm_server.log" 2>&1 &
    VLLM_PID=$!
    echo "[vLLM] PID: $VLLM_PID, 等待启动..."

    for i in $(seq 1 60); do
        if curl -s http://localhost:$PORT/v1/models > /dev/null 2>&1; then
            echo "[vLLM] 启动成功 (${i}s)"
            return 0
        fi
        sleep 2
    done

    echo "[vLLM] 启动超时！"
    return 1
}

run_pipeline() {
    echo ""
    echo "[Pipeline] 启动生成..."
    echo "[Pipeline] 时间: $(date '+%Y-%m-%d %H:%M:%S')"

    HF_HUB_OFFLINE=1 python -m rag.pipeline \
        --phase 5.2 \
        --llm-backend api \
        --api-url http://localhost:$PORT/v1 \
        --model "$MODEL_PATH" \
        --concurrency 16

    return $?
}

# 主循环：自动重试
RETRY=0
while [ $RETRY -lt $MAX_RETRIES ]; do
    start_vllm
    if [ $? -ne 0 ]; then
        echo "[错误] vLLM 无法启动"
        RETRY=$((RETRY + 1))
        echo "[重试] $RETRY / $MAX_RETRIES, 等待 30 秒..."
        sleep 30
        continue
    fi

    run_pipeline
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "============================================================"
        echo "  Phase 5.2 完成！时间: $(date '+%Y-%m-%d %H:%M:%S')"
        RAG_FINAL=$(wc -l < results/explanations_rag.jsonl 2>/dev/null || echo 0)
        PO_FINAL=$(wc -l < results/explanations_prompt_only.jsonl 2>/dev/null || echo 0)
        echo "  RAG:         $RAG_FINAL 条"
        echo "  Prompt-only: $PO_FINAL 条"
        echo "============================================================"
        exit 0
    fi

    # Pipeline 失败，可能是 vLLM 崩溃
    RETRY=$((RETRY + 1))
    RAG_NOW=$(wc -l < results/explanations_rag.jsonl 2>/dev/null || echo 0)
    echo ""
    echo "[错误] Pipeline 退出码: $EXIT_CODE (已保存 $RAG_NOW 条)"
    echo "[重试] $RETRY / $MAX_RETRIES, 等待 30 秒后重启 vLLM..."
    sleep 30
done

echo "[失败] 达到最大重试次数 ($MAX_RETRIES)"
exit 1
