#!/bin/bash
# ============================================================
# 启动 vLLM 推理服务
# 用法 (从项目根目录): bash scripts/start_vllm.sh
# 服务启动后监听 http://localhost:8000
# ============================================================

MODEL_PATH="/root/autodl-tmp/models/Qwen/Qwen2.5-7B-Instruct"
PORT=8000

echo "============================================================"
echo "  启动 vLLM 服务"
echo "  模型: $MODEL_PATH"
echo "  端口: $PORT"
echo "  时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# 检查是否已在运行
if curl -s http://localhost:$PORT/v1/models > /dev/null 2>&1; then
    echo "[OK] vLLM 服务已在运行，无需重复启动"
    curl -s http://localhost:$PORT/v1/models | python3 -c \
        "import sys,json; d=json.load(sys.stdin); print('  模型:', d['data'][0]['id'])"
    exit 0
fi

echo "[启动] 加载模型中（约 60-90 秒）..."
echo "[提示] 出现 'Application startup complete.' 后即可使用"
echo ""

/root/miniconda3/envs/vllm/bin/python \
    -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --port $PORT \
    --gpu-memory-utilization 0.90 \
    --max-model-len 4096 \
    --dtype auto
