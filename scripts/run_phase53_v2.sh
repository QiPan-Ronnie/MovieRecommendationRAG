#!/bin/bash
# ============================================================
#  Phase 5.3 Perturbation Experiments E1-E4
#  Output:  results/RAG_Phase_5.3/perturbation_results.jsonl
#  Log:     logs/phase53.log
#  Usage:   nohup bash scripts/run_phase53_v2.sh >> logs/phase53.log 2>&1 &
# ============================================================
set -e
cd /root/autodl-tmp/MovieRecommendationRAG

OUTPUT_DIR="results/RAG_Phase_5.3"
LOG_DIR="logs"
PERTURB_FILE="${OUTPUT_DIR}/perturbation_results.jsonl"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo "============================================================"
echo "  Phase 5.3 Perturbation Experiments E1-E4"
echo "  Dir:    $(pwd)"
echo "  Output: ${OUTPUT_DIR}"
echo "  Time:   $(date +'%Y-%m-%d %H:%M:%S')"
echo "============================================================"

RAG_FILE="results/explanations_rag.jsonl"
if [ ! -f "${RAG_FILE}" ]; then
    echo "[ERROR] Missing ${RAG_FILE} - run Phase 5.2 first"
    exit 1
fi
RAG_COUNT=$(wc -l < "${RAG_FILE}")
echo "[Check] Phase 5.2 RAG results: ${RAG_COUNT} lines"

if [ -f "${PERTURB_FILE}" ]; then
    DONE_COUNT=$(wc -l < "${PERTURB_FILE}")
    echo "[Resume] Already done: ${DONE_COUNT} / 800"
else
    echo "[Resume] Fresh start"
fi

echo "[vLLM] Checking service..."
if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo "[vLLM] Service is online"
else
    echo "[vLLM] Not running, starting..."
    /root/miniconda3/envs/vllm/bin/python -m vllm.entrypoints.openai.api_server         --model /root/autodl-tmp/models/Qwen/Qwen2.5-7B-Instruct         --port 8000 --gpu-memory-utilization 0.90 --max-model-len 4096         >> logs/vllm_server.log 2>&1 &
    echo "[vLLM] Waiting up to 60s..."
    for i in $(seq 1 30); do
        sleep 2
        if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
            echo "[vLLM] Ready"
            break
        fi
    done
    if ! curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "[ERROR] vLLM not ready after 60s, check logs/vllm_server.log"
        exit 1
    fi
fi

echo ""
echo "[Pipeline] Starting perturbation experiments (200 x 4 = 800 calls)"
echo "[Pipeline] Time: $(date +'%Y-%m-%d %H:%M:%S')"
echo ""

HF_HUB_OFFLINE=1 python -m rag.pipeline     --phase 5.3     --llm-backend api     --api-url http://localhost:8000/v1     --model /root/autodl-tmp/models/Qwen/Qwen2.5-7B-Instruct     --num-samples 200     --output-dir "${OUTPUT_DIR}"

FINAL_COUNT=$(wc -l < "${PERTURB_FILE}" 2>/dev/null || echo 0)
echo ""
echo "============================================================"
echo "  Phase 5.3 DONE  Time: $(date +'%Y-%m-%d %H:%M:%S')"
echo "  Output: ${PERTURB_FILE}"
echo "  Lines:  ${FINAL_COUNT} / 800"
echo "============================================================"
