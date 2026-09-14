#!/usr/bin/env bash
set -Eeuo pipefail

readonly MODEL_DIR=/data/model/GLM-5.3-Flash
readonly ENV_DIR=/opt/miniforge3/envs/glm53-sglang
readonly CACHE_ROOT=/data/sglang-cache

export CUDA_VISIBLE_DEVICES="${GLM_GPU_IDS:-0,1,2,3,4,5,6,7}"
export PYTHONUNBUFFERED=1
export CUDA_HOME="${ENV_DIR}/lib/python3.12/site-packages/nvidia/cu13"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${CACHE_ROOT}/huggingface"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/torchinductor"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"

readonly TP_SIZE="${GLM_TP_SIZE:-8}"
readonly EP_SIZE="${GLM_EP_SIZE:-8}"
readonly HOST="${GLM_HOST:-127.0.0.1}"
readonly PORT="${GLM_PORT:-30000}"
readonly CONTEXT_LENGTH="${GLM_CONTEXT_LENGTH:-32768}"
readonly MEM_FRACTION="${GLM_MEM_FRACTION:-0.70}"

test -x "${ENV_DIR}/bin/python"
test -f "${MODEL_DIR}/.download-complete"
mkdir -p "${HF_HOME}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"

exec "${ENV_DIR}/bin/python" -m sglang.launch_server \
  --model-path "${MODEL_DIR}" \
  --served-model-name GLM-5.3-Flash \
  --enable-multimodal \
  --tp-size "${TP_SIZE}" \
  --ep-size "${EP_SIZE}" \
  --attention-backend dsa \
  --dsa-prefill-backend tilelang \
  --dsa-decode-backend tilelang \
  --linear-attn-backend triton \
  --kv-cache-dtype bfloat16 \
  --quantization fp8 \
  --moe-runner-backend deep_gemm \
  --context-length "${CONTEXT_LENGTH}" \
  --max-running-requests 32 \
  --chunked-prefill-size 8192 \
  --max-prefill-tokens 8192 \
  --mem-fraction-static "${MEM_FRACTION}" \
  --cuda-graph-backend-decode full \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 6 \
  --speculative-adaptive \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --host "${HOST}" \
  --port "${PORT}"
