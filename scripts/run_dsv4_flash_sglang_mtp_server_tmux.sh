#!/usr/bin/env bash
set -uo pipefail

ENV_DIR=/opt/miniforge3/envs/sglang-dsv4-flash
MODEL_DIR=/data/model/DeepSeek-V4-Flash-0731
LOG_ROOT=/data/benchmarks/dsv4-flash-sglang-mtp/logs
STAMP=$(date +%Y%m%d-%H%M%S)
LOG_FILE=${LOG_ROOT}/server-${STAMP}.log
CUDA_PY_ROOT=${ENV_DIR}/lib/python3.12/site-packages/nvidia/cu13

mkdir -p "${LOG_ROOT}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY='*'
export no_proxy='*'
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_HOME=${CUDA_PY_ROOT}
export CUDA_PATH=${CUDA_PY_ROOT}
export PATH=${CUDA_PY_ROOT}/bin:${ENV_DIR}/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=${ENV_DIR}/lib/python3.12/site-packages/torch/lib:${CUDA_PY_ROOT}/lib:/usr/lib/x86_64-linux-gnu
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

echo "SGLang DeepSeek-V4-Flash starting; log=${LOG_FILE}"
"${ENV_DIR}/bin/python" -m sglang.launch_server \
  --model-path "${MODEL_DIR}" \
  --served-model-name deepseek-v4-flash-0731 \
  --trust-remote-code \
  --host 127.0.0.1 \
  --port 30000 \
  --tp-size 8 \
  --moe-runner-backend marlin \
  --context-length 65536 \
  --mem-fraction-static 0.90 \
  --cuda-graph-max-bs-decode 16 \
  --max-running-requests 16 \
  --speculative-algorithm DSPARK \
  --reasoning-parser deepseek-v4 \
  --tool-call-parser deepseekv4 \
  2>&1 | tee "${LOG_FILE}"
status=${PIPESTATUS[0]}

echo
echo "SGLang server exited with status ${status}. Log: ${LOG_FILE}"
echo "The tmux pane will remain at an interactive shell."
exec /bin/bash -i
