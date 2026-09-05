#!/usr/bin/env bash
set -euo pipefail

# Defaults reproduce the vLLM service used for the 2026-09-03 CVM campaign.
# Override these variables when the bare-metal filesystem layout differs.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_VENV="${VLLM_VENV:-/data/venvs/vllm025}"
MODEL_PATH="${MODEL_PATH:-/models/DeepSeek-V4-Flash-0731}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-DeepSeek-V4-Flash-0731}"
PORT="${PORT:-8000}"
TMUX_SESSION="${TMUX_SESSION:-vllm-server}"
WORK_DIR="${WORK_DIR:-${ROOT}}"
PROFILE_DIR="${PROFILE_DIR:-/data/vllm-profiles/v025-off-shapes}"
LOG_FILE="${LOG_FILE:-/data/vllm-logs/v025-tmux.log}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
CUDA_PY_ROOT="${CUDA_PY_ROOT:-${VLLM_VENV}/lib/python${PYTHON_VERSION}/site-packages/nvidia/cu13}"
TORCH_LIB_ROOT="${TORCH_LIB_ROOT:-${VLLM_VENV}/lib/python${PYTHON_VERSION}/site-packages/torch/lib}"

if [[ ! -x "${VLLM_VENV}/bin/vllm" ]]; then
  echo "vLLM executable not found: ${VLLM_VENV}/bin/vllm" >&2
  exit 1
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Model directory not found: ${MODEL_PATH}" >&2
  exit 1
fi
if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required" >&2
  exit 1
fi
if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${TMUX_SESSION}" >&2
  exit 1
fi

mkdir -p "$(dirname "${LOG_FILE}")" "${PROFILE_DIR}"

SERVICE_PATH="${CUDA_PY_ROOT}/bin:${VLLM_VENV}/bin:/usr/local/cuda/bin:/usr/bin:/bin"
SERVICE_LD_LIBRARY_PATH="${TORCH_LIB_ROOT}:${CUDA_PY_ROOT}/lib:/usr/lib/x86_64-linux-gnu"
PROFILER_CONFIG="$(printf '{\"profiler\":\"torch\",\"torch_profiler_dir\":\"%s\",\"torch_profiler_with_stack\":false,\"torch_profiler_record_shapes\":true,\"torch_profiler_with_memory\":false,\"torch_profiler_with_flops\":false,\"torch_profiler_dump_cuda_time_total\":false,\"torch_profiler_use_gzip\":true}' "${PROFILE_DIR}")"

service_argv=(
  env
  "VLLM_DEEP_GEMM_WARMUP=${VLLM_DEEP_GEMM_WARMUP:-skip}"
  "CUDA_HOME=${CUDA_PY_ROOT}"
  "CUDA_PATH=${CUDA_PY_ROOT}"
  "PATH=${SERVICE_PATH}"
  "LD_LIBRARY_PATH=${SERVICE_LD_LIBRARY_PATH}"
  "${VLLM_VENV}/bin/vllm" serve "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host 0.0.0.0
  --port "${PORT}"
  --tensor-parallel-size 8
  --gpu-memory-utilization 0.90
  --max-model-len 32768
  --trust-remote-code
  --tokenizer-mode deepseek_v4
  --reasoning-parser deepseek_v4
  --kv-cache-dtype fp8
  --block-size 256
  --enable-expert-parallel
  --moe-backend marlin
  --linear-backend deep_gemm
  --profiler-config "${PROFILER_CONFIG}"
)

printf -v service_command '%q ' "${service_argv[@]}"
printf -v log_file_quoted '%q' "${LOG_FILE}"
service_command+="2>&1 | tee -a ${log_file_quoted}"

tmux new-session -d -s "${TMUX_SESSION}" -c "${WORK_DIR}" "${service_command}"

echo "Started tmux session: ${TMUX_SESSION}"
echo "Log: ${LOG_FILE}"
echo "Profile output: ${PROFILE_DIR}"
echo "Inspect: tmux attach -t ${TMUX_SESSION}"
echo "Health: curl --noproxy '*' http://127.0.0.1:${PORT}/health"
