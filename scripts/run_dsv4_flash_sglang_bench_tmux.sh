#!/usr/bin/env bash
set -uo pipefail

REPO=/root/GPU-TEE-Inference-Bench
CLIENT_ENV=/opt/miniforge3/envs/glm52-sglang
CONFIG=${REPO}/scenarios/dsv4_flash_sglang_single_node.json
LOG_ROOT=/data/benchmarks/dsv4-flash-sglang-single-node/logs
STAMP=$(date +%Y%m%d-%H%M%S)
LOG_FILE=${LOG_ROOT}/campaign-${STAMP}.log

mkdir -p "${LOG_ROOT}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY='*'
export no_proxy='*'
export PYTHONPATH=${REPO}/src

cd "${REPO}"
echo "SGLang DeepSeek-V4-Flash campaign starting; log=${LOG_FILE}"
"${CLIENT_ENV}/bin/python" scripts/glm52_pp2_campaign.py all --config "${CONFIG}" 2>&1 | tee "${LOG_FILE}"
status=${PIPESTATUS[0]}

echo
echo "Campaign exited with status ${status}. Log: ${LOG_FILE}"
echo "The tmux pane will remain at an interactive shell."
exec /bin/bash -i
