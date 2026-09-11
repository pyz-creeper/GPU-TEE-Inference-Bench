#!/usr/bin/env bash
set -euo pipefail

# Dedicated GLM-5.3-Flash build published by vLLM, pinned for reproducibility.
readonly IMAGE='vllm/vllm-openai@sha256:2e771fa615452282cc331eb418b3ef21636fce355bea0491fca89e6d362ab703'
readonly NAME='glm53-vllm'

if [[ "${GLM53_ALLOW_KNOWN_BAD_VLLM:-0}" != "1" ]]; then
  echo "refusing to start: this pinned day-0 vLLM image produced degenerate repeated-token output on 8x H20" >&2
  echo "set GLM53_ALLOW_KNOWN_BAD_VLLM=1 only to reproduce the documented failure" >&2
  exit 2
fi

if docker ps -a --format '{{.Names}}' | grep -Fxq "${NAME}"; then
  echo "container ${NAME} already exists; remove it explicitly before starting another instance" >&2
  exit 1
fi

mkdir -p /root/glm53-flash/cache/vllm

exec docker run --detach \
  --name "${NAME}" \
  --gpus all \
  --ipc host \
  --network host \
  --ulimit memlock=-1 \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env NO_PROXY=127.0.0.1,localhost \
  --env VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0 \
  --volume /data/model/GLM-5.3-Flash:/models/GLM-5.3-Flash:ro \
  --volume /root/glm53-flash/cache/vllm:/root/.cache \
  "${IMAGE}" \
  /models/GLM-5.3-Flash \
  --served-model-name GLM-5.3-Flash \
  --host 127.0.0.1 \
  --port 18000 \
  --tensor-parallel-size 8 \
  --max-model-len 32768 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.80 \
  --kv-cache-dtype bfloat16 \
  --disable-custom-all-reduce \
  --enforce-eager \
  --kernel-config '{"moe_backend":"deep_gemm","linear_backend":"deep_gemm","enable_flashinfer_autotune":false}' \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --no-enable-log-requests
