#!/usr/bin/env bash
# Run the same file on both nodes; node 0 is 192.168.0.63.
set -Eeuo pipefail
if [[ $# -ne 1 || ! "$1" =~ ^[01]$ ]]; then
    printf 'Usage: %s NODE_RANK(0|1)\n' "$0" >&2
    exit 2
fi
node_rank=$1
env_dir=/opt/miniforge3/envs/glm52-sglang
model_dir=/data/model/GLM-5.3
cache_root=/data/sglang-cache-glm53-pp2
log_root=/data/benchmarks/glm53-pp2-swe/logs
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
unset LD_PRELOAD PYTHONPATH
export NO_PROXY='*' no_proxy='*'
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONUNBUFFERED=1
export CUDA_HOME="$env_dir/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$env_dir/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$CUDA_HOME/lib"
export CC=/usr/bin/gcc CXX=/usr/bin/g++
export HF_HOME="$cache_root/huggingface"
export TORCHINDUCTOR_CACHE_DIR="$cache_root/torchinductor"
export TRITON_CACHE_DIR="$cache_root/triton"
# Preserve the established two-node TCP transport; no RDMA configuration change.
export NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1 NCCL_SOCKET_FAMILY=AF_INET
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
export SGLANG_UNBALANCED_MODEL_LOADING_TIMEOUT_S=3600
test -x "$env_dir/bin/python"
test -f "$model_dir/model.safetensors.index.json"
command -v ninja
mkdir -p "$HF_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$log_root"
log_file="$log_root/node${node_rank}-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee --ignore-interrupts "$log_file") 2>&1
printf 'GLM-5.3 FP8: node=%s TP=8 PP=2 MTP=off log=%s\n' "$node_rank" "$log_file"
exec "$env_dir/bin/python" -m sglang.launch_server \
    --model-path "$model_dir" --served-model-name GLM-5.3 \
    --host 127.0.0.1 --port 30000 \
    --dist-init-addr 192.168.0.63:29500 --nnodes 2 --node-rank "$node_rank" \
    --tp-size 8 --pp-size 2 --disable-overlap-schedule \
    --quantization fp8 --kv-cache-dtype bfloat16 \
    --attention-backend dsa --dsa-prefill-backend flashmla_sparse \
    --dsa-decode-backend fa3 --dsa-topk-backend sgl-kernel \
    --context-length 65536 --mem-fraction-static 0.90 \
    --max-running-requests 16 --chunked-prefill-size 8192 --max-prefill-tokens 16384 \
    --cuda-graph-max-bs-decode 16 --cuda-graph-bs-decode 1 2 4 8 16 \
    --disable-prefill-cuda-graph \
    --reasoning-parser glm45 --tool-call-parser glm47 \
    --watchdog-timeout 1200 --decode-log-interval 50
