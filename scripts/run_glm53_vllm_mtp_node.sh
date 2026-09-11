#!/usr/bin/env bash
set -Eeuo pipefail
[[ $# -ge 1 && "$1" =~ ^[01]$ ]] || { echo 'Usage: script NODE_RANK [MTP_TOKENS=1]'; exit 2; }
node_rank=$1
mtp_tokens=${2:-1}
[[ "$mtp_tokens" =~ ^[0-5]$ ]] || exit 2
env_dir=/data/envs/glm53-vllm-pp2-mtp
run_root=/data/benchmarks/glm53-vllm-pp2-mtp
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy LD_PRELOAD PYTHONPATH
unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS LD AR CMAKE_PREFIX_PATH CONDA_BUILD_SYSROOT
export CC=/usr/bin/gcc CXX=/usr/bin/g++ LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
export NO_PROXY='*' no_proxy='*'
export PATH="$env_dir/bin:/usr/bin:/bin"
export CUDA_HOME="$env_dir/lib/python3.12/site-packages/nvidia/cu13"
export LIBRARY_PATH="$CUDA_HOME/lib:/usr/lib/x86_64-linux-gnu"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:/opt/miniforge3/envs/glm52-sglang/lib"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONUNBUFFERED=1
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_HOST_IP=192.168.0.63
export VLLM_GLM53_UVA_FIX=1
unset VLLM_GLM53_DEBUG_SAMPLING
export TORCH_EXTENSIONS_DIR="$run_root/torch-extensions"
export VLLM_SERVER_DEV_MODE=1
[[ "$node_rank" == 0 ]] || export VLLM_HOST_IP=192.168.0.65
export NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0 NCCL_IB_DISABLE=1 NCCL_SOCKET_FAMILY=AF_INET
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
export HF_HUB_OFFLINE=1 HF_HOME="$run_root/huggingface"
export VLLM_CACHE_ROOT="$run_root/cache-node$node_rank"
export TRITON_CACHE_DIR="$run_root/triton-node$node_rank"
export FLASHINFER_WORKSPACE_BASE="$run_root/flashinfer-node$node_rank"
mkdir -p "$run_root/logs" "$HF_HOME" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR"
log="$run_root/logs/node${node_rank}-k${mtp_tokens}-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee --ignore-interrupts "$log") 2>&1
if [[ "${3:-}" == --check-kernels ]]; then
  exec "$env_dir/bin/python" -c 'from flashinfer.jit.gemm.fp8_blockscale import gen_fp8_blockscale_gemm_sm90_module; gen_fp8_blockscale_gemm_sm90_module().build_and_load(); print("FlashInfer SM90 FP8 kernel compiled and loaded successfully", flush=True)'
fi
if [[ "${3:-}" == --check-uva ]]; then
  test_path="$(dirname "${BASH_SOURCE[0]}")/patches/test_glm53_uva_view.py"
  [[ -f "$test_path" ]] || test_path="$(dirname "${BASH_SOURCE[0]}")/test_glm53_uva_view.py"
  exec "$env_dir/bin/python" -m pytest -c /dev/null -q "$test_path"
fi
args=(serve /data/model/GLM-5.3 --served-model-name GLM-5.3
  --host 127.0.0.1 --port 30002 --tensor-parallel-size 8 --pipeline-parallel-size 2
  --distributed-executor-backend mp --nnodes 2 --node-rank "$node_rank"
  --master-addr 192.168.0.63 --master-port 29501
  --max-model-len 65536 --max-num-seqs 16 --max-num-batched-tokens 8192
  --gpu-memory-utilization 0.80 --kv-cache-dtype fp8
  --enable-prefix-caching --enable-chunked-prefill
  # Frozen SWE replay expects shell commands as text, with no OpenAI tools.
  # vLLM's streaming tool parser otherwise extracts spontaneous tool tags.
  --reasoning-parser glm45
  --generation-config auto --seed 42)
if [[ "$mtp_tokens" != 0 ]]; then
  args+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$mtp_tokens}")
fi
[[ "$node_rank" == 0 ]] || args+=(--headless)
printf 'vLLM 0.28.0 + PR46994 backport, TP8 PP2 MTP=%s node=%s log=%s\n' "$mtp_tokens" "$node_rank" "$log"
exec "$env_dir/bin/vllm" "${args[@]}"
