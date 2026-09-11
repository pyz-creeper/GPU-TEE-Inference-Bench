#!/usr/bin/env bash
# Independent environment; existing inference environments are never modified.
set -Eeuo pipefail
env_dir=/data/envs/glm53-vllm-pp2-mtp
base_python=/opt/miniforge3/envs/glm52-sglang/bin/python
uv=/opt/miniforge3/envs/glm52-sglang/bin/uv
export UV_CACHE_DIR=/data/benchmarks/glm53-vllm-pp2-mtp/uv-cache
unset LD_PRELOAD LD_LIBRARY_PATH PYTHONPATH
mkdir -p /data/benchmarks/glm53-vllm-pp2-mtp
if [[ ! -x "$env_dir/bin/python" ]]; then
    "$base_python" -m venv "$env_dir"
fi
"$uv" pip install --python "$env_dir/bin/python" 'vllm==0.28.0' 'transformers==5.15.0' pytest \
    'nvidia-cuda-nvcc==13.0.88' 'nvidia-cuda-crt==13.0.88' \
    'nvidia-nvvm==13.0.88' 'nvidia-cuda-cccl==13.0.85'
"$uv" pip check --python "$env_dir/bin/python"
cuda_dir="$env_dir/lib/python3.12/site-packages/nvidia/cu13"
[[ -e "$cuda_dir/lib64" ]] || ln -s lib "$cuda_dir/lib64"
for cuda_lib in cudart nvrtc; do
    [[ -e "$cuda_dir/lib/lib${cuda_lib}.so" ]] || \
        ln -s "lib${cuda_lib}.so.13" "$cuda_dir/lib/lib${cuda_lib}.so"
done
"$env_dir/bin/python" -c 'import importlib.metadata as m; print({p:m.version(p) for p in ["vllm","torch","transformers"]})'
