#!/usr/bin/env bash
# Run on 192.168.0.65. Reuse the previously validated environment and launcher.
set -euo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY='*'
export no_proxy='*'
export GLM_CONTEXT_LENGTH=65536
export GLM_HOST=127.0.0.1
export GLM_PORT=30000
# tmux does not activate Conda. FlashInfer JIT invokes ninja by executable name.
export PATH="/opt/miniforge3/envs/glm53-sglang/bin:/usr/bin:/bin"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
command -v ninja
ninja --version
log_root=/data/benchmarks/glm53-flash-swe-mtp/logs
mkdir -p "$log_root"
log_file="$log_root/server-$(date +%Y%m%d-%H%M%S).log"
printf 'GLM-5.3-Flash MTP server log: %s\n' "$log_file"
exec > >(tee --ignore-interrupts "$log_file") 2>&1
exec /root/glm53-flash/scripts/start_sglang.sh
