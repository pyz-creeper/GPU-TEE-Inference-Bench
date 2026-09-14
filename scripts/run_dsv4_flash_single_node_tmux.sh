#!/usr/bin/env bash
set -Eeuo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
client_env="${INPUT_BENCH_ENV:-/opt/miniforge3/envs/glm52-sglang}"
config="$repo_dir/scenarios/dsv4_flash_single_node.json"
log_dir="/data/benchmarks/dsv4-flash-single-node/logs"
mkdir -p "$log_dir"
log_file="$log_dir/campaign-$(date +%Y%m%d-%H%M%S).log"

# Keep both the local API traffic and any data access away from Clash.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY="*"
export no_proxy="*"
export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"

cd "$repo_dir"
printf 'DeepSeek-V4-Flash single-node campaign starting; log=%s\n' "$log_file"
set +e
"$client_env/bin/python" scripts/glm52_pp2_campaign.py all --config "$config" "$@" \
  2>&1 | tee -a "$log_file"
status="${PIPESTATUS[0]}"
set -e
printf '\nCampaign exited with status %s. Log: %s\n' "$status" "$log_file"
printf 'The tmux pane will remain at an interactive shell.\n'
exec /bin/bash -i
