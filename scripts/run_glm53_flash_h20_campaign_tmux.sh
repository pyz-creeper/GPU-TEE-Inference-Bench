#!/usr/bin/env bash
set -Eeuo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
env_dir="${GLM53_ENV_DIR:-/opt/miniforge3/envs/glm53-sglang}"
scenario="${GLM53_SCENARIO:-$repo_dir/scenarios/glm53_flash_sglang_h20_mtp.json}"
log_dir="${GLM53_CAMPAIGN_LOG_DIR:-/data/benchmarks/glm53-flash-mooncake-h20-mtp/logs}"
mkdir -p "$log_dir"
log_file="$log_dir/campaign-$(date +%Y%m%d-%H%M%S).log"

# Keep benchmark and trace preparation off the host's Clash proxy.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY="*"
export no_proxy="*"
export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"

cd "$repo_dir"
printf 'GLM-5.3-Flash H20 Mooncake MTP-on campaign starting; log=%s\n' "$log_file"
set +e
"$env_dir/bin/python" scripts/glm52_pp2_campaign.py all \
  --config "$scenario" "$@" 2>&1 | tee -a "$log_file"
status="${PIPESTATUS[0]}"
set -e
printf '\nCampaign exited with status %s. Log: %s\n' "$status" "$log_file"
printf 'The tmux pane will remain at an interactive shell.\n'
exec /bin/bash -i
