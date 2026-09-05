#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:?usage: $0 cvm|baremetal|vm [performance|profile] [base-url] [vllm|sglang]}"
SUITE="${2:-performance}"
BASE_URL="${3:-}"
BACKEND="${4:-}"

cd "$ROOT"
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="$NO_PROXY"
command=(python3 scripts/large_scale_campaign.py run --mode "$MODE" --suite "$SUITE")
if [[ -n "$BASE_URL" ]]; then
  command+=(--base-url "$BASE_URL")
fi
if [[ -n "$BACKEND" ]]; then
  command+=(--backend "$BACKEND")
fi
exec "${command[@]}"
