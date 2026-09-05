#!/usr/bin/env bash
set -Eeuo pipefail

BENCHMARK_ROOT="${INPUT_BENCH_DATA_ROOT:-/data/benchmarks}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${BENCHMARK_ROOT}/.hf-cache}"
HF_MAX_WORKERS="${HF_MAX_WORKERS:-8}"

log() {
  printf '\n[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"
}

trap 'printf "\nDownload failed at line %s. Re-run this script to resume.\n" "$LINENO" >&2' ERR

mkdir -p \
  "${BENCHMARK_ROOT}/chat" \
  "${BENCHMARK_ROOT}/coding" \
  "${BENCHMARK_ROOT}/summarization" \
  "${BENCHMARK_ROOT}/traces/mooncake" \
  "${BENCHMARK_ROOT}/generators"

if ! command -v hf >/dev/null 2>&1; then
  log "Installing Hugging Face CLI"
  python3 -m pip install --user --upgrade huggingface_hub
  export PATH="${HOME}/.local/bin:${PATH}"
fi

for dependency in hf curl git; do
  command -v "${dependency}" >/dev/null 2>&1 || {
    echo "Missing dependency: ${dependency}" >&2
    exit 1
  }
done

log "Data root: ${BENCHMARK_ROOT}"
log "Hugging Face endpoint: ${HF_ENDPOINT}"

log "Downloading ShareGPT V3"
hf download learnanything/sharegpt_v3_unfiltered_cleaned_split \
  --repo-type dataset --local-dir "${BENCHMARK_ROOT}/chat/sharegpt-v3" \
  --max-workers "${HF_MAX_WORKERS}"

log "Downloading SWE-agent trajectories"
hf download nebius/SWE-agent-trajectories \
  --repo-type dataset --local-dir "${BENCHMARK_ROOT}/coding/swe-agent-trajectories" \
  --max-workers "${HF_MAX_WORKERS}"

log "Downloading SWE-bench Verified"
hf download SWE-bench/SWE-bench_Verified \
  --repo-type dataset --local-dir "${BENCHMARK_ROOT}/coding/swe-bench-verified" \
  --max-workers "${HF_MAX_WORKERS}"

log "Downloading Thoughtworks agentic coding trajectories"
hf download thoughtworks/agentic-coding-trajectories \
  --repo-type dataset --local-dir "${BENCHMARK_ROOT}/coding/thoughtworks-agentic-trajectories" \
  --max-workers "${HF_MAX_WORKERS}"

log "Downloading arXiv summarization validation and test sets"
hf download ccdv/arxiv-summarization \
  document/validation-00000-of-00001.parquet \
  document/test-00000-of-00001.parquet README.md \
  --repo-type dataset --local-dir "${BENCHMARK_ROOT}/summarization/arxiv" \
  --max-workers "${HF_MAX_WORKERS}"

log "Downloading LongBench"
hf download THUDM/LongBench data.zip \
  --repo-type dataset --local-dir "${BENCHMARK_ROOT}/summarization/longbench" \
  --max-workers "${HF_MAX_WORKERS}"

for trace in conversation_trace toolagent_trace synthetic_trace; do
  log "Downloading Mooncake ${trace}"
  curl --fail --show-error --location --retry 5 --retry-delay 2 \
    "https://raw.githubusercontent.com/kvcache-ai/Mooncake/refs/heads/main/FAST25-release/traces/${trace}.jsonl" \
    --output "${BENCHMARK_ROOT}/traces/mooncake/${trace}.jsonl"
done

if [[ -d "${BENCHMARK_ROOT}/generators/ServeGen/.git" ]]; then
  log "ServeGen already exists; keeping the checked-out revision"
else
  log "Cloning ServeGen"
  git clone --depth 1 https://github.com/alibaba/ServeGen.git \
    "${BENCHMARK_ROOT}/generators/ServeGen"
fi

log "ServeGen revision: $(git -C "${BENCHMARK_ROOT}/generators/ServeGen" rev-parse HEAD)"
log "All public benchmark downloads completed"
