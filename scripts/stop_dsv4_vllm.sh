#!/usr/bin/env bash
set -euo pipefail

TMUX_SESSION="${TMUX_SESSION:-vllm-server}"
PORT="${PORT:-8000}"
WAIT_SECONDS="${WAIT_SECONDS:-90}"

if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
  tmux send-keys -t "${TMUX_SESSION}" C-c
  deadline=$((SECONDS + WAIT_SECONDS))
  while tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; do
    if (( SECONDS >= deadline )); then
      echo "Timed out waiting for ${TMUX_SESSION} to exit; no forced kill was issued." >&2
      exit 1
    fi
    sleep 1
  done
else
  echo "tmux session is not running: ${TMUX_SESSION}"
fi

if ss -ltn 2>/dev/null | grep -Eq ":${PORT}[[:space:]]"; then
  echo "Port ${PORT} is still listening after tmux exited." >&2
  exit 1
fi

echo "Stopped ${TMUX_SESSION}; port ${PORT} is not listening."
