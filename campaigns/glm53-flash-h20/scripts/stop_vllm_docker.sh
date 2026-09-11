#!/usr/bin/env bash
set -euo pipefail

readonly NAME='glm53-vllm'
readonly LOG='/root/glm53-flash/logs/vllm-server.log'

if ! docker ps -a --format '{{.Names}}' | grep -Fxq "${NAME}"; then
  echo "container ${NAME} does not exist"
  exit 0
fi

docker logs "${NAME}" >"${LOG}" 2>&1
docker stop --time 120 "${NAME}"
docker rm "${NAME}"
echo "saved ${LOG} and removed ${NAME}"
