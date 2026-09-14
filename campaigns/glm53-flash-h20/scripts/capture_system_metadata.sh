#!/usr/bin/env bash
set -euo pipefail

readonly ARTIFACT_ROOT=/root/glm53-flash
readonly OUT="${ARTIFACT_ROOT}/system"
mkdir -p "${OUT}"

nvidia-smi > "${OUT}/nvidia-smi.txt"
nvidia-smi topo -m > "${OUT}/nvidia-topology.txt"
nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,driver_version,memory.total,memory.used,power.limit --format=csv > "${OUT}/gpu-inventory.csv"
lscpu > "${OUT}/lscpu.txt"
cp /etc/os-release "${OUT}/os-release.txt"
/opt/miniforge3/bin/conda env list > "${OUT}/conda-envs.txt"
/opt/miniforge3/bin/conda list -n glm53-sglang > "${OUT}/conda-glm53-sglang.txt"
/opt/miniforge3/bin/conda list -n glm53-vllm > "${OUT}/conda-glm53-vllm.txt"
docker image inspect vllm/vllm-openai@sha256:2e771fa615452282cc331eb418b3ef21636fce355bea0491fca89e6d362ab703 > "${OUT}/vllm-image-inspect.json"
git -C /root/GPU-TEE-Inference-Bench rev-parse HEAD > "${OUT}/input-bench-commit.txt"
git -C /data/src/sglang-glm53 rev-parse HEAD > "${OUT}/sglang-commit.txt"
git -C /data/src/sglang-glm53 status --short > "${OUT}/sglang-worktree-status.txt"
sha256sum /data/src/sglang-glm53/python/sglang/srt/model_executor/model_runner_components/load_model_utils.py > "${OUT}/sglang-dirty-file-sha256.txt"
sha256sum \
  /data/model/GLM-5.3-Flash/config.json \
  /data/model/GLM-5.3-Flash/model.safetensors.index.json \
  /data/model/GLM-5.3-Flash/tokenizer.json \
  /data/model/GLM-5.3-Flash/chat_template.jinja \
  > "${OUT}/model-file-sha256.txt"

echo "captured metadata in ${OUT}"
