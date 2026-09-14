#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
unset ALIYUN_BASE_URL INPUT_BENCH_WORKLOAD INPUT_BENCH_RESULTS_ROOT
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost PYTHONUNBUFFERED=1
python=/opt/miniforge3/envs/test-api/bin/python
run_id=${1:-vllm-glm53-pp2-mtp-high-20-r1}
audit="runs/agentic-replay/vllm-glm53-pp2-mtp-high-20/launches/$run_id"
mkdir -p "$(dirname "$audit")"
mkdir "$audit"
exec > >(tee --ignore-interrupts "$audit/launch.log") 2>&1
"$python" - "$audit" <<'PY'
import json, sys, urllib.request
from pathlib import Path
root = Path(sys.argv[1])
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
cfg = json.loads(Path('scenarios/vllm_glm53_pp2_mtp_swe_trajectory_high.json').read_text())
with opener.open('http://127.0.0.1:30002/server_info?config_format=json', timeout=120) as r:
    info = json.load(r)
actual = info['vllm_config']
assert actual['parallel_config']['tensor_parallel_size'] == 8
assert actual['parallel_config']['pipeline_parallel_size'] == 2
assert actual['model_config']['model'] == '/data/model/GLM-5.3'
assert actual['model_config']['max_model_len'] == 65536
assert actual['speculative_config']['method'] == 'mtp'
assert actual['speculative_config']['num_speculative_tokens'] == cfg['deployment']['num_speculative_tokens']
assert info['vllm_env']['VLLM_USE_V2_MODEL_RUNNER']
(root/'server-before.json').write_text(json.dumps({'vllm_config':actual, 'model_runner_v2':True},indent=2)+'\n')
with opener.open('http://127.0.0.1:30002/metrics', timeout=10) as r:
    (root/'metrics-before.txt').write_bytes(r.read())
print('Verified GLM-5.3, TP8/PP2, V2 runner and MTP configuration.')
PY
set +e
"$python" scripts/run_agentic_replay.py run \
  --config scenarios/vllm_glm53_pp2_mtp_swe_trajectory_high.json \
  --base-url http://127.0.0.1:30002/v1 --model all --run-id "$run_id"
status=$?
set -e
"$python" - "$audit" <<'PY'
import sys,time,urllib.request
from pathlib import Path
time.sleep(2) # let final scheduler counters reach the API metrics exporter
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open('http://127.0.0.1:30002/metrics',timeout=10) as r:
    (Path(sys.argv[1])/'metrics-after.txt').write_bytes(r.read())
PY
exit "$status"
