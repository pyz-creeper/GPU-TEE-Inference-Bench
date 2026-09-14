#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
unset ALIYUN_BASE_URL INPUT_BENCH_WORKLOAD INPUT_BENCH_RESULTS_ROOT
export NO_PROXY=127.0.0.1,localhost
export no_proxy="$NO_PROXY"
export PYTHONUNBUFFERED=1
PYTHON=/opt/miniforge3/envs/test-api/bin/python
RUN_ID=${1:-sglang-glm53-flash-mtp-high-20-r1}
AUDIT_ROOT="runs/agentic-replay/sglang-glm53-flash-mtp-high-20/launches/${RUN_ID}"
mkdir -p "$(dirname "$AUDIT_ROOT")"
mkdir "$AUDIT_ROOT"
exec > >(tee --ignore-interrupts "$AUDIT_ROOT/launch.log") 2>&1

# Local deployment checks only. /health may generate one internal probe token;
# this is outside measurement and never replays a trajectory prompt.
"$PYTHON" - "$AUDIT_ROOT" <<'PY'
import datetime, json, sys, time, urllib.request
from pathlib import Path
root = Path(sys.argv[1])
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
deadline = time.monotonic() + 1200
while True:
    try:
        with opener.open('http://127.0.0.1:30001/health', timeout=5) as response:
            if response.status == 200:
                break
    except OSError:
        pass
    if time.monotonic() > deadline:
        raise SystemExit('Local server did not become healthy within 20 minutes; inspect server log.')
    print('Waiting for local SGLang readiness...', flush=True)
    time.sleep(10)
with opener.open('http://127.0.0.1:30001/server_info', timeout=10) as response:
    info = json.load(response)
args = info.get('server_args', info)
required = {'served_model_name':'GLM-5.3-Flash', 'speculative_algorithm':'EAGLE',
            'speculative_num_steps':5, 'speculative_eagle_topk':1, 'speculative_num_draft_tokens':6,
            'speculative_adaptive':True, 'tp_size':8, 'ep_size':8,
            'context_length':65536, 'reasoning_parser':'glm45'}
for key, value in required.items():
    if args.get(key) != value:
        raise SystemExit(f'Server configuration mismatch: {key} expected {value!r}, got {args.get(key)!r}')
fields = list(required) + ['model_path','context_length','moe_runner_backend','kv_cache_dtype',
                         'mem_fraction_static','max_running_requests','disable_radix_cache',
                         'sampling_defaults','random_seed','skip_server_warmup']
snapshot = {'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'server_args':{k:args.get(k) for k in fields},
            'sglang_version':info.get('version'), 'note':'Remote readiness via SSH tunnel; health may generate an internal token before measurement; no trajectory warmup.'}
(root/'server-before.json').write_text(json.dumps(snapshot, indent=2)+'\n')
print('Verified remote GLM-5.3-Flash, 64K context, TP/EP=8 and adaptive EAGLE MTP 5/1/6.', flush=True)
PY

exec "$PYTHON" scripts/run_agentic_replay.py run \
  --config scenarios/sglang_glm53_flash_mtp_swe_trajectory_high.json \
  --base-url http://127.0.0.1:30001/v1 \
  --model all --run-id "$RUN_ID"
