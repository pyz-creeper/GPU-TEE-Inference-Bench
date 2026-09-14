#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
unset ALIYUN_BASE_URL INPUT_BENCH_WORKLOAD INPUT_BENCH_RESULTS_ROOT
export NO_PROXY=127.0.0.1,localhost
export no_proxy="$NO_PROXY"
export PYTHONUNBUFFERED=1
PYTHON=/opt/miniforge3/envs/test-api/bin/python
RUN_ID=${1:-sglang-dsv4-dspark-high-20-r1}
AUDIT_ROOT="runs/agentic-replay/sglang-dsv4-mtp-high-20/launches/${RUN_ID}"
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
        with opener.open('http://127.0.0.1:30000/health', timeout=5) as response:
            if response.status == 200:
                break
    except OSError:
        pass
    if time.monotonic() > deadline:
        raise SystemExit('Local server did not become healthy within 20 minutes; inspect server log.')
    print('Waiting for local SGLang readiness...', flush=True)
    time.sleep(10)
with opener.open('http://127.0.0.1:30000/server_info', timeout=10) as response:
    info = json.load(response)
args = info.get('server_args', info)
required = {'served_model_name':'deepseek-v4-flash-0731', 'speculative_algorithm':'DSPARK',
            'speculative_num_draft_tokens':6, 'tp_size':8, 'reasoning_parser':'deepseek-v4'}
for key, value in required.items():
    if args.get(key) != value:
        raise SystemExit(f'Server configuration mismatch: {key} expected {value!r}, got {args.get(key)!r}')
fields = list(required) + ['model_path','context_length','moe_runner_backend','kv_cache_dtype',
                         'mem_fraction_static','max_running_requests','disable_radix_cache',
                         'sampling_defaults','random_seed','skip_server_warmup']
snapshot = {'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'server_args':{k:args.get(k) for k in fields},
            'sglang_version':info.get('version'), 'note':'Local readiness only; no trajectory warmup.'}
(root/'server-before.json').write_text(json.dumps(snapshot, indent=2)+'\n')
print('Verified local SGLang model, TP=8 and DSpark (5 proposed tokens).', flush=True)
PY

exec "$PYTHON" scripts/run_agentic_replay.py run \
  --config scenarios/sglang_dsv4_mtp_swe_trajectory_high.json \
  --base-url http://127.0.0.1:30000/v1 \
  --model all --run-id "$RUN_ID"
