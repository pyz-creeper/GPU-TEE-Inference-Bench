# DSV4-Pro PP=2 CVM：原 20 个 SWE-agent case 的固定轨迹回放

交给已经完成 Mooncake 环境准备的远端 Codex 执行。目标是当前已启动的 SGLang DSV4-Pro PP=2；沿用其部署和客户端环境，不重启、不修改 PP/TP/MTP/显存参数。先确认 Mooncake 已结束，测试期间不要并行运行其他推理压测。

## 实验口径

复用原 `aliyun-high-20/bundle`，**20 个完整 session，376 次 Chat Completions 请求**。每个请求使用录制的完整历史，上一请求完成后才继续依赖它的请求；本轮模型输出不写回下一轮历史。不执行仓库修复、shell 工具或 SWE-bench 测试，因此这是轨迹 API 执行耗时，不是 20 个问题的现场求解或解决率评估。

保持 stream=true、thinking=true、reasoning_effort=high、每轮 max_tokens=4096、session/HTTP 并发均为 1、repeat=1、warmup=0、retries=0、单请求 timeout=300 秒。错误即停止并保留结果。4096 是每次生成的上限，不是整条轨迹总预算；不能沿用 Mooncake 的 128/ignore_eos，也不能使用 test.py 的 256。

保持 provider-default prefix cache、不 flush、不加 nonce；记录服务之前跑过 Mooncake，不能声称是刚启动的冷缓存。客户端准备与接口探测不计入正式 sender_total_s。不同模型的 high 不代表相同思考计算量。

## 1. 环境与冻结 bundle

在仓库根目录继续使用 `.venv-mooncake`。所有路径都放在 `/data/benchmarks`，不依赖本机旧部署的 conda 路径。

```bash
set -euo pipefail
source .venv-mooncake/bin/activate
export AGENTIC_ROOT=/data/benchmarks/agentic/dsv4-pro-pp2-cvm-high-20
export BUNDLE=/data/benchmarks/agentic/aliyun-high-20/bundle
mkdir -p "$AGENTIC_ROOT"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost,::1"
export no_proxy="$NO_PROXY"
unset ALIYUN_BASE_URL INPUT_BENCH_WORKLOAD INPUT_BENCH_RESULTS_ROOT
```

**Git 拉取不包含 bundle。优先迁移原始冻结文件，运行不需要下载全量数据集，也不需要安装 GLM tokenizer。** 若远端已有该 bundle，直接校验；否则从原实验机器复制以下三个文件，保持字节不变：

```bash
# 在远端执行；BENCH_SOURCE 改为可 SSH 访问的原实验机器 user@host。
export BENCH_SOURCE='替换为原实验机器的SSH地址'
mkdir -p "$BUNDLE"
for file in workload.jsonl manifest.json sessions.json; do
  scp "$BENCH_SOURCE:/root/GPU-TEE-Inference-Bench/runs/agentic-replay/aliyun-high-20/bundle/$file" "$BUNDLE/"
done
```

源目录约 17 MB。SSH 地址/访问方式未提供且本机没有 bundle 时，只读完成其他检查后向用户索取源机器连接方式；不要自行换一组 20 个 case。

验证内容、依赖关系与身份：

```bash
python - <<'PY'
import os
from pathlib import Path
from input_bench.replay_bundle import validate_bundle
bundle = Path(os.environ['BUNDLE'])
requests, manifest, sessions = validate_bundle(bundle)
assert manifest['workload_sha256'] == '8a8b4d97202f388b7719c232d69ec5181091d1b8bc61a8f89c4df36a7a3e60b1'
assert len(requests) == 376 and len(sessions['selected']) == 20
assert all(r.max_output_tokens == 4096 for r in requests)
print('Verified: 20 sessions / 376 requests / original workload SHA256')
PY
sha256sum "$BUNDLE/"*.json* > "$AGENTIC_ROOT/bundle-sha256.txt"
```

不要为适配 DSV4-Pro 改写 manifest 的 GLM reference tokenizer、重排/截断 messages 或重新计算并写回 input_tokens。它记录原编译口径；DSV4-Pro 的实际输入长度应另外检查。用当前服务实际 chat encoder/template 离线检查最大输入加 4096 是否落在服务上下文内，reference 的 32768 不能代替该检查。如果现有 tokenizer 无法复现服务编码，记录限制并先解决，不静默截断。

只有原 bundle 无法找回时才考虑重建：原数据集为 `nebius/SWE-agent-trajectories`，revision 为 `68195a1450865274106246d0d0296a1d6807b88e`。以下仅为恢复路径，不是默认执行步骤：

```bash
python -m pip install huggingface_hub
hf download nebius/SWE-agent-trajectories --repo-type dataset \
  --revision 68195a1450865274106246d0d0296a1d6807b88e \
  --local-dir /data/benchmarks/coding/swe-agent-trajectories
python scripts/prepare_agentic_replay.py \
  --config scenarios/aliyun_swe_trajectory_high.json --bundle "$BUNDLE"
```

恢复需要原 `/data/model/GLM-5.3` reference tokenizer 及兼容版本；不能改用 DSV4-Pro tokenizer。prepare 脚本附带历史百炼规模/费用报告，那不是本地实验预算，也不会调用 API。恢复后仍须验证原 workload hash、session/request IDs、manifest 内容身份；不匹配就停止，不能标成原实验复跑。

## 2. 确认目标与思考参数映射

沿用 Mooncake 查明的实际 served model ID、HTTP 入口、tokenizer 目录和部署快照。这里 `AGENTIC_BASE_URL` 必须包含 `/v1`：

```bash
export AGENTIC_BASE_URL=http://127.0.0.1:30000/v1  # 改成实际本地入口
export AGENTIC_MODEL='替换为 /v1/models 返回的 DSV4-Pro id'
```

仓库 runner 的 `sglang-dsv4-0731` 是参数映射名称，不强制模型 ID：它发送 `chat_template_kwargs={"thinking":true}` 和顶层 `reasoning_effort="high"`。该映射此前在 Flash 0731 上验证，**不能未经检查就声称适用于当前 Pro**。Codex 应读取当前安装的 SGLang DeepSeek V4 chat encoder、reasoning parser 和模型配置，确认支持该映射并记录文件位置/版本。不要套用百炼的顶层 enable_thinking。

可在正式计时前发送一次独立短 Chat Completions 探测（最多 128 输出 tokens，stream=true，使用同一思考参数），验证 HTTP/SSE、reasoning 字段和模型路由；不要使用轨迹 prompt。请求被接受只证明协议可用，不能证明服务内部 high 的实际计算量。若映射不兼容，停止并报告证据，不能删掉思考参数后继续；也不要运行旧 Flash/GLM 启动脚本。

本地服务无鉴权时使用 auth=none，不读取任何云端 key。若服务有鉴权，通过 shell 隐式输入 `BENCH_API_KEY`，生成配置时使用 auth=env、api_key_env=BENCH_API_KEY；不传 `--env-file`，不使用 DASHSCOPE_API_KEY 或 Codex 的 key。

## 3. 生成独立配置并离线检查

确认上述映射适用后执行：

```bash
python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['AGENTIC_ROOT'])
cfg = json.loads(Path('scenarios/sglang_dsv4_mtp_swe_trajectory_high.json').read_text())
cfg['status'] = 'DSV4-Pro PP2 CVM frozen replay; deployment verified separately'
cfg['bundle'] = str(Path(os.environ['BUNDLE']).resolve())
cfg['results_root'] = str(root / 'results')
model = os.environ['AGENTIC_MODEL']
run = cfg['planned_run']
run.update(models=[model], base_url=os.environ['AGENTIC_BASE_URL'],
           auth='env' if os.getenv('BENCH_API_KEY') else 'none', api_key_env='BENCH_API_KEY')
run['output_cap_semantics'] = {model: 'requested max_tokens=4096; verify current server semantics and preserve usage'}
cfg['deployment'] = {'note': 'See separately recorded current Pro PP2 deployment; do not inherit Flash metadata'}
cfg['network'] = {'client_location': 'CVM benchmark client', 'api_region': 'local',
                  'cache': 'provider default; no flush/nonce; record previous Mooncake activity',
                  'rate_limit': 'session=1, HTTP=1; dedicated measurement window'}
path = root / 'config.json'
if path.exists():
    raise SystemExit('config exists: inspect/reuse it; do not overwrite previous experiment')
path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + '\n')
PY
python scripts/run_agentic_replay.py dry-run \
  --config "$AGENTIC_ROOT/config.json" --bundle "$BUNDLE" \
  --base-url "$AGENTIC_BASE_URL" --model all \
  | tee "$AGENTIC_ROOT/dry-run.log"
git rev-parse HEAD > "$AGENTIC_ROOT/repository-commit.txt"
python -m pip freeze > "$AGENTIC_ROOT/client-requirements.txt"
```

确认 dry-run 输出只有当前 Pro 一个模型、20 sessions、376 requests、high、4096、并发 1、stream 和本地 URL。dry-run 不调用 API，也不核验服务已就绪。将实际 TP/PP/MTP、GPU/节点、模型修订、上下文、SGLang/encoder/parser 版本、机密计算证据、缓存状态写入单独的 `deployment.md`。不能沿用旧配置的 Flash 权重 hash 或 DSpark 参数。

## 4. 正式运行一次

当前用户已授权完成这轮本地回放；预检通过就执行，不再询问预算或普通实现选择。Codex 留在当前 tmux 会话持续观察，不需要另开 Codex。可运行数小时，不能从旧 GLM 的约 120 分钟推断 Pro 必然更快。

```bash
export RUN_ID="dsv4-pro-pp2-cvm-high-20-$(date -u +%Y%m%dT%H%M%SZ)"
python -u scripts/run_agentic_replay.py run \
  --config "$AGENTIC_ROOT/config.json" --bundle "$BUNDLE" \
  --base-url "$AGENTIC_BASE_URL" --model all \
  --output-root "$AGENTIC_ROOT/results" --run-id "$RUN_ID" \
  2>&1 | tee "$AGENTIC_ROOT/$RUN_ID.log"
```

不传 `--continue-on-error`。如 300 秒超时，保存失败轮次；不要自动增加 timeout、补跑剩余请求后拼接总耗时。中断也不能 SIGSTOP/SIGCONT 暂停计时，后续完整重跑须新 run-id 并记录参数变化。现有模型进程始终保留。

## 5. 结果核验与报告

实际模型目录名由 runner 规范化，在 `$AGENTIC_ROOT/results/$RUN_ID/` 下查找，不硬编码 Pro 名称。对每个 summary 检查：

- `complete=true`，planned/completed=376；failed/cancelled/not_sent/missing 均为 0；20 个 sessions 均 complete。
- 保存 `effective-config.json`、`events.jsonl`、summary、会话明细和 runner 生成的其余工件。核对 request/session IDs 和 bundle identity。
- 报告 `measurement.sender_total_s` 及分钟数，每 case 的 `session_duration_s`、轮数和 instance_id（从 sessions.json 映射）。sender 总时间包含调度等开销，不能直接当作纯 GPU 时间。
- 报告 request E2E、ttft_observed、answer_ttft 的分位数；首个 reasoning 与首个 answer 的时延分开。当前 runner 不提供可靠 reasoning TPOT，不自行用 SSE chunk 数冒充 token 数。
- 报告 server_prompt/server_completion、unknown_usage_requests、answerless、finish_reasons 和输出 tokens/s。`finish_reason=length` 或无最终答案仍可能是完整性能回放，但必须单列，不能算问题解决成功；reasoning 通常已包含在 completion 中，不能重复相加。

写入 `$AGENTIC_ROOT/report.md`，包含部署证据、开始/结束 UTC、输入 hash、普通选择及异常。只运行当前 Pro，一轮 376 请求；不顺带重跑云端或其他本地模型。

如已迁移之前的完整结果目录，可离线比较（首目录为基线，ratio=本轮/基线，>1 表示更慢）：

```bash
python scripts/run_agentic_replay.py compare \
  /替换为基线模型结果目录 /替换为本轮模型结果目录 \
  --output "$AGENTIC_ROOT/comparisons/$RUN_ID"
```

比较器会重建结果并核验完整性和输入契约；不匹配时不能手算一个“有效加速比”绕过检查。Flash/GLM/百炼与 Pro 的比较仅代表端点体验，权重、输出长度、缓存、思考实现和网络可能不同。CC 开销需要同权重 Pro、同 PP/TP/MTP、硬件和同 bundle 的非机密基线。没有基线时交付本轮绝对指标即可。
