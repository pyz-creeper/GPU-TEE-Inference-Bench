# DSV4-Pro / SGLang / PP=2 机密计算节点：Mooncake 执行手册

供远端 Codex 从 `mooncake` 分支执行。前提是 DSV4-Pro 已通过 SGLang 启动，客户端在提供 HTTP 接口的节点上运行。这里只安装 CPU 测试客户端；保留现有模型进程、SGLang 环境、PP/TP/MTP 和显存参数。另一 PP worker 不需要重复安装客户端或发起第二份测试。

这是 Mooncake FAST'25 trace 的合成 prompt 性能回放，走 `/v1/completions`，不是 SWE agent trajectory、正确率评测或 Mooncake Transfer Engine 带宽测试。首次默认复用仓库现有小规模 campaign：

| 测试点 | 请求数 | 并发/到达方式 |
| --- | ---: | --- |
| conversation | 12 × 2 | 闭环并发 1、4 |
| toolagent | 24 × 2 | 闭环并发 4、8 |
| toolagent-trace | 48 | 时间戳到达，提交跨度 60 秒，并发上限 16 |

合计 **120 次正式请求，5 个测试点**，另有 1 次接口检查。输入筛选 1,024–24,576 tokens；输出上限 128，`ignore_eos=true`。trace 到达跨度不等于总完成时间；128 是本轮长度控制，不是模型能力上限。默认每点只跑一次，适合先完成一轮迁移验证；统计稳定性和饱和吞吐需要另行扩样、重复。

## 1. 拉取与启动 Codex（用户执行）

新机器首次拉取：

```bash
git clone --branch mooncake https://github.com/pyz-creeper/GPU-TEE-Inference-Bench.git
cd GPU-TEE-Inference-Bench
```

已有仓库先 `git status --short`，保留未提交内容，再在 `mooncake` 分支 `git pull --ff-only`。不要 reset/clean 覆盖工作。

```bash
tmux new -s mooncake-cvm
# 在这个 tmux shell 中启用你自己的代理，然后进入仓库根目录。
mkdir -p /data/benchmarks
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost,::1"
export no_proxy="$NO_PROXY"
codex login status
codex --cd "$PWD" --no-alt-screen \
  --sandbox workspace-write --ask-for-approval on-request \
  --add-dir /data/benchmarks \
  -c sandbox_workspace_write.network_access=true \
  '完整阅读 docs/MOONCAKE_DSV4_PRO_PP2_CVM_RUNBOOK.md，按文档实际完成环境准备、下载、接口检查、冻结 workload、5 个测试点和结果核验。目标是本机已启动的 SGLang DSV4-Pro PP=2。先只读确认 endpoint、served model、tokenizer 和上下文长度，普通选择自行判断并记录。保留未提交改动和现有模型进程，不修改服务环境或部署参数，不向百炼发请求。正式测试使用独立 BENCH_API_KEY，不能使用 Codex 的密钥。若已有同模型基线 bundle，优先复用，不能重新采样冒充同一输入。完成后用中文报告参数、规模、失败数、耗时、性能指标和结果路径，不要只输出计划。'
```

`--no-alt-screen` 保留 tmux 滚动历史；workspace-write 配合 `--add-dir` 允许写仓库及数据目录，网络开关允许下载和调用本地接口。越界操作仍可能需要交互批准。参数参考 [Codex CLI 官方文档](https://developers.openai.com/codex/cli/reference) 与 [网络/沙箱配置](https://developers.openai.com/codex/security)。Codex 使用你原有的登录和代理，不要把它的模型提供方改成被测 SGLang，以免混入额外推理流量。

断开终端用 `Ctrl-b d`；回来用 `tmux attach -t mooncake-cvm`。以下各节交给 Codex 按顺序执行，shell 变量需保持或重新加载。

## 2. 客户端环境与目标确认

在仓库根目录执行，使用 Python 3.11+，单独创建客户端环境；不要在正在 serving 的 conda 环境中 pip install。

```bash
set -euo pipefail
python3 -m venv .venv-mooncake
source .venv-mooncake/bin/activate
python -m pip install -e '.[test]'
input-bench --help
python -m pytest -q --ignore=tests/test_real_data_smoke.py
export HF_HOME=/data/benchmarks/hf-cache
export BASE_URL=http://127.0.0.1:30000  # 改成实际 HTTP 入口；不带 /v1
export CAMPAIGN=/data/benchmarks/dsv4-pro-pp2-cvm-mooncake
mkdir -p "$CAMPAIGN"
# 避免旧实验配置注入额外 headers 或其他默认参数。
for name in ${!INPUT_BENCH_@}; do unset "$name"; done
```

本地服务未配置 API key 时无需设置 `BENCH_API_KEY`。若启用了鉴权，在 shell 中 `read -rsp 'SGLang key: ' BENCH_API_KEY; export BENCH_API_KEY; echo`，不把值写进命令、文档、配置或结果。它与 Codex/OpenAI、百炼的 key 无关。

以下请求明确绕过代理，仅向 `BASE_URL` 发请求。检查实际 served model ID，不能凭下载目录名猜 API 模型名：

```bash
python - <<'PY'
import json, os, urllib.request
base = os.environ['BASE_URL'].rstrip('/')
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
headers = {}
if os.getenv('BENCH_API_KEY'):
    headers['Authorization'] = 'Bearer ' + os.environ['BENCH_API_KEY']
req = urllib.request.Request(base + '/v1/models', headers=headers)
with opener.open(req, timeout=30) as response:
    print(json.dumps(json.load(response), indent=2))
PY
export MODEL='替换为上述接口返回的 DSV4-Pro id'
export TOKENIZER='/data/model/替换为实际DSV4-Pro目录'
export CONTEXT_LENGTH=65536  # 替换为实际启动值，不能超出服务限制
```

Codex 只读核对启动日志/配置或 `/get_server_info`，记录模型版本、tokenizer 来源、SGLang 版本、TP、PP=2、节点/GPU 数、上下文、量化、MTP 是否开启及其参数、显存比例、prefix cache 设置和机密计算模式证据。不要整份输出进程环境或含密钥的启动命令。无法确认的项标记未知，不能自行声称 MTP/CC 已启用。tokenizer 可读取已挂载模型目录；如只在 serving 容器中可见，复制 tokenizer 所需文件到客户端可读目录并记录 hash，无需重下模型权重。

## 3. 只下载 Mooncake 数据

固定一次上游 revision，并保存文件校验值。已有用于比较的原始数据和 workload 时直接复用，不覆盖。这里下载两份本轮需要的 trace，无需运行下载全部数据集的脚本。

```bash
export TRACE_DIR=/data/benchmarks/traces/mooncake
mkdir -p "$TRACE_DIR"
if [ ! -f "$TRACE_DIR/upstream-revision.txt" ]; then
  git ls-remote https://github.com/kvcache-ai/Mooncake.git refs/heads/main \
    | awk '{print $1}' > "$TRACE_DIR/upstream-revision.txt"
fi
MOONCAKE_REV=$(cat "$TRACE_DIR/upstream-revision.txt")
[[ "$MOONCAKE_REV" =~ ^[0-9a-f]{40}$ ]]
for trace in conversation_trace toolagent_trace; do
  if [ ! -f "$TRACE_DIR/$trace.jsonl" ]; then
    curl --fail --location --retry 3 \
      "https://raw.githubusercontent.com/kvcache-ai/Mooncake/$MOONCAKE_REV/FAST25-release/traces/$trace.jsonl" \
      --output "$TRACE_DIR/$trace.jsonl.part"
    mv "$TRACE_DIR/$trace.jsonl.part" "$TRACE_DIR/$trace.jsonl"
  fi
done
sha256sum "$TRACE_DIR/"*.jsonl > "$CAMPAIGN/source-sha256.txt"
```

已有文件若不能证明来自上述 revision，记录“已有文件，来源 revision 未核实”，以文件 hash 为准，不把新解析的 revision 当作旧文件来源。

## 4. 生成配置、冻结并验证输入

复用现有 campaign 的筛选和 seed，只换目标信息。脚本名称包含 GLM52，但 prepare 逻辑是通用 Mooncake 编译器。

```bash
python - <<'PY'
import json, os
from pathlib import Path
cfg = json.loads(Path('scenarios/dsv4_flash_sglang_single_node.json').read_text())
cfg.update(campaign_root=os.environ['CAMPAIGN'], tokenizer=os.environ['TOKENIZER'],
           context_length=int(os.environ['CONTEXT_LENGTH']))
cfg['endpoint'].update(base_url=os.environ['BASE_URL'], model=os.environ['MODEL'])
path = Path(os.environ['CAMPAIGN']) / 'config.json'
if path.exists():
    raise SystemExit('config 已存在：先核对是否复用，不覆盖已有实验')
path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + '\n')
PY
python scripts/glm52_pp2_campaign.py prepare --config "$CAMPAIGN/config.json"
find "$CAMPAIGN/workloads" -name workload.jsonl -exec input-bench validate {} \;
git rev-parse HEAD > "$CAMPAIGN/repository-commit.txt"
python -m pip freeze > "$CAMPAIGN/client-requirements.txt"
```

prepare 会先筛原始长度、选子集，再调用 tokenizer，避免给全量长 trace 无谓分词。会生成 3 份 workload 和 5 点 plan；相同 workload 用于各并发档。检查 plan 合计 120 请求，最大输入加输出不超过实际服务上下文。依赖不支持本机 tokenizer 时，记录报错并仅修复客户端依赖，不升级服务环境。

**跨裸机/CVM 比较时只编译一次**，迁移 `workloads/`、`derived-data/`、`plan.json`、`config.snapshot.json`、source hash 和 tokenizer 信息。plan 内含绝对路径：迁移后只修正 plan 的路径引用，CLI 用 `--tokenizer` 指定本机路径，保持 workload 与 manifest 内容及 SHA256 不变。不能分别重编译后假设相同输入。合成文本用于匹配长度及近似前缀模式，不代表原始语义或真实 MTP 接受率。

## 5. 接口检查与正式运行

下面脚本先发 1 次 8-token 非流式探测，检查 completions 与 ignore_eos，然后逐点以流式模式运行正式回放。每点开始前清空 SGLang prefix cache，无额外 warmup，点内仍允许前缀复用。应在专用、无其他推理请求的服务上执行；检查服务空闲后开始。如服务共享，先安排空闲窗口。缓存清理失败即停止，不把未清理的结果当作同样条件。

```bash
python - <<'PY'
import hashlib, json, os, subprocess, sys, urllib.request
from datetime import datetime, timezone
from pathlib import Path
campaign = Path(os.environ['CAMPAIGN'])
cfg = json.loads((campaign / 'config.json').read_text())
plan = json.loads((campaign / 'plan.json').read_text())
assert len(plan) == 5 and sum(p['requests'] for p in plan) == 120
base = os.environ['BASE_URL'].rstrip('/')
headers = {'Content-Type': 'application/json'}
if os.getenv('BENCH_API_KEY'):
    headers['Authorization'] = 'Bearer ' + os.environ['BENCH_API_KEY']
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
def post(path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers=headers)
    with opener.open(req, timeout=120) as response:
        return response.read()
probe = json.loads(post('/v1/completions', {
    'model': os.environ['MODEL'], 'prompt': 'Hello', 'max_tokens': 8,
    'temperature': 0, 'ignore_eos': True, 'stream': False}))
assert probe.get('choices'), probe
assert probe.get('usage', {}).get('completion_tokens') == 8, probe
root = campaign / 'results' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
root.mkdir(parents=True, exist_ok=False)
(root / 'probe.json').write_text(json.dumps(probe, indent=2))
commands, summaries = [], []
for point in plan:
    workload = Path(point['workload'])
    assert hashlib.sha256(workload.read_bytes()).hexdigest() == point['workload_sha256']
    subprocess.run(['input-bench', 'validate', str(workload)], check=True)
    post('/flush_cache', {})
    output = root / point['label']
    cmd = [sys.executable, '-m', 'input_bench.cli', 'run',
           '--workload', str(workload), '--backend', 'sglang',
           '--base-url', base, '--model', os.environ['MODEL'],
           '--tokenizer', os.environ['TOKENIZER'], '--api-key-env', 'BENCH_API_KEY',
           '--output-dir', str(output), '--max-concurrency', str(point['concurrency']),
           '--pool-size', '32', '--timeout', '1200', '--retries', '0', '--warmup', '0', '--stream',
           '--ttft-slo-ms', str(cfg['slo_ms']['ttft']),
           '--tpot-slo-ms', str(cfg['slo_ms']['tpot']), '--e2e-slo-ms', str(cfg['slo_ms']['e2e'])]
    commands.append(cmd)
    (root / 'commands.json').write_text(json.dumps(commands, indent=2))
    subprocess.run(cmd, check=True)
    summary = json.loads((output / 'summary.json').read_text())
    summaries.append({'label': point['label'], 'summary': summary})
    (root / 'campaign-summary.json').write_text(json.dumps(summaries, indent=2))
    counts = summary['counts']
    assert counts['offered'] == counts['completed'] == point['requests'] and counts['failed'] == 0, counts
print('RESULT_ROOT=' + str(root))
PY
```

命令退出成功还不够：Codex 需核对每点 events、summary、parquet 齐全，5 点完成数相加 120，错误为 0；核对输出 token 是否达到各请求 max_output_tokens，以及服务端 usage 与客户端长度差异。若流式接口不提供 usage，应明确统计来源，不能把本地重分词当作服务端计数。不要混用请求延迟和整轮墙钟耗时。

使用的 `slo_ms` 沿用模板，仅用于统一统计阈值，不是 DSV4-Pro 性能承诺。报告每点墙钟时间、TTFT/TPOT/E2E 的 P50/P95/P99、请求吞吐、输出 token 吞吐、排队延迟及 token 统计来源；以实际 summary 字段为准，缺失则标注。首次小规模样本的 P99 不作为稳定结论。

## 6. 交付与异常处理

在 `$CAMPAIGN/report.md` 写明仓库 commit、输入 hash、配置、环境证据、上述指标和普通实现选择；完整保留结果目录。没有相同 DSV4-Pro 权重、PP/TP/MTP、硬件和相同 workload 的非机密基线时，仅报告 CVM 绝对性能，不能用 Flash/GLM 的既有结果推算 CC 开销。

HTTP 错误、OOM、上下文不匹配、tokenizer 不兼容或超时：保留失败点结果、说明原因，停止继续加压，不自动重启模型或调整服务参数。中断后不要 SIGSTOP/SIGCONT 拼接计时；重跑用新结果目录，把未完成轮次标记为不完整。不要提交密钥、模型、原始 trace 或大体积生成数据到 Git。
