# GPU TEE Inference Bench

Input Bench 是一个可复现的 LLM serving workload 编译与回放工具，用于在裸金属、CVM、普通虚拟机或不同 serving 栈之间做公平对比。它坚持两阶段执行：`compile` 读取语义数据、用目标模型 tokenizer 重算长度并冻结请求顺序和到达时间；`run` 只读取已冻结的 JSONL，不重新采样或改序。请求端支持 vLLM 和 SGLang 的 OpenAI-compatible API。

数据根目录可配置，程序始终把原始数据视为只读。Input Bench 不会在运行阶段执行 coding agent 的真实工具或做输出质量评测。

## 安装

要求 Python 3.11+：

```bash
cd GPU-TEE-Inference-Bench
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
input-bench --help
```

## 可迁移配置

`--config` 接受 JSON 配置。相对路径以配置文件所在目录为基准；CLI 参数覆盖环境变量，环境变量覆盖 JSON 配置。仓库提供 [vLLM](configs/runtime.vllm.example.json) 和 [SGLang](configs/runtime.sglang.example.json) 两个模板：

```json
{
  "data": {"root": "/data/benchmarks"},
  "tokenizer": {"path": "/models/DeepSeek-V4-Flash-0731"},
  "workload": {"path": "../runs/example/workload.jsonl"},
  "results": {"root": "../runs/results"},
  "endpoint": {
    "backend": "vllm",
    "base_url": "http://127.0.0.1:8000",
    "model": "DeepSeek-V4-Flash-0731",
    "api_key_env": "OPENAI_API_KEY",
    "headers": {},
    "tls_verify": true
  }
}
```

支持以下环境变量，适合容器和只有 guest access 的租用虚拟机：

```text
INPUT_BENCH_CONFIG
INPUT_BENCH_DATA_ROOT
INPUT_BENCH_TOKENIZER
INPUT_BENCH_WORKLOAD
INPUT_BENCH_RESULTS_ROOT
INPUT_BENCH_BACKEND
INPUT_BENCH_BASE_URL
INPUT_BENCH_MODEL
```

使用配置后，CLI 只需提供与本次 workload 有关的参数：

```bash
input-bench compile --config configs/runtime.vllm.example.json \
  --adapter sharegpt --arrival poisson --request-rate 10 --max-samples 1000

input-bench run --config configs/runtime.vllm.example.json
```

本地 tokenizer 在新 manifest 中保存为相对 workload 目录的路径，因此整个 workload bundle 移动到虚拟机后仍可解析。也可以在配置中使用 `tokenizer.id` 和固定 `revision` 指向 Hugging Face tokenizer。

### 仅有虚拟机访问时

原始数据只在 `compile` 阶段需要。推荐在有数据集的准备机生成一次冻结 workload，再把 `workload.jsonl`、`manifest.json` 和所需 tokenizer 一起复制进虚拟机；guest 内只安装本项目并执行 `run`。如果 serving 进程也在 guest 内，vLLM 通常使用 `http://127.0.0.1:8000`，SGLang 通常使用 `http://127.0.0.1:30000`。

```bash
git clone git@github.com:pyz-creeper/GPU-TEE-Inference-Bench.git
cd GPU-TEE-Inference-Bench
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .

export INPUT_BENCH_CONFIG="$PWD/configs/runtime.vllm.example.json"
input-bench validate /path/to/bundle/workload.jsonl
input-bench run --workload /path/to/bundle/workload.jsonl \
  --tokenizer /path/to/bundle/tokenizer
```

同 VM 运行 client 与 server 会共享 CPU 和内存带宽。正式测量应记录这一拓扑，并观察 `scheduler_lag`；条件允许时给 client 设置独立 CPU affinity。

`aiohttp` 用于 HTTP 热路径，`transformers` 加载目标 tokenizer，`pyarrow` 读取源 Parquet 并导出结果。模型 tokenizer 应预先缓存；工具不会自动下载任何数据集。仅做离线流程冒烟测试时可用内置的 `whitespace` tokenizer，它不适合作为正式实验 tokenizer。

## 支持的数据源

| adapter | 数据根目录下的默认路径 | 说明 |
|---|---|---|
| `sharegpt` | `chat/sharegpt-v3/ShareGPT_V3_unfiltered_cleaned_split.json` | 多轮 chat，支持 `independent_turn` / `session_replay` |
| `swe-agent` | `coding/swe-agent-trajectories/data` | recorded coding trajectory，不执行工具 |
| `swebench-verified` | `coding/swe-bench-verified/data/test-00000-of-00001.parquet` | 单轮 task seed，默认不把 gold patch 注入 prompt |
| `thoughtworks` | `coding/thoughtworks-agentic-trajectories/sessions.parquet` | 保留 tool call/response 和扩展字段 |
| `arxiv` | `summarization/arxiv/document` | `validation` / `test` 长文摘要 |
| `longbench` | `summarization/longbench/data.zip` | 直接读取 zip；支持全部 JSONL 子集及独立 `_e` ID 空间 |
| `mooncake` | `traces/mooncake/conversation_trace.jsonl` | shape-only；另可指定 toolagent/synthetic 文件 |
| `servegen` | 无 | 导入 ServeGen 的 JSON/JSONL/CSV 或 Python `Request` iterable |

查看可用性和少量语义记录：

```bash
input-bench datasets
input-bench inspect --adapter sharegpt --limit 2
input-bench inspect --adapter longbench --subset gov_report --limit 2
input-bench inspect --adapter arxiv --split validation --limit 1
```

公开数据可以用 mirror 下载到可配置的数据根目录：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export INPUT_BENCH_DATA_ROOT=/data/benchmarks
scripts/download_datasets.sh
```

`inspect` 默认使用轻量 tokenizer，因为它只展示 adapter 输出。正式 `compile` 必须传目标模型 tokenizer。

Mooncake 官方 FAST'25 trace 将 timestamp 定义为相对到达毫秒。adapter 默认按 milliseconds 转换，并把依据和来源 URL 写入 manifest。其文本不可恢复，因此工具从目标 tokenizer 的有效词表 ID 构造确定性 token blocks：相同的 512-token `hash_id` 产生相同 synthetic block，重新 tokenize 后长度必须精确相等，否则拒绝记录。

## 编译 workload

Poisson 开环：

```bash
input-bench compile \
  --adapter sharegpt \
  --mode independent_turn \
  --arrival poisson \
  --request-rate 10 \
  --max-concurrency 256 \
  --seed 42 \
  --tokenizer Qwen/Qwen3-8B \
  --min-input-tokens 128 \
  --max-input-tokens 8192 \
  --max-output-tokens 512 \
  --max-samples 1000 \
  --output runs/sharegpt/workload.jsonl
```

固定并发闭环：

```bash
input-bench compile \
  --adapter swebench-verified \
  --arrival fixed-concurrency \
  --concurrency 32 \
  --tokenizer Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --include-repo --include-base-commit \
  --output runs/swebench/workload.jsonl
```

Mooncake 原时间回放（`time-scale 10` 表示按十倍速度回放）：

```bash
input-bench compile \
  --adapter mooncake \
  --source /data/benchmarks/traces/mooncake/toolagent_trace.jsonl \
  --arrival timestamp-trace \
  --time-scale 10 \
  --max-concurrency 512 \
  --tokenizer Qwen/Qwen3-8B \
  --output runs/mooncake/workload.jsonl
```

Poisson 的全部间隔在编译时由 `--seed` 一次生成；trace 会以第一个保留请求归一化到零并稳定保留同 timestamp 的源顺序；fixed concurrency 的 `scheduled_offset_s` 为 null。`--duration` 可裁剪 Poisson 时间线，trace 可用 `--window-start/--window-end` 裁剪并用 `--start-offset` 延后。`--bucket-boundary` 可重复传入，分桶标签会写入每条 metadata。

每次编译生成：

- `workload.jsonl`：版本化 `1.0` IR，含确定的 `request_id`、`sequence_no`、endpoint、实际 token 数、采样参数和冻结的到达偏移。
- `manifest.json`：源文件 size/mtime/SHA-256、adapter 参数和模板 hash、tokenizer revision/chat-template hash、seed/arrival 参数、过滤原因、长度分位数以及 workload SHA-256。

同一对比实验必须复制或挂载同一个 `workload.jsonl` 和 `manifest.json`，不要在裸金属与 CVM 两侧分别编译。运行前验证：

```bash
input-bench validate runs/sharegpt/workload.jsonl
```

## 请求回放

```bash
export OPENAI_API_KEY=...
input-bench run \
  --workload runs/sharegpt/workload.jsonl \
  --backend vllm \
  --base-url http://127.0.0.1:8000 \
  --model Qwen/Qwen3-8B \
  --output-dir runs/sharegpt/result \
  --stream \
  --pool-size 256 \
  --timeout 300 \
  --ttft-slo-ms 1000 \
  --e2e-slo-ms 10000 \
  --tpot-slo-ms 50
```

SGLang 使用同一份冻结 workload，只需切换 backend 和 endpoint：

```bash
input-bench run \
  --workload runs/sharegpt/workload.jsonl \
  --backend sglang \
  --base-url http://127.0.0.1:30000 \
  --model Qwen/Qwen3-8B \
  --tokenizer Qwen/Qwen3-8B \
  --output-dir runs/sharegpt/sglang
```

两种 backend 都使用 `/v1/completions` 和 `/v1/chat/completions`。SGLang 的 `/metrics` 需要服务启动时启用 metrics；profile suite 对两种 backend 都使用 `/start_profile` 和 `/stop_profile`。

支持 `/v1/completions` 和 `/v1/chat/completions`、SSE 与 non-streaming（`--no-stream`）、自定义 `--header KEY=VALUE`、`--api-key-env`、重试、连接池大小和 `--no-tls-verify`。一个长生命周期 `aiohttp.ClientSession` 被整个实验复用。开环调度使用 monotonic clock 的绝对 deadline；并发上限下分别记录 scheduled、admitted 和 request start。fixed concurrency 只有请求完成后才释放槽位并补发下一条。

`session_replay` 的子请求还会等待其 `parent_request_id` 完成；父请求即使失败也会释放依赖，避免整场挂死，同时事件中的 queue delay 会反映这段等待。

`--warmup N` 会先立即发送前 N 条请求，但正式测量仍完整、原序回放 workload；warmup 响应不写入正式 events 或指标。Ctrl-C 会取消未完成请求，并尽可能把已经完成的事件写到输出目录。

输出目录包含：

- `events.jsonl`：逐请求原始审计记录，所有 monotonic 时间点为整数纳秒；包含 HTTP/错误分类、重试、截断后的响应、server usage 和客户端 output token recount。
- `summary.json`：实验 UTC/monotonic 边界、配置、RPS、token throughput、错误、goodput 与 latency 分布。
- `results.parquet`：与 events 对应，便于 DuckDB/Pandas 分析。

TTFT 从实际 request start 到首个非空 content；role-only/空 delta 不算首 token。`inter_chunk` 是客户端观察到的 content chunk 间隔，不是严格 per-token ITL。TPOT 是首 content 后剩余时长除以客户端 output token 数。延迟均输出 mean/p50/p90/p95/p99/max；同时报告 scheduler lag 和 client queue delay，以识别客户端自身饱和。已有事件可重新汇总：

```bash
input-bench summarize runs/sharegpt/result/events.jsonl \
  --output runs/sharegpt/result/summary.recomputed.json
```

## 测试与校准

```bash
pytest -q
python -m compileall -q src
input-bench --help
input-bench compile --help
input-bench run --help
```

测试在临时目录生成小型 JSON、zip 和 Parquet fixture；本机挂载 `/data/benchmarks` 时还会对每个真实来源只读一条记录，不扫描完整数据。mock aiohttp server 覆盖流式 chunk 边界、一个 chunk 多事件、role-only 首块、usage、non-streaming、HTTP error、timeout、中途断流和并发上限。

与 `vllm bench serve` 校准时，先编译一份小型 ShareGPT workload，然后保持 model、请求数、temperature、max output tokens、streaming 与 request rate 相同：

```bash
input-bench compile --adapter sharegpt --arrival poisson --request-rate 5 \
  --seed 7 --max-samples 100 --tokenizer "$MODEL" \
  --max-output-tokens 128 --output runs/calibration/workload.jsonl

input-bench run --workload runs/calibration/workload.jsonl \
  --base-url "$BASE_URL" --model "$MODEL" --output-dir runs/calibration/input-bench

vllm bench serve --backend openai --base-url "$BASE_URL" --model "$MODEL" \
  --dataset-name sharegpt \
  --dataset-path /data/benchmarks/chat/sharegpt-v3/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 100 --request-rate 5 --seed 7
```

两种客户端的 sampling/过滤语义并不完全相同，因此校准重点是相近 shape 下的请求计数、吞吐和延迟量级，而不是期待逐请求相同。性能实验最好让客户端与服务端运行在不同机器；若同机运行，应同时记录客户端 CPU 使用率，并重点检查 summary 中的 scheduler lag，避免把客户端瓶颈误判成服务端退化。

## DeepSeek-V4 H800 CVM/裸金属 campaign

`scripts/large_scale_campaign.py` 固化了本次 H800 对照实验。`prepare` 只运行一次：从三类真实数据中确定性抽样，使用模型 tokenizer 和 vLLM 0.25 的 DeepSeek-V4 消息编码器冻结 token 数、请求内容、到达时间与 cold-prefix nonce。固定并发、Poisson 和 Mooncake timestamp trace 共用同一份 `plan.json`，裸金属侧不得重新编译。

完整的服务版本、实际启动命令、模型校验值、停服核验与裸金属复跑步骤见 `campaigns/dsv4-large-scale/BAREMETAL_RUNBOOK.md`。可用 `scripts/start_dsv4_vllm.sh` 在 tmux 中按相同参数启动服务，实验后用 `scripts/stop_dsv4_vllm.sh` 优雅停止。

```bash
cd GPU-TEE-Inference-Bench
python3 scripts/large_scale_campaign.py prepare

# CVM；裸金属时把第一个参数换成 baremetal，并按实际端口修改 URL。
scripts/run_dsv4_campaign.sh cvm performance http://127.0.0.1:18000
scripts/run_dsv4_campaign.sh cvm profile http://127.0.0.1:18000

# benchmark 与服务都在普通 VM 内；第四个参数选择 backend。
scripts/run_dsv4_campaign.sh vm performance http://127.0.0.1:8000 vllm
scripts/run_dsv4_campaign.sh vm performance http://127.0.0.1:30000 sglang
```

长时间运行建议放入 tmux：

```bash
tmux new-session -s dsv4-baremetal -c "$PWD"
scripts/run_dsv4_campaign.sh baremetal performance http://127.0.0.1:8000
```

性能 suite 默认执行独立 preflight、三类 workload 的 `concurrency=1/8/32/64` 两次重复、`0.5/1/2/4 RPS` 的 60 秒 Poisson 开环测试，以及保留 prefix hash 的 Mooncake trace。每个点保存原始 events、Parquet、客户端汇总和 0.5 秒粒度的 backend running/waiting/KV-cache/prefix-cache metrics。`--point 'glob'` 可以只复跑某些点；已有完整点默认跳过，`--force` 才覆盖。

Profile suite 仅运行 12 个代表性请求，并通过所选 backend 的 `/start_profile`、`/stop_profile` 控制 profiler。Profiler 输出位置由 vLLM `--profiler-config` 或 SGLang `SGLANG_TORCH_PROFILER_DIR` 决定；性能 suite 不会开启 profiler。

## 设计边界

- adapter 只恢复语义结构；token 过滤、clamp、sampling、分桶和 arrival 都由 compiler 显式完成并进入 manifest。
- session replay 仅重放已记录的每个 assistant turn，不根据新模型输出继续 agent 推理。
- SWE-bench gold patch/test 信息仅保存在 metadata，发送器绝不把 metadata 加入 HTTP payload。
- ServeGen bridge 只转换其既有输出；执行器仍只有 fixed concurrency、Poisson 和 timestamp trace 三种 policy。
- response 默认最多保留 10,000 字符，可用 `--response-max-chars` 调整。正式实验不应提交 workload、响应正文、Parquet 或缓存，仓库 `.gitignore` 已覆盖这些产物。

## 数据与结果发布边界

Git 仓库只保存代码、配置模板、下载配方、小型测试 fixture 和聚合报告。原始数据由使用者从官方来源下载；包含原始 prompt/message 的冻结 workload 必须在确认上游许可证与隐私要求后，作为独立的版本化 artifact 发布，并附 manifest 与 SHA-256。大型 CUPTI/Nsight trace 和逐请求结果应放在 GitHub Release、Hugging Face Dataset 或对象存储，不进入 Git 历史。
