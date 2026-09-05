# Input Bench 实现指导

## 1. 目标

在本目录实现一个可复现的 LLM serving workload 编译与请求回放工具，用于比较裸金属和机密虚拟机中的推理性能。

核心原则是“两阶段执行”：

1. **compile**：从原始数据集提取语义数据，使用目标模型 tokenizer 计算长度，并生成完全确定的 workload JSONL。
2. **run**：只读取已经生成的 workload，按照指定 arrival policy 向 OpenAI-compatible endpoint 发送请求并记录原始事件。

裸金属和 CVM 必须复用同一份 workload 文件。运行阶段不得重新采样数据、重新生成到达时间或改变请求顺序。

## 2. 数据位置和边界

原始数据均位于 `/data/benchmarks`。原始数据只读，程序不得修改其中任何文件。

当前需要支持的全部来源：

| Adapter / bridge | 路径 | 类型 |
|---|---|---|
| ShareGPT V3 | `/data/benchmarks/chat/sharegpt-v3/ShareGPT_V3_unfiltered_cleaned_split.json` | chat，多轮 |
| SWE-agent trajectories | `/data/benchmarks/coding/swe-agent-trajectories/data/*.parquet` | coding agent，多轮轨迹 |
| SWE-bench Verified | `/data/benchmarks/coding/swe-bench-verified/data/test-00000-of-00001.parquet` | coding task，单轮种子 |
| Thoughtworks agentic trajectories | `/data/benchmarks/coding/thoughtworks-agentic-trajectories/sessions.parquet` | coding agent，多轮轨迹 |
| arXiv summarization | `/data/benchmarks/summarization/arxiv/document/*.parquet` | 长文本摘要 |
| LongBench | `/data/benchmarks/summarization/longbench/data.zip` | 多任务长上下文；重点是摘要子集 |
| Mooncake conversation trace | `/data/benchmarks/traces/mooncake/conversation_trace.jsonl` | shape + timestamp trace |
| Mooncake tool-agent trace | `/data/benchmarks/traces/mooncake/toolagent_trace.jsonl` | shape + timestamp trace |
| Mooncake synthetic trace | `/data/benchmarks/traces/mooncake/synthetic_trace.jsonl` | shape + timestamp trace |
| ServeGen | `/data/benchmarks/generators/ServeGen` | workload generator bridge，不是语义数据集 |

不得在实现或测试中自动下载其他数据集。缺少可选依赖时给出清晰错误。

## 3. MVP 范围

必须完成：

- 上表所有数据源的 adapter 或 bridge。
- 统一、版本化的 workload IR。
- tokenizer-aware 提取、过滤和分桶。
- 三种 arrival policy：固定并发闭环、Poisson 开环、timestamp trace replay。
- OpenAI-compatible completions 和 chat-completions endpoint。
- SSE streaming 和 non-streaming。
- 原始事件记录、汇总指标和 Parquet 导出。
- 单元测试、集成测试、CLI 帮助和使用文档。
- 用 mock HTTP/SSE server 完成测试，不要求真实 GPU 服务在线。

MVP 不做：

- 真实容器中的 coding tool 执行。
- 根据模型新输出继续自主 agent 推理。
- ServeGen 的 Gamma/Weibull 实时生成；只提供可导入/导出的 bridge，执行器仍只支持本节规定的三种 policy。
- GPU 或 vLLM 服务端 profiling。
- 输出质量评测。

## 4. 推荐工程结构

```text
input-bench/
├── pyproject.toml
├── README.md
├── IMPLEMENTATION_GUIDE.md
├── src/input_bench/
│   ├── cli.py
│   ├── schema.py
│   ├── tokenizer.py
│   ├── adapters/
│   │   ├── base.py
│   │   ├── sharegpt.py
│   │   ├── swe_agent.py
│   │   ├── swebench_verified.py
│   │   ├── thoughtworks_agent.py
│   │   ├── arxiv.py
│   │   ├── longbench.py
│   │   ├── mooncake.py
│   │   └── servegen.py
│   ├── arrivals/
│   │   ├── base.py
│   │   ├── concurrency.py
│   │   ├── poisson.py
│   │   └── trace.py
│   ├── compiler.py
│   ├── sender.py
│   ├── sse.py
│   ├── recorder.py
│   └── metrics.py
└── tests/
    ├── fixtures/
    ├── test_adapters.py
    ├── test_arrivals.py
    ├── test_compiler.py
    ├── test_sender.py
    └── test_metrics.py
```

建议依赖：Python 3.11+、`aiohttp`、`transformers`、`datasets`、`pyarrow`、`numpy`、`orjson`、`typer`、`pydantic`、`pytest`、`pytest-asyncio`。不要在 HTTP 热路径中使用 OpenAI SDK。

## 5. 统一数据模型

### 5.1 Adapter 输出：SemanticSample

Adapter 只负责语义结构，不负责到达时间：

```python
class SemanticSample:
    sample_id: str
    source: str
    workload_class: str       # chat | agent_coding | summarization | long_context
    mode: str                 # single | independent_turn | session_replay | shape_only
    session_id: str | None
    turn_id: int | None
    parent_sample_id: str | None
    prompt: str | None
    messages: list[Message] | None
    reference_output: str | None
    requested_output_tokens: int | None
    think_time_s: float | None
    metadata: dict
```

约束：

- `prompt` 和 `messages` 二选一。
- ID 必须稳定，不得依赖 Python `hash()`。
- 保留 source record ID、split、trajectory/session ID 和 turn index。
- adapter 输出顺序必须确定。
- adapter 不得偷偷截断；过滤和截断必须由显式 compiler 参数控制并写入 manifest。

### 5.2 编译输出：WorkloadRequest

`workload.jsonl` 每行一个请求：

```json
{
  "schema_version": "1.0",
  "request_id": "sharegpt:QWJhYvA_0:turn:2",
  "sequence_no": 12,
  "source": "sharegpt",
  "workload_class": "chat",
  "session_id": "sharegpt:QWJhYvA_0",
  "parent_request_id": null,
  "scheduled_offset_s": 1.274,
  "endpoint_kind": "chat",
  "messages": [{"role": "user", "content": "..."}],
  "prompt": null,
  "input_tokens": 1536,
  "max_output_tokens": 256,
  "sampling": {"temperature": 0.0, "top_p": 1.0, "seed": 1},
  "metadata": {}
}
```

在 workload 同目录写 `manifest.json`，至少记录：

- schema version、创建时间和工具版本。
- dataset 路径及其文件大小、mtime、SHA-256。
- adapter 名称和全部参数。
- tokenizer ID、revision、chat template hash。
- seed、arrival policy 及全部参数。
- 过滤前后数量、丢弃原因计数。
- input/output token 的 min、mean、p50、p90、p95、p99、max。
- workload JSONL 的 SHA-256。

必须使用目标模型 tokenizer 重新计算 token 数。Thoughtworks 中的 `cl100k_base` token 统计以及数据卡中的空格 token 统计只能作为 metadata，不能作为实验长度。

## 6. Dataset Adapter 要求

### 6.1 ShareGPTAdapter

输入字段为 `id` 和 `conversations[{from,value}]`。

- 将 `human`、`gpt`、`system` 规范化为 `user`、`assistant`、`system`。
- 丢弃空内容、非法角色顺序并统计原因；不得静默修复破坏性数据。
- 支持 `independent_turn`：每个 assistant turn 形成一个样本，请求内容截止到该 assistant turn 之前，reference output 是该 assistant 内容。
- 支持 `session_replay`：保留同一会话中所有 assistant turn 的依赖关系。
- 支持 min/max input tokens、min/max output tokens、最大轮数和 seed sampling。

### 6.2 SWEAgentTrajectoriesAdapter

字段包括 `instance_id`、`model_name`、`target`、`trajectory`、`exit_status`、`generated_patch`、`eval_logs`。`trajectory` 在 Parquet 中表现为结构化 list，仍应兼容字符串 JSON 形式。

- 规范化 `system`、`ai`、`user` 为标准消息角色。
- `ai` step 是待推理 assistant turn；后续 observation 作为已记录 tool/environment result 注入下一请求上下文。
- 支持完整 session replay 和独立 turn 两种模式。
- 保留 `mask`、`cutoff_date`、`system_prompt`、target、model_name 和 exit_status。
- MVP 只做 recorded trajectory replay，不执行真实工具，不使用 generated patch 作为模型输入。

### 6.3 SWEBenchVerifiedAdapter

这是 task-seed adapter，而不是完整 agent trajectory。

- 默认用 `problem_statement` 构造单请求 coding prompt。
- 提供可配置模板，默认模板必须写入代码并记录其 hash。
- 可选加入 `repo`、`base_commit`、`hints_text`；默认不加入 gold `patch`、`test_patch`、`FAIL_TO_PASS` 或 `PASS_TO_PASS`，避免答案泄漏。
- reference patch 只保存在 metadata/评测侧，不发送给服务端。
- `instance_id` 用作稳定 sample ID。

### 6.4 ThoughtworksAgenticAdapter

字段包括 `session_id`、`source_dataset`、`source_id`、`agent_framework`、`recorded_model`、`messages_json`、`n_turns`、`max_isl`、`total_tokens` 和 `ground_truth_meta_json`。

- 解析并规范化 `messages_json`。
- 支持 independent turn 和 session replay。
- 保留 tool call/tool response 的原有结构；未知扩展字段放入 metadata，不能丢失。
- 使用目标 tokenizer 重算每一轮累计上下文长度。
- 对损坏 JSON 给出可定位的错误或显式 skip reason。

### 6.5 ArxivSummarizationAdapter

字段为 `id`、`article`、`abstract`。

- validation/test 两个文件都支持，split 可选。
- 默认 prompt 模板为确定性的 summarization instruction + article。
- abstract 是 reference output；默认 `max_output_tokens` 可取 reference 的目标 tokenizer 长度，并允许 clamp。
- 支持按输入 token 长度分桶和采样。

### 6.6 LongBenchAdapter

- 直接支持读取 `data.zip`，不得要求用户手工解压。
- 能枚举并加载 zip 中全部 JSONL 子集，通过 `--subset` 选择。
- 对通用 LongBench 字段 `input`、`context`、`answers`、`length`、`dataset`、`language`、`all_classes` 做兼容解析。
- 根据子集提供确定性模板，并将模板名称/hash 写入 manifest。
- 首批验收重点为 `gov_report`、`qmsum`、`multi_news` 三个 summarization 子集。
- 其他 LongBench 子集也必须能编译，分类为 `long_context` 或明确的 task metadata，不要求 MVP 做质量评分。
- `_e` 扩展版子集与普通版分开识别，不能混在同一 source ID 下。

### 6.7 MooncakeTraceAdapter

三个文件均包含 `timestamp`、`input_length`、`output_length`、`hash_ids`。

- 这是 shape-only adapter，不存在可恢复的真实文本。
- 保留原始 timestamp、长度和 hash IDs。
- 根据目标 tokenizer 的有效 token ID 范围生成确定性 synthetic prompt，生成方法和 seed 写入 manifest。
- 生成文本后必须重新 tokenize 并验证长度；允许的误差必须为 0，或明确记录并拒绝该条请求。
- `output_length` 映射到 `max_output_tokens`。
- timestamp trace policy 必须能原样使用该 timestamp，也允许显式 `time_scale` 加速/减速。
- 不应把 Mooncake 长度随机匹配到另一个语义数据集；若未来实现 shape matching，必须作为单独、显式模式。

### 6.8 ServeGenBridge

ServeGen 不是语义数据集。bridge 负责：

- 读取或调用 `/data/benchmarks/generators/ServeGen` 的输出。
- 将 ServeGen `Request(timestamp, data)` 转换为统一的 shape-only `SemanticSample`/`WorkloadRequest`。
- 保留 client ID（若上游提供）、input/output token shape 和原始 timestamp。
- 仅提供转换能力；MVP CLI 不新增第四种 arrival policy。
- 不修改 vendored ServeGen 源码，必要适配写在本项目内。

## 7. 三种 Arrival Policy

### 7.1 FixedConcurrencyPolicy（闭环）

- 参数：`concurrency`。
- 最多维持 N 个 outstanding requests；一个请求完成后才补发下一个。
- `scheduled_offset_s` 可为 `null`，运行时仍记录 sequence。
- 用于最大吞吐、稳定 profiling 和 concurrency sweep。
- 必须记录等待 semaphore 的 client queue time。

### 7.2 PoissonPolicy（开环）

- 参数：`request_rate`、`seed`，可选 `duration`。
- inter-arrival time 使用指数分布，均值为 `1 / request_rate`。
- 编译阶段一次性生成所有 `scheduled_offset_s`，运行阶段禁止重新采样。
- 运行调度使用 monotonic clock 和绝对 deadline，不能通过逐次 `sleep(iat)` 累积漂移。
- 并发上限如果启用，必须将 scheduled、admitted 和 request_start 分开记录。

### 7.3 TimestampTracePolicy（开环回放）

- 参数：`time_scale`、`start_offset`，可选裁剪窗口。
- 将 trace 第一个保留请求归一化到 0。
- 保留原始相对间隔；同 timestamp 的稳定顺序由原始行号决定。
- Mooncake 的时间单位必须从数据说明或统计中验证，并在 manifest 中写明；不得仅凭字段名猜测。
- 对无序、负值、NaN timestamp 必须验证并报错。

## 8. Endpoint 和发送器

支持：

- `POST /v1/completions`
- `POST /v1/chat/completions`
- streaming 和 non-streaming
- 自定义 base URL、model、API key 环境变量、headers、timeout
- 可选 TLS verify 配置

使用一个长生命周期 `aiohttp.ClientSession`，连接池大小可配置。SSE parser 必须正确处理：

- 一个事件跨多个 TCP chunk。
- 一个 TCP chunk 包含多个 SSE event。
- `data: [DONE]`。
- chat stream 首个只有 role、没有 content 的 chunk。
- usage chunk、空 delta、错误 JSON、非 2xx 和中途断流。

TTFT 定义为从实际开始请求到首个**非空有效 content** 的时间，不把 role-only chunk 当作首 token。客户端记录的是 inter-chunk latency，不得将它命名为严格的 per-token ITL。

## 9. 原始事件和指标

每个请求记录下列 monotonic 时间点，原始存储使用整数纳秒：

- `scheduled_ns`
- `admitted_ns`
- `request_start_ns`
- `headers_ns`
- `first_content_ns`
- `chunk_times_ns[]`
- `request_end_ns`

同时记录 wall-clock UTC 实验起止时间、HTTP 状态、错误类型、重试次数、响应文本或其可配置截断版本、服务端 usage、客户端重新计算的 output tokens。

至少输出：

- `events.jsonl`：逐请求原始结果。
- `summary.json`：聚合结果和实验配置。
- `results.parquet`：便于 DuckDB/Pandas 分析。

至少汇总：

- offered、started、completed、failed RPS。
- input、output、total token throughput。
- scheduler lag、client queue delay、TTFT、E2E、TPOT/inter-chunk 的 mean、p50、p90、p95、p99、max。
- goodput：支持通过 CLI 指定 TTFT/E2E/TPOT SLO。
- HTTP 错误、timeout、parse error 分类计数。
- event-loop lag 或调度落后分布，以识别客户端自身成为瓶颈。

warmup 请求不得计入正式 workload 指标。实验持续时间必须明确给出测量边界。

## 10. CLI 设计

至少提供：

```bash
input-bench datasets
input-bench inspect --adapter sharegpt --source /data/benchmarks/chat/sharegpt-v3/ShareGPT_V3_unfiltered_cleaned_split.json
input-bench compile --adapter sharegpt --arrival poisson --request-rate 10 --tokenizer MODEL --output runs/workload.jsonl
input-bench validate runs/workload.jsonl
input-bench run --workload runs/workload.jsonl --base-url http://127.0.0.1:8000 --model MODEL --output-dir runs/example
input-bench summarize runs/example/events.jsonl
```

所有随机行为都必须接受 `--seed`。CLI 实际选项名可以细化，但 README 示例和 `--help` 必须一致。

## 11. 测试要求

### Adapter

- 每个 adapter 至少一个最小 fixture 和对应单元测试。
- 至少对真实数据各读取少量记录做 smoke test；测试不能扫描完整 5.6 GB 数据。
- 测试稳定 ID、role 转换、turn 构造、reference 隔离、token recount、过滤原因和确定性。
- LongBench 测试直接从小型临时 zip 读取。
- Parquet fixture 在测试中生成，不提交大型二进制文件。

### Arrival

- 固定 seed 的 workload 文件逐字节可复现。
- Poisson 检查数量、单调 timestamp 和基本统计性质；不要写概率性 flaky test。
- trace 检查归一化、time scale、同 timestamp 稳定顺序。
- fixed concurrency 通过 mock server 验证实际 outstanding 不超过 N。

### Sender

- 本地 mock aiohttp server 覆盖 streaming、non-streaming、role-only first chunk、chunk 分裂、错误状态、timeout 和断流。
- 验证 scheduled/admitted/start 分离。
- 验证失败请求不会导致其他任务泄漏或整场 benchmark 挂死。

### 校准

- 提供一个小型 ShareGPT workload 的命令，与 `vllm bench serve` 做同参数交叉检查。
- README 明确客户端和服务端最好运行在不同机器；若同机运行，记录客户端 CPU 使用情况并报告 event-loop lag。

## 12. 工程质量和完成标准

- 使用 `pyproject.toml`，支持 editable install。
- 类型标注覆盖核心 schema、adapter、arrival 和 sender。
- 不修改 `/data/benchmarks`。
- 不提交生成的 workload、响应正文、大型数据或 cache。
- 提供 `.gitignore`。
- 对中断使用 graceful cancellation，并尽可能写完已完成请求的 events。
- 错误信息包含 adapter、source file 和 record ID/line number。
- 测试、lint（如果配置）和 CLI smoke test 全部通过。
- README 给出从环境安装、inspect、compile、run 到 summarize 的最小完整流程。

## 13. 推荐实施顺序

1. 建立 package、schema、adapter protocol 和 CLI 骨架。
2. 实现 ShareGPT、arXiv、LongBench adapter。
3. 实现三套 coding adapter。
4. 实现 Mooncake adapter 和 ServeGen bridge。
5. 实现 workload compiler、manifest、tokenization 和三种 arrival policy。
6. 实现 aiohttp sender、SSE parser 和原始事件记录。
7. 实现 metrics/Parquet 输出。
8. 完成 mock server 集成测试、真实数据 smoke test 和 README。

实现过程中可以调整内部模块拆分，但不得改变“两阶段、同一 workload 公平回放、原始时间点可审计”这三个核心约束。
