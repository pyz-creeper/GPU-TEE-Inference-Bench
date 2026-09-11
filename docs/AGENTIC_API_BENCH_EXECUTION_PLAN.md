# Input Bench：CVM 与公有云 API 轨迹回放执行计划

状态：待实施。本文件是下一轮 Codex 的实现任务说明，并非已经完成的功能文档。

## 1. 目标和实验范围

让同一批已经录制的 agentic 请求，分别在 CVM 中的 vLLM/SGLang 模型服务和公有云模型 API 上执行，比较客户端观察到的整批请求总耗时，同时保留解释差异所需的逐请求指标。

首选数据源为 `nebius/SWE-agent-trajectories`，采用固定轨迹回放。每个 assistant turn 是一次模型请求；输入由该轮之前的录制历史构成，包含历史动作和环境 observation。上一轮本次新生成的答案只用于记录性能，不拼入下一轮输入。工具结果来自录制数据，bench 不执行 shell、搜索、文件编辑或测试。

第一阶段支持 vLLM、SGLang 和提供 OpenAI-compatible Chat Completions 协议的远端 API。原生 Responses、Anthropic Messages、Gemini 等其他协议作为后续扩展，不宣称本阶段支持所有公有云接口。CVM/API 的实际模型、地址、凭据和 reasoning 参数由部署者配置，不能替用户假定。

本阶段不运行 SWE-bench 任务解决率评估，不部署模型，不引入真实 agent loop。后续若需任务完成时间和 resolved rate，再使用 SWE-bench Verified 配合统一 agent harness 和工具环境。

实验结论定义为“固定 agent 轨迹下的服务端点速度对比”。若两端模型、硬件或推理设置不同，不能将时间差直接归因为 TEE/CVM 开销；隔离该开销需要另做同模型、同硬件条件的裸金属/CVM 对照。

## 2. 开始实施前的仓库核查

先读取适用的 AGENTS.md、README、pyproject.toml、相关代码和 `git status --short` / `git diff`。本工作树已经存在用户改动和未跟踪的实验文件，不能用 reset、checkout 或清理命令覆盖。

优先检查：

- `src/input_bench/adapters/swe_agent.py`、`schema.py`、`compiler.py`、`arrivals/`。
- `src/input_bench/config.py`、`backends.py`、`cli.py`、`sender.py`、`sse.py`。
- `src/input_bench/metrics.py`、`recorder.py` 和相关 tests。
- 前一轮新增的 `scripts/swebench_request_bench.py`、`src/input_bench/benchmark.py`、`scenarios/swebench_request_bench.example.json` 和 `tests/test_request_benchmark.py`。

前一轮新增 bench 使用 SWE-bench issue 单轮输入，可复用其中报告聚合的思路，但它不满足本计划的轨迹回放目标。把通用实现放进包中；旧入口可以保留兼容包装或明确标注为单轮 smoke bench，避免维护两套 sender。不要大规模重构历史 campaign。

当前代码中需要验证和修正的具体问题：

1. SWE-agent session ID 仅使用 `instance_id`，同一 issue 的不同尝试可能冲突。需要唯一、稳定的轨迹身份。
2. compiler 在 turn 层面做随机抽样和长度过滤，可能保留子请求却丢失父请求。完整 session 不能沿用这种采样方式。
3. sender 支持的 backend 只有 vLLM/SGLang，固定拼接 `/v1/...`，sampling 几乎直接透传。
4. sender 强制加载 tokenizer，并以 server usage 覆盖 token 计数；无法保留两种计数来源的区别。
5. reasoning 和 answer 当前直接合并，隐藏 reasoning 与可见 reasoning 的计时口径需要明确。
6. 单个 request semaphore 只限制在途请求，不等价于限制活跃 session 数量。
7. 现有 summary 的 duration 覆盖 sender 生命周期，和“首请求开始到末请求完成”不完全相同；不能混用名称。

上述是实施前的检查项，需用当前代码和数据 fixture 确认，不把本文件当作替代验证的证据。

## 3. 数据与回放契约

### 3.1 数据结构与身份

固定数据集 revision，并记录源文件 SHA-256、相对文件标识和行位置。一个源记录代表一次 trajectory attempt；`instance_id` 标识 issue，不保证标识唯一轨迹。

为 session/request 生成稳定 ID，例如由“数据源版本、相对 shard、行号、内容摘要”组成。不要使用绝对目录路径或 Python `hash()`，移动同一 bundle 不应改变 ID。保留原始 instance_id、生成轨迹的模型、source row、target、exit_status 供审计。

核对真实字段：该数据集的环境 observation 可以使用 `user` role，不能假定它们全是 `tool` role；`trajectory` 也需要验证 list/JSON 字符串两种输入。不能把生成 patch、评估日志或当前轮参考答案泄漏进该轮 prompt。

### 3.2 编译期统一文本表示

实现显式、版本化的 normalization，例如 `recorded-tools-as-text-v1`：

- 输出使用双方都支持的 system/user/assistant 文本 messages。
- 保留录制 system、issue、assistant 动作及 observation 的内容和顺序。
- 原本已经是 user 文本的 observation 保持语义，不重复插入工具结果。
- 对真正的 tool/function 消息转换成带明确来源标签的 user 文本；结构化 tool call 的 name/arguments 转成 assistant 文本，不丢失内容。
- 工具的描述可以作为历史文本保留；请求不携带可执行 tools、functions、tool_calls、tool_call_id 等结构化字段。不得由客户端执行文本中的命令。
- normalization 在 compile 阶段完成，保存版本与参数；不能到不同 target 的 sender 才分别改写内容。

回放示意（示意内容，不作为真实数据引用）：

```text
请求 0 输入 = system + issue
请求 1 输入 = system + issue + 录制动作 A0 + 录制 observation O0
请求 2 输入 = system + issue + A0 + O0 + 录制动作 A1 + 录制 observation O1
```

请求 1 必须等本次请求 0 结束才发送，但无论本次模型生成了什么，请求 1 的 messages 都保持上述冻结内容。失败的父请求释放依赖，允许完成剩余回放，同时将 session/run 标记为不完整。

### 3.3 完整 session 采样与冻结

按 session 采样，不按 assistant turn 采样。支持 `max_sessions` 与 seed，记录选择结果和排除原因。对过长、结构非法或空的 session，第一版采用整条排除策略；不静默截断或重新连接断裂的依赖链。

主实验使用两端都能接受的上下文范围；编译 token 数是明确 reference tokenizer 下的计数，不保证不同模型 tokenizer 下长度完全相同。端点拒绝上下文过长的请求时记录错误，不能仅在该端截断。需要调整样本时生成新 bundle，再让两端都重跑。

建议先用 2 个 session 做真实数据 smoke，再用 20 个完整 session 做初始实验。完整 session 的请求总数由源数据决定，不承诺“20 条 session 只有 100～300 个请求”。prepare/dry-run 应打印轮数、输入长度、输出上限及总请求量，让用户能估计 API 用量。

相同数据 revision、normalization、seed 和编译设置应得到字节相同的 workload。manifest 时间戳可变化，但内容 hash 必须可复现。生成后验证 ID 唯一、父请求先于子请求、sequence 连续和 hash；run 时再次校验。旧 schema 如需升级，要保留旧 workload 的读取兼容，或提供明确迁移说明。

## 4. Target 配置与通用 sender

复用现有 asyncio/aiohttp 发送热路径，加入明确的 `openai-compatible` backend，以及一个小型、可审计的 target 配置/协议适配层。避免引入只为切换 base_url 而存在的第二套请求客户端。

每个 target 至少描述：label、backend、base_url、model、API key 的环境变量名、headers、TLS、streaming、timeout、重试策略，以及允许的请求参数与输出长度参数映射。密钥不写入配置快照、日志、错误信息和导出的命令。

URL 处理必须覆盖域名根路径、已有 `/v1`、尾斜杠和自定义代理前缀。使用明确的 API root/endpoint path 约定和测试，不采用会丢掉代理前缀的简单 urljoin，也不产生 `/v1/v1/...`。

能力差异要求：

- `seed`、`temperature`、`top_p`、`stream_options.include_usage`、输出上限参数及 reasoning 参数由 target 显式声明支持情况。
- 支持将逻辑输出上限映射为 `max_tokens` 或 `max_completion_tokens` 等已声明字段，并记录其语义是否包含 reasoning。
- 通用 API 默认不发送 `ignore_eos`、`cache_salt` 等私有字段。参数不受支持时应在 preflight 明确报错或按显式配置省略，并记录差异；不能在 400 后悄悄删参重试。
- 禁止以额外 payload 覆盖冻结 messages、request identity 或主实验的 logical output cap。必要的 provider 参数要和 effective request 配置一并保存。
- 不调用远端的 `/health`、`/metrics`、`/flush_cache`、`/start_profile` 等私有接口；通用 preflight 默认是离线配置与 payload 检查。
- 返回结构化 tool call 的意外响应应明确标记，不能因为 content 为空而当作正常文本样本。在线探测如需消耗真实 API 请求，要显式启用并单独记录。

CVM 和 API 共用已冻结的 messages、请求顺序和 session 依赖；允许 model 名、鉴权、URL、已声明参数映射不同。比较报告保存所有有效参数及省略项，不能仅凭“用了相同 JSONL”就认定计算负载完全等价。

## 5. Thinking、生成长度与 token 计数

第一组实验优先使用两端都能够明确关闭 thinking 的模型配置。thinking 控制是模型/provider 相关能力，不能推断 `reasoning-parser` 能关闭 thinking，也不能对所有端点强行发同一个参数。

配置记录 requested reasoning mode、发出的控制字段及 verification 状态。未知状态标为 unknown。若任一模型无法关闭 thinking，结果可作为产品级端点对比，并在报告标记 reasoning 不可对齐；不能声称是等计算量或纯 CVM 开销对照。

把回答和可见 reasoning 分开记录；保留兼容旧 response_text 的方式和说明。隐藏 reasoning 文本不可见时不得伪造内容或首 reasoning 时间。

主实验采用固定的逻辑 output cap，初始可用每轮 256 token，允许统一调整。相同 cap 不代表实际生成长度相同，不强制云端支持 ignore_eos。记录 finish_reason、达到上限/提前结束情况及实际输出量；不同 tokenizer 和 reasoning budget 语义差异必须可见。

区分以下计数：编译时 reference input tokens、客户端可选 recount、服务端 usage。保留原始 usage 与来源标签，不用服务器值覆盖掉审计所需的客户端值。reasoning token 可能已包含在 completion token 中，不可重复相加。

允许 run 不加载目标模型的 HF tokenizer：可使用服务端 usage，缺失时计数为 null/unknown，相关 token throughput/TPOT 为不可计算。不能把缺失值记为零或用 whitespace 假装正式 token 数；整组 wall-clock 和 E2E 在没有 token usage 时仍可计算。

## 6. 时间和并发语义

使用 monotonic clock；UTC 仅用于 run 身份与实验时间窗口。以下量必须分开命名、解释和测试：

| 指标 | 定义 |
| --- | --- |
| sender_total_s | sender 测量起点到生命周期结束，包含调度、排队、HTTP session 创建/关闭和当前客户端解析工作 |
| request_span_s | 首个正式 request start 到最后一个 request end |
| cumulative_request_e2e_s | 每个逻辑请求 E2E 的总和；并发时可以大于 sender_total_s |
| session_duration_s | 同一 session 首请求开始到末请求结束；包含该 session 轮间调度等待 |
| request_e2e_ms | 请求开始到完整响应处理结束；若包含重试/退避需显式记录 |
| ttft_observed_ms | 首个客户端可见的非空生成片段，可能是 reasoning，也可能是 answer |
| answer_ttft_ms | 首个非空回答片段；没有回答时为空 |

主比较使用 `sender_total_s`，辅助报告 request_span/session/E2E、TTFT、queue delay、scheduler lag、token 数与失败计数。排除 compile、workload/tokenizer 加载、warmup、结果落盘。保留原有 `measurement.duration_s` 的语义或明确做版本化兼容，不能静默改名改口径。

非流式响应只能观察完整返回时间，不能将它展示成可与 streaming 对照的真实首 token 时间。隐藏 reasoning 的首个可见片段也不是模型实际开始 decode 的时间。现有 TPOT 公式若调整，新增清晰命名/版本和回归测试；缺乏完整 token 与时间观测时不输出貌似精确的 reasoning TPOT。

默认 retries=0，记录 429、timeout、parse/transport error 和 finish_reason。开启重试时把次数与总退避纳入审计，不能把尝试次数和逻辑请求数混淆。取消或崩溃后的残余结果要保留 planned/completed/failed/cancelled 等状态，不能让缺失请求被视为成功。

第一版采用固定活跃 session 数量的闭环调度：一个 worker 从队列取一条完整 session，按父子顺序处理，结束后再取下一条；另设 HTTP 在途上限。若借助现有 sender 实现，要验证它确实限制活跃 session，而不是仅限制同时运行的单轮请求。避免不相关 session 的等待者阻塞或饿死已有 session 的后续 turn。

建议点为 `session_concurrency=1/4/16`，不得将此数误标为固定 RPS。先实现并验证该实验；保留已有 Poisson/trace 路径，不在本阶段扩展为新的研究矩阵。

## 7. 实验编排、产物与比较

提供清晰的 prepare、dry-run、run、compare 入口。可选择 package 子命令或薄脚本，但用户只需一套明确文档，且 prepare 与 run 可分别在有数据的准备机和只含 bundle 的客户端运行。

run 一次选择一个 target；第二个 target 可以隔日独立运行，compare 读取已有目录，不发新请求。target 列表不是默认授权逐个调用所有云 API。

配置模板展示 `cvm` 和 `cloud` 两个 target，占位模型/URL/环境变量名。路径相对配置文件解析，复用或明确扩展现有 CLI > env > JSON 优先级；保存 effective config 并脱敏，避免独立脚本暗中采用另一套优先级。

推荐产物：

```text
runs/agentic-replay/
  bundle/
    workload.jsonl
    manifest.json
    sessions.json
  results/<run-id>/<target>/c<session-concurrency>/repeat-<n>/
    events.jsonl
    summary.json
    results.parquet
    effective-config.json
  comparisons/<comparison-id>/
    comparison.json
    comparison.csv
    REPORT.md
```

每次结果固定记录 workload/manifest hash、target/model、代码版本及 dirty 状态、stream/reasoning 配置、计数来源、session/request 数、并发、重复编号和 UTC 时间。既有结果不静默覆盖；缺少原始数据不阻止冻结 bundle 的 run。

建议每个点先做 3 次重复；样本少时以原始值、median 和范围为主，不把 3 次重复的 p99 当作有力结论。固定负载两端各跑一次即完成最小实验，重复和并发矩阵可配置。

比较输出 ratio = T_CVM / T_API，ratio > 1 表示 CVM 在该端点对照中更慢。核对相同 workload、session 集合、并发、streaming、逻辑生成设置和完整性；不同模型/tokenizer/隐藏 reasoning 等不可对齐因素需标注。失败或缺失请求不能通过更短的总时间赢得比较，应标记不可直接比较，仍保留原始耗时。

客户端在同一位置分别连接两个端点，记录网络拓扑、API region/tier、可见限流和缓存信息。网络、排队和云端多租户波动本来就是端点体验的一部分；不得据此推算未知的纯 GPU 耗时。若可以安排，交替运行两端并记录顺序，减轻时间段差异。

默认不主动 flush cache 或添加单侧 nonce；warmup 默认零，开启时两边采用同一规则并从正式指标排除。共享前缀与重复请求可能命中 cache，报告缓存策略及服务端可见 usage；无法清空云缓存时不能宣称 cold-cache 对照。

## 8. 实施里程碑与验收

### M0：建立基线

记录工作树现状，读取实现，运行能够运行的现有测试。依赖缺失时优先仓库已有环境或隔离环境，准确报告环境故障，不能声称测试通过。验证真实数据结构只读少量记录；缺少数据时使用贴合数据 schema 的小型人工 fixture。

### M1：正确编译轨迹

完成唯一 session ID、文本 normalization、整 session 抽样/过滤、依赖完整性和可复现 manifest。验收覆盖同 issue 多 attempt、list/JSON trajectory、system_prompt、user observation、结构化工具字段转换、金标隔离、无孤儿 parent、空数据/不足 session 和重复编译 hash。

### M2：通用 API 与计时

加入 target 配置、URL/参数能力映射、可选 tokenizer、分离 reasoning/answer 与 usage 来源，以及可审计的错误/时间边界。以本地 mock HTTP/SSE server 验证严格 chat payload、禁止执行工具、reasoning/answer/usage chunk、usage 缺失、提前 EOS、tool-call-only、401/429/timeout/中途断流、重试和鉴权脱敏。

### M3：完整实验闭环

实现 session 调度、prepare/dry-run/run/compare、产物与配置模板。用两个 mock endpoint 完成“同一 bundle 分别运行→离线比较”。验证两端 messages 一致，第二轮仍使用录制历史而不是第一次新输出；父请求完成后 child 才发，活跃 session/HTTP 并发上限均有效。

计时测试用可控 clock/事件屏障和宽容时间区间，避免脆弱的毫秒精度 wall-clock 断言。测试特别覆盖并发下 cumulative E2E 不等于 makespan、失败不能产生有效 speedup、未知 token 不计零、report 可从保存的 bounds/config/events 重建。

### M4：文档和交付

更新 README，提供安装、准备、单 target 运行、第二 target 独立运行与 compare 的完整命令。说明固定历史、工具关闭、reasoning、缓存、长度和网络差异。核心单元/集成测试、CLI smoke 与适用的现有回归必须通过；真实数据与真实服务 smoke 明确做 opt-in。

交付报告列出改动、验证命令与结果、尚缺的真实端点/数据配置和运行方式。没有真实端点不能伪造 CVM/API 性能数字；mock 集成成功可作为实现验收，不代表真实性能实验已完成。

## 9. Codex 执行约束与完成标准

本文件供用户在下一轮明确要求实现时执行。按 M0→M4 完成必要代码、测试、模板、文档，不只复述方案。优先小幅扩展现有模块，保留已有 CLI/workload 兼容性；无法完全兼容时提供迁移和回归证据。

原始数据保持只读，数据集不会随普通单元测试自动下载。无需真实 API 凭据即可完成 mock 验收。真实云请求在用户提供目标和运行范围后显式执行；不要自动扫描本机密钥或启动/停止 GPU 服务。本任务不要求 Git 提交、推送或新增子 agent。

遇到未指定的实现细节，选择最小可验证方案并在交付说明中记录。只有影响实验定义且无法从配置或本计划解决的问题才需要提问。

完成时确认：

- [ ] 完整轨迹唯一标识、确定性选择、文本化与依赖链测试通过。
- [ ] CVM/云 API 可复用同一 bundle，run 无需原始数据。
- [ ] tool 文本从不被执行，云端请求不启用内置工具。
- [ ] capability、token 来源、thinking 和生成长度差异可审计。
- [ ] 活跃 session 与 HTTP 并发语义清楚且经过测试。
- [ ] 总耗时、累计请求耗时、session 耗时和失败统计定义明确。
- [ ] 两个 mock target 的独立回放与离线比较通过。
- [ ] 现有功能回归、使用文档和真实运行配置模板已交付。

## 10. 启动 Codex CLI

本机核查版本：`codex-cli 0.144.6`。GPT-6 Astra 的模型 ID 为 `gpt-6-astra`；high 是独立的 reasoning effort 设置，不是模型 ID 的后缀。实际使用权限取决于当前登录账户/provider；本文件生成时未启动付费模型调用。

```bash
codex -C /root/GPU-TEE-Inference-Bench \
  -m gpt-6-astra \
  -c 'model_reasoning_effort="high"' \
  --sandbox workspace-write \
  --ask-for-approval on-request \
  --search \
  '请完整阅读 docs/AGENTIC_API_BENCH_EXECUTION_PLAN.md，并按 M0 到 M4 实施 input-bench 改进。先核对现有代码与未提交改动，保留用户已有工作；完成轨迹编译、通用 API sender、实验编排、离线比较、测试和文档。以双 mock endpoint 的完整回放作为验收。遇到普通实现选择自行判断并记录，不要只输出计划。真实 API 压测待我配置目标后执行。最终用中文报告改动、验证结果和运行命令。'
```

启动后可通过 `/status` 查看当前模型与会话配置；CLI/provider 如拒绝指定模型，应明确报告，不自动替换成其他模型。

## 11. 参考资料

- [SWE-agent trajectories 数据集与 schema](https://huggingface.co/datasets/nebius/SWE-agent-trajectories)
- [SWE-agent 轨迹说明](https://swe-agent.com/latest/usage/trajectories/)
- [GPT-6 Astra 模型与 reasoning effort](https://developers.openai.com/api/docs/models/gpt-6-astra)
- [Codex 配置参考：model_reasoning_effort](https://learn.chatgpt.com/docs/config-file/config-reference)

历史数据中的 thought 文本不等于新端点返回的 reasoning_content；分析时分别标明“输入中的录制历史”和“本次生成的 reasoning”。
