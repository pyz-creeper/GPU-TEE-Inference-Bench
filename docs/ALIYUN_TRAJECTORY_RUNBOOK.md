# 百炼 SWE-agent 固定轨迹运行说明

已准备 20 条完整轨迹、每模型 376 次请求。三个模型按 DeepSeek → GLM-5.3 → GLM-5.3-Flash
顺序独立回放，共 1,128 次正式请求。当前未执行真实回放。

配置为 `scenarios/aliyun_swe_trajectory_high.json`：thinking 开启，reasoning_effort=high，
max_tokens=4096，session/HTTP 并发1，重复1次，超时300秒，warmup0，retries0。
默认任一请求错误就停止后续请求和模型；已在途请求可能完成。显式 `--continue-on-error`
可在父请求失败后继续冻结轨迹，但结果始终标为不完整。

## 环境与 Key

在已有 `test-api` 环境运行即可，run 不需要 Transformers、原始数据或本地 tokenizer。
运行依赖为 aiohttp 和 pyarrow；若换机器，先执行：

```bash
python3 -m pip install 'aiohttp>=3.9' 'pyarrow>=14'
```

Key 不放在参数或 JSON 配置里。在启动命令的同一个终端执行：

```bash
read -rsp '北京地域百炼 API Key: ' DASHSCOPE_API_KEY
echo
export DASHSCOPE_API_KEY
```

这只设置当前终端及其后续启动的子进程；不会自动进入已经启动的 agent/其他终端。
如需让后续 agent 或其他终端执行，可以显式保存到项目根目录 `.env`：

```bash
umask 077
printf 'DASHSCOPE_API_KEY=%s\n' "$DASHSCOPE_API_KEY" > .env
chmod 600 .env
```

`.env` 已被 Git 忽略；以上命令会替换该文件，若已有其他设置请保留原内容并只更新此项。
无需把 Key 发到聊天。脚本仅在 `run --env-file .env` 时读取指定文件，环境变量优先；
不会执行 dotenv 内容、扫描历史记录或读取其他密钥。不要 `cat .env` 或打印 Key。

默认北京 DashScope 兼容根地址。若使用业务空间专属域名：

```bash
export ALIYUN_BASE_URL='https://你的业务空间ID.cn-beijing.maas.aliyuncs.com/compatible-mode/v1'
```

URL 优先级为 CLI `--base-url` > `ALIYUN_BASE_URL` > JSON。配置内的相对文件路径相对配置文件解析；
`--env-file` 相对当前终端工作目录。

## 运行顺序

1. 离线校验冻结文件和三个模型的 payload；不会读取 Key、探测端点或产生费用：

```bash
python3 scripts/run_agentic_replay.py dry-run \
  --config scenarios/aliyun_swe_trajectory_high.json --model all
```

2. 三个模型完整回放（此命令会产生真实 API 费用）：

```bash
python3 scripts/run_agentic_replay.py run \
  --config scenarios/aliyun_swe_trajectory_high.json \
  --model all --env-file .env --run-id aliyun-high-20-r1
```

如果 Key 已在当前终端 export，省略 `--env-file .env` 即可。
也可以独立执行一个模型，使用新的 run-id，便于分阶段检查或分天运行：

```bash
python3 scripts/run_agentic_replay.py run \
  --config scenarios/aliyun_swe_trajectory_high.json \
  --model glm5.3 --env-file .env --run-id aliyun-glm53-high-r1
```

`--model` 支持 `dsv4-flash`、`glm5.3`、`glm5.3-flash`、`all`，默认仅 DeepSeek。
已有 run-id 目录会报错，不覆盖，也不自动重试付费请求。
取消/失败后先检查记录；重新跑某模型要使用新 run-id 并完整重跑该模型，
不能把半次回放拼接成一个连续总耗时结果。固定 cap 或样本发生变化时，应新编译 bundle 并重新跑所有端点。

## 产物与离线比较

保存到 `runs/agentic-replay/aliyun-high-20/results/<run-id>/<model>/`，模型 ID 内 `/` 替换为 `_`。
每个模型含 events.jsonl、bounds.json、effective-config.json、summary.json、results.parquet、state.json；
整次调用有 run-index.json。events 在后台线程逐请求写入，Ctrl-C 尽量保存剩余状态；
进程崩溃后的 state 可能停留在 running，离线比较会根据 bounds 和缺失请求标记不完整。
Parquet 中 usage、attempt_history 等可变结构字段保存为 JSON 字符串，events 保持原结构。

```bash
python3 scripts/run_agentic_replay.py compare \
  runs/agentic-replay/aliyun-high-20/results/aliyun-high-20-r1/deepseek-v4-flash \
  runs/agentic-replay/aliyun-high-20/results/aliyun-high-20-r1/ZHIPU_GLM-5.3 \
  runs/agentic-replay/aliyun-high-20/results/aliyun-high-20-r1/ZHIPU_GLM-5.3-Flash \
  --output runs/agentic-replay/aliyun-high-20/comparisons/r1
```

compare 仅读取 events/config/bounds，不发新请求；可跨 run-id 选择目录。
生成 comparison.json、comparison.csv 和 REPORT.md。比值为当前模型耗时 / 第一个目录模型耗时，
大于1表示更慢；失败、缺失或配置不兼容时比值为空。

- 主指标 sender_total_s：HTTP session 创建到关闭，包含调度、排队、解析、进度输出和 checkpoint 入队；
  不包括编译、tokenizer 加载、最终文件导出与 journal drain。后台 checkpoint I/O 与测量重叠。
- request_span_s：首个正式请求开始到末请求结束；cumulative_request_e2e_s：逻辑请求 E2E 累加。
- session_duration_s：该 session 首请求开始到末请求结束。
- answer/reasoning 分开记录；TTFT 是首个客户端可见片段，answer TTFT 只统计回答。
  usage 缺失是 null，不伪造 token；不计算貌似精确的 reasoning TPOT。
- `max_tokens` 是请求参数，不强制生成足量。记录 finish_reason=length、实际 usage 和无回答数量。
  GLM 直供的 reasoning 长度语义仍需在线确认；若服务端报告 completion_tokens 超过 cap+10，
  标记 reported_output_exceeds_cap，默认停止以检查配置。无法凭此探测隐藏但未报告的 token。
- 冻结 workload 中 seed/temperature/top_p 在本轮明确省略，并保存在 effective config；
  high 的内部预算和服务端默认采样参数不保证跨模型一致。缓存保留 provider 默认，不 flush、不加 nonce。
- 固定输入历史不使用新答案推进、不执行工具，不是 SWE-bench resolved-rate 测试，也不能推算 CVM/TEE 开销。

公开参数依据：[DeepSeek](https://help.aliyun.com/zh/model-studio/deepseek-api)、
[智谱直供 GLM](https://help.aliyun.com/zh/model-studio/glm-zhipu)。

## 本地验证

```bash
PYTHONPATH=src python3 -m pytest -q tests/test_agentic_runner.py tests/test_replay_bundle.py tests/test_sender.py
```

双 mock 集成覆盖同一 bundle 的独立回放、录制历史不被新答案替换、session/HTTP 并发、父子依赖、
UTF-8 分块、错误/取消留存、未知 usage、参数保护和离线重建比较；不访问真实 API。
