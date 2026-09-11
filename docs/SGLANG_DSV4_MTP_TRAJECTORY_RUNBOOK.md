# 本机 DeepSeek V4 Flash MTP 固定轨迹回放

本机使用 `/opt/miniforge3/envs/sglang-dsv4-flash`（SGLang 0.5.18、Torch 2.13.0+cu130），
权重 `/data/model/DeepSeek-V4-Flash-0731`，8 张 H20、TP=8、Marlin、65536 上下文。
开启本权重自带的投机解码：DSPARK，checkpoint 默认提出 5 个 draft token、验证窗口 6。
0731 的草稿头为 DSpark，不能套用旧版 Flash 的 EAGLE 配置。
参数参考 [SGLang DeepSeek V4 官方说明](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx)，
并核对本机 `deepseek_v4_hook.py` 与 `speculative_hook.py` 对本权重 DSpark 的支持。

## 启动与运行

先确认 GPU 和 30000 端口空闲。以下命令不会关闭已有模型或 tmux session；同名 session 已存在时直接查看。

```bash
tmux new-session -d -s dsv4-flash-mtp \
  'bash scripts/run_dsv4_flash_sglang_mtp_server_tmux.sh'
tmux attach -t dsv4-flash-mtp
```

服务绑定 `127.0.0.1:30000`，不需要 API Key。模型启动日志在
`/data/benchmarks/dsv4-flash-sglang-mtp/logs/`。
原来未启用 MTP 的启动脚本保留。

另一个终端启动本地回放：

```bash
tmux new-session -d -s test-run-local-mtp \
  'bash scripts/run_dsv4_flash_mtp_trajectory_tmux.sh sglang-dsv4-dspark-high-20-r1'
tmux attach -t test-run-local-mtp
```

该脚本等待本地 `/health`，读取 `/server_info` 核验实际模型、TP、MTP 参数后，
保存非敏感部署快照，才开始正式回放。没有额外的 Chat Completions smoke 请求；
当前 SGLang 的 `/health` 可执行内部单 token 探测，它位于正式计时之前。
SGLang 自身的启动 warmup 和 CUDA graph capture 在正式计时之外；不对轨迹做 warmup。
本地脚本清除 URL/workload/results 的环境覆盖与 HTTP 代理，避免意外连接云端。

也可以在服务就绪后直接执行：

```bash
python3 scripts/run_agentic_replay.py dry-run \
  --config scenarios/sglang_dsv4_mtp_swe_trajectory_high.json --model all
python3 scripts/run_agentic_replay.py run \
  --config scenarios/sglang_dsv4_mtp_swe_trajectory_high.json \
  --base-url http://127.0.0.1:30000/v1 --model all \
  --run-id sglang-dsv4-dspark-high-20-r2
```

每次正式重跑使用新 run-id，已有结果不覆盖。
结果位于 `runs/agentic-replay/sglang-dsv4-mtp-high-20/results/<run-id>/deepseek-v4-flash-0731/`；
tmux 编排日志与部署快照位于相邻 `launches/<run-id>/`。

## 与云端的实验契约

- 复用原来的 `aliyun-high-20/bundle`：20 个完整 session、376 次请求，workload SHA-256 为
  `8a8b4d97202f388b7719c232d69ec5181091d1b8bc61a8f89c4df36a7a3e60b1`。
- 录制历史、父子依赖和逻辑输出上限不变。stream=true、high、每请求 max_tokens=4096、
  session/HTTP 并发1、repeat1、warmup0、retries0；错误即停止并保存状态。
- SGLang DeepSeek V4 使用 `chat_template_kwargs: {"thinking": true}` 和顶层
  `reasoning_effort: "high"`。其 `DeepSeekV4Detector.reasoning_default` 是 `explicit_thinking`，
  编码器也读取 `thinking`。百炼使用顶层 `enable_thinking`；通用 runner 按声明的 profile 映射，
  不在错误后删参重试。mock 测试验证两端冻结 messages 一致。
- 本地 profile 显式 `auth=none`，不会读取百炼 Key。只在本机 loopback 提供服务。
- 本地模型为 0731，云端无日期 alias 不证明权重相同；本地 MTP、服务端采样默认值、
  tokenizer、实际输出长度、缓存和网络均可能不同。只能比较端点体验，不能归因为 TEE 开销。
- 保留 SGLang 默认 radix cache，不 flush、不加 nonce；若云端回放与本地回放重叠，
  两者共用客户端主机，需根据 UTC 窗口在报告中说明。

## 离线比较

两端完成后执行。首目录是云端，因此 ratio=本地总耗时/云端总耗时：

```bash
python3 scripts/run_agentic_replay.py compare \
  runs/agentic-replay/aliyun-high-20/results/aliyun-high-20-r1/deepseek-v4-flash \
  runs/agentic-replay/sglang-dsv4-mtp-high-20/results/sglang-dsv4-dspark-high-20-r1/deepseek-v4-flash-0731 \
  --output runs/agentic-replay/sglang-dsv4-mtp-high-20/comparisons/cloud-vs-local-r1
```

比较器允许已验证的两种思考参数映射，保留实际参数差异，仍核验逻辑控制、冻结输入与完整性。
未知或与声明不符的映射不能静默通过；失败/缺失请求不产生有效速度比。

## 启动验证修正记录

首次 EAGLE 试跑（`sglang-dsv4-mtp-high-20-r1`）观察到 accept len≈1、accept rate≈0，
与官方对带 DSpark 头权重的告警一致。已取消并保留部分结果及 EXCLUDED.md，不进入正式比较。
正确的 DSPARK 配置从新 run-id 完整重跑，不能将两次的耗时拼接。
