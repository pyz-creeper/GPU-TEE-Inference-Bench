# GLM-5.3-Flash / 8×H20 初步分析

> 2026-09-06 更新：与 GLM5.2、DSV4 相同的五点 Mooncake campaign 已完成，见 [`mooncake-mtp/REPORT.md`](mooncake-mtp/REPORT.md)。本文件以下内容主要分析此前的七点定长合成矩阵；其中“尚未完成同 workload 横评”的表述已由新 campaign 补齐，但关于旧矩阵 warm-prefix 污染的审计仍然有效。

## 一句话结论

这组数据足以确认 SGLang 在 node1 上能够正确部署 GLM-5.3-Flash，并给出一组可复查的**缓存开启、闭环固定并发、合成定长输入**性能结果；但它还不是冷缓存或真实 conversation/tool-agent workload 的最终基线，也没有形成有效的 SGLang/vLLM 横向对比。

## 结果有效性

- 正式 SGLang 轮次：`results/sglang/20260906T033021Z/`
- 7 个测试点共 448 个请求，448 成功、0 失败。
- 每个点至少预热一个完整并发批次。
- 使用同一个冻结 workload、固定 seed、精确 token 长度、stream、`temperature=0`、`ignore_eos=true`。
- 独立烟测能对简单问题输出连贯答案，说明 SGLang 部署的基本语义正确性通过。
- 正式性能 workload 是 tokenizer vocabulary block 生成的合成 completion，不是自然语言对话，不能据此评估回答质量。

## 聚合吞吐和单请求速度不是一个口径

下表的“单请求生成速度”由 `1000 / TPOT` 反算。慢尾速度使用 TPOT p95，因此表示较慢一侧的请求，不是聚合吞吐。

| 场景 | 并发 | 聚合输出 tok/s | 单请求 p50 生成速度 | 单请求慢尾生成速度 | TTFT p95 | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|
| 128→512 decode | 1 | 199.0 | 221.2 tok/s | 151.7 tok/s | 288 ms | 3.66 s |
| 128→512 decode | 16 | 949.9 | 70.9 tok/s | 49.7 tok/s | 567 ms | 10.63 s |
| 1024→256 balanced | 1 | 184.8 | 254.5 tok/s | 151.5 tok/s | 286 ms | 1.97 s |
| 1024→256 balanced | 16 | 867.2 | 72.0 tok/s | 46.5 tok/s | 724 ms | 6.17 s |
| 1024→256 balanced | 32 | 1295.3 | 51.3 tok/s | 36.7 tok/s | 1,781 ms | 7.81 s |
| 8192→128 prefill | 1 | 94.8 | 284.9 tok/s | 98.6 tok/s | 937 ms | 1.79 s |
| 8192→128 prefill | 16 | 166.3 | 15.1 tok/s | 7.7 tok/s | 8,144 ms | 22.27 s |

因此，如果目标是“每个用户至少约 50 tok/s”：

- C1 明显满足。
- decode C16 的慢尾约 49.7 tok/s，已在目标边界。
- balanced C16 的慢尾约 46.5 tok/s；C32 虽然 p50 约 51.3 tok/s，但慢尾只有约 36.7 tok/s。
- 8192-token prefill 与 decode 混跑的 C16 明显不满足。

## 并发扩展效率

- Decode C1→C16：GPU 数不变，聚合输出吞吐从 199.0 增至 949.9 tok/s，为 4.77×；并发是 16×，所以单请求延迟明显变差。
- Balanced C1→C16：184.8→867.2 tok/s，为 4.69×。
- Balanced C16→C32：867.2→1295.3 tok/s，只再增加 49.4%；同时 TTFT p95 从 724 ms 增至 1,781 ms。
- Prefill C1→C16：输入吞吐只从 6069.6 增至 10640.9 tok/s，为 1.75×；E2E p50 却从 1.34 s 增至 12.44 s，为 9.29×。

对低延迟服务而言，balanced 的合理工作区间更可能在 C8–C16，而不是直接使用 C32。现有矩阵缺少 C2/C4/C8/C24，尚不能精确找到吞吐—延迟拐点。

## 前缀缓存污染

正式轮次之前，同一服务进程已经跑过相同 seed 和相同 workload；SGLang Radix Cache 未禁用或清空。将 `summary.json` 的正式测量 UTC 窗口转换为 node1 本地时间，并汇总 `logs/sglang-server.log` 中该窗口内的 `#new-token` 与 `#cached-token`，得到：

| 场景 | 新计算 prompt token | 命中缓存 token | 缓存占逻辑输入比例 |
|---|---:|---:|---:|
| balanced C16 | 43,264 | 38,656 | 47.2% |
| balanced C32 | 92,608 | 71,232 | 43.5% |
| prefill C16 | 539,328 | 116,032 | 17.7% |

这三个点的日志 token 总数与矩阵逻辑输入 token 总数完全一致，所以缓存比例不是抽样估算。影响如下：

- `performance-summary` 中 input tok/s 的分子是完整逻辑输入 token，包括已缓存部分；它不是 GPU 实际重新完成的 prefill token/s。
- balanced C16/C32 的 TTFT 和整体吞吐明显受重复前缀缓存帮助。
- output tok/s 的计数本身仍正确，但因为 prefill 占用减少，decode 获得了更多调度/计算资源，所以也不能直接当作冷缓存结果。
- 如果生产业务确实具有大量相同 system prompt 或共享历史前缀，那么这组 warm-prefix 结果有实际意义；否则需要单列一组冷缓存结果。

## 合成文本和 MTP 偏置

workload prompt 是类似 `teilungteilung...`、` Gesundheit Gesundheit...` 的单 token 重复块，部分生成结果也呈现高度重复文本。它能保证 token 长度精确，却会改变模型输出分布和 EAGLE speculative decoding 的接受率。

正式窗口内 server 周期日志的 `accept len` 平均值约为：decode C1 4.87、decode C16 3.41、balanced C1 5.11、balanced C16 3.36、balanced C32 2.04、prefill C1 4.86、prefill C16 3.25。它说明 MTP 确实贡献了多个接受 token，但接受率高度依赖这种合成内容。

因此 199 tok/s 的 C1 结果更准确的名字是“该定长合成 completion + adaptive EAGLE MTP 下的窗口吞吐”，不能直接推广到自然 conversation 或 tool-agent 输出。

## 当前最明确的瓶颈

### 长 prompt C16：prefill 调度预算是直接瓶颈

服务配置同时设置：

- `chunked_prefill_size=8192`
- `max_prefill_tokens=8192`
- 每个请求输入恰好 8192 token

也就是说，一个完整长请求就会吃完一次 scheduler prefill token budget。日志中可以看到大量单请求 8192-token prefill batch、pending token 和队列逐步排空，decode 只能与这些长 prefill 交错执行。这解释了 prefill C16 下 TTFT p95=8.14 s、TPOT p95=129.75 ms 和 E2E p95=22.27 s。

### Balanced C32：进入吞吐收益递减区

C16 加倍到 C32 后吞吐只增 49.4%，TTFT p95 增加 146%。这表示 GPU kernel、TP/EP 通信以及 scheduler 混合批处理已经成为主要竞争资源。它不是 KV 容量不足：server 报告 1,292,096-token KV 池，相关日志中的 full-token usage 仍远未打满。

### 冷启动：主要不是权重读取

权重加载约 28 s，但 scheduler 启动约 260 s、API 完全可用约 270 s；其中 target-verify CUDA graph 捕获约 133.8 s，draft decode/extend graph 各约 9 s。部署冷启动主要花在内核 JIT、DeepGEMM/MTP 与图捕获，而不是模型权重读取。

现有日志没有 Nsight、kernel 时间线、NCCL 分项或 GPU 功耗/利用率采样，所以还不能把 balanced/decode 的饱和进一步定量拆成 GEMM、attention、MoE dispatch、all-reduce/all-to-all 和 CPU scheduler 百分比。

## vLLM 结果应该怎样解读

这轮不存在可用于性能比较的 vLLM 数字：

- MTP=5 和 MTP=1 在 sampling warm-up 触发 CUDA device-side assertion，服务不能就绪。
- 禁用 MTP 后 HTTP 200，但 completion/chat 都从首 token 开始重复输出 `lock`。
- Hopper override、纯 DeepGEMM、eager、禁用 custom all-reduce 等诊断组合没有恢复正确输出。

因此这是“实现/版本正确性阻塞”，不能记成 vLLM 吞吐为 0，也不能据此得出 SGLang 比 vLLM 快多少。只有新的 vLLM 构建先通过同等语义烟测，才应运行冻结矩阵。

## 建议的下一轮矩阵

1. 保留当前结果，明确标记为 `warm-prefix / synthetic / MTP-on`。
2. 重启服务后立即跑一轮，或禁用 Radix Cache，得到 `cold-prefix` 基线；不要在正式轮次前跑同一批 prompt。
3. 使用真实 conversation、tool-agent、timestamp-trace workload，再做 MTP on/off 对照。
4. 补 C2/C4/C8/C24，优先定位满足“慢尾 ≥50 tok/s”的最大并发。
5. 对 8192-token 输入扫描 `max_prefill_tokens=8192/16384/32768`，同时观察 TTFT、TPOT 和峰值显存；必要时启用动态 chunking 做对照。
6. 若要定位计算与通信瓶颈，采集一次短矩阵的 Nsight Systems + NCCL trace，并同步记录 GPU SM、HBM、NVLink、功耗和 CPU scheduler 指标。

## 数据入口

- 原报告：[`REPORT.md`](REPORT.md)
- 聚合表：[`performance-summary.csv`](performance-summary.csv)
- SGLang 配置：[`configs/sglang-runtime.json`](configs/sglang-runtime.json)
- vLLM 诊断配置：[`configs/vllm-runtime.json`](configs/vllm-runtime.json)
- 正式逐请求结果：[`results/sglang/20260906T033021Z/`](results/sglang/20260906T033021Z/)
- 服务日志：[`logs/sglang-server.log`](logs/sglang-server.log)
