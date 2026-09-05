# DeepSeek-V4-Flash H800 CVM 大规模性能与 Profile 报告

日期：2026-09-03（UTC）

## 结论摘要

- 完整性能 campaign 执行 38 个点、4,633 次请求；4,632 次成功，1 次 host→CVM `ServerDisconnectedError`。针对饱和点的补充 metrics 轮次执行 1,439 次请求并全部成功，原错误未复现。
- Chat 的稳态 closed-loop 吞吐从 concurrency 1 的 0.75 RPS 增长到 concurrency 64 的 9.45 RPS，但 TPOT P95 从 12.79 ms 增至 147.25 ms。
- Agent coding 和 summarization 的输入均值约为 7.1k/7.3k token，容量拐点大致位于 concurrency 8--32。到 concurrency 64 时，服务端 waiting 峰值分别为 57/53，TTFT P95 达 17.28/16.68 秒。
- Poisson 开环下，chat 4 RPS 尚有 0.81 goodput fraction；agent coding 与 summarization 的 4 RPS goodput 已接近 0。对长输入而言，能完成请求不等于满足交互 SLO。
- Mooncake conversation trace 的 256 个请求在 61.33 秒完成，4.17 RPS，TTFT P95 380.96 ms；补充采样测得 prefix-cache hit fraction 97.58%、waiting 峰值 0。这一结果不能与 cold-prefix 语义流直接比较，它体现了 trace 自带的高前缀复用。
- CVM 中 CUPTI 仍不可用。新的 12-request profile 只有 CPU op/user annotation，没有 CUDA kernel、memcpy 或 NCCL device activity；它用于解释执行边界，不作为性能基线。

## 被测环境

- 8 × NVIDIA H800 80GB，CVM 内 vLLM 0.25.0。
- 模型：`/models/DeepSeek-V4-Flash-0731`。
- TP=8、EP enabled、FP8 KV cache、block size 256、max model length 32768。
- Dense linear：DeepGEMM；MoE：Marlin；未使用 TensorRT-LLM。
- CUDA Graph：`FULL_AND_PIECEWISE`；speculative decoding：关闭；prefix caching：开启。
- Host endpoint：`http://127.0.0.1:18000`，经 QEMU user-network host forwarding 到 guest `:8000`。
- 性能 suite 未开启 Torch profiler；profile suite 通过 `/start_profile` 与 `/stop_profile` 单独执行。

## 数据与冻结方式

准备阶段使用模型的 129,280-token tokenizer，并直接复用 vLLM 0.25 的 DeepSeek-V4 message renderer。抽样请求与服务端 `/tokenize` 对 chat、agent coding、summarization 三类逐一校验，编译 token 数完全相同。

| class | source | requests in base | input token mean | input min--max | output cap mean |
|---|---|---:|---:|---:|---:|
| chat | ShareGPT | 512 | 867.7 | 64--3,213 | 110.4 |
| agent coding | Thoughtworks trajectories | 505 | 7,104.8 | 515--12,268 | 80.2 |
| summarization | Arxiv | 320 | 7,333.5 | 2,127--16,229 | 121.8 |
| long-context trace | Mooncake conversation | 256 | 9,984.4 | 1,033--29,852 | 114.2 |

固定并发和 Poisson 的每个测试点、每个请求都具有从首 token 开始不同的等价 nonce，避免测试点之间以及请求之间的 prefix-cache 污染。Mooncake trace 不添加 nonce，保留原始 `hash_ids` 所定义的共享前缀。

冻结计划包含：

- preflight：三类语义输入各 8 个长度分位样本，concurrency 8；不计入正式结果。
- fixed concurrency：三类 × concurrency 1/8/32/64 × 两次重复；每点 128 请求。
- Poisson：三类 × 0.5/1/2/4 RPS；每点 60 秒，最大客户端并发 256。
- timestamp trace：Mooncake conversation 256 请求，原时间轴等比例压缩到约 60 秒。
- profile：三类各 4 个请求，串行执行。

SLO 定义为 TTFT ≤ 2,000 ms、TPOT ≤ 50 ms、E2E ≤ 30,000 ms，三项必须同时满足才计入 goodput。

## Fixed concurrency 稳态结果

下表采用第二次重复。第一次重复保留在原始结果中，用于观察 JIT/cold-shape 成本。

| class | concurrency | RPS | input tok/s | output tok/s | TTFT P95 ms | TPOT P95 ms | E2E P95 ms | goodput fraction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chat | 1 | 0.75 | 679 | 76 | 266.92 | 12.79 | 1,896.74 | 1.00 |
| chat | 8 | 3.01 | 2,741 | 309 | 545.32 | 43.62 | 3,781.58 | 0.97 |
| chat | 32 | 6.31 | 5,735 | 645 | 1,277.83 | 73.94 | 7,617.81 | 0.72 |
| chat | 64 | 9.45 | 8,594 | 968 | 2,419.91 | 147.25 | 10,036.55 | 0.40 |
| agent coding | 1 | 0.86 | 5,752 | 56 | 632.23 | 12.58 | 2,171.66 | 1.00 |
| agent coding | 8 | 2.35 | 15,762 | 154 | 1,306.35 | 66.46 | 6,418.49 | 0.76 |
| agent coding | 32 | 3.01 | 20,212 | 197 | 8,639.99 | 193.86 | 20,382.33 | 0.05 |
| agent coding | 64 | 3.40 | 22,864 | 223 | 17,277.57 | 304.97 | 36,081.43 | 0.08 |
| summarization | 1 | 0.56 | 4,007 | 68 | 647.13 | 12.66 | 2,159.92 | 1.00 |
| summarization | 8 | 1.88 | 13,596 | 228 | 1,311.86 | 36.50 | 5,134.11 | 0.98 |
| summarization | 32 | 2.73 | 19,660 | 331 | 7,524.76 | 101.27 | 17,105.43 | 0.08 |
| summarization | 64 | 3.03 | 21,826 | 368 | 16,683.18 | 166.62 | 36,672.44 | 0.08 |

最明显的首次执行效应出现在 chat concurrency 1：第一轮仅 0.43 RPS、TTFT P95 3,307 ms；第二轮为 0.75 RPS、267 ms。服务使用 `VLLM_DEEP_GEMM_WARMUP=skip`，而请求输入长度高度离散，因此单独的 preflight 仍不能覆盖所有首次 shape JIT。CVM/裸金属必须以相同顺序执行完整 plan，并分别比较 r1 与 r2，不能只挑其中较好的数值。

## Poisson 开环结果

| class | offered RPS | requests | completed RPS | input tok/s | TTFT P95 ms | TPOT P95 ms | E2E P95 ms | goodput fraction | failures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chat | 0.5 | 26 | 0.43 | 321 | 351 | 15.37 | 2,115 | 1.00 | 0 |
| chat | 1 | 64 | 1.05 | 856 | 480 | 28.25 | 2,889 | 0.98 | 0 |
| chat | 2 | 106 | 1.71 | 1,502 | 426 | 26.09 | 3,322 | 1.00 | 0 |
| chat | 4 | 231 | 3.74 | 3,300 | 580 | 60.85 | 7,565 | 0.81 | 0 |
| agent coding | 0.5 | 26 | 0.44 | 3,569 | 851 | 26.66 | 3,081 | 1.00 | 0 |
| agent coding | 1 | 64 | 1.06 | 7,494 | 1,006 | 49.01 | 4,055 | 0.95 | 0 |
| agent coding | 2 | 106 | 1.72 | 11,557 | 1,207 | 69.23 | 6,362 | 0.78 | 0 |
| agent coding | 4 | 231 | 3.43 | 23,570 | 6,479 | 326.62 | 39,562 | 0.004 | 0 |
| summarization | 0.5 | 26 | 0.43 | 3,098 | 960 | 20.74 | 3,305 | 1.00 | 0 |
| summarization | 1 | 64 | 1.04 | 7,213 | 1,185 | 35.02 | 4,510 | 1.00 | 0 |
| summarization | 2 | 106 | 1.68 | 11,930 | 1,267 | 78.86 | 11,069 | 0.81 | 1 |
| summarization | 4 | 231 | 3.09 | 23,096 | 12,116 | 330.53 | 46,777 | 0.00 | 0 |

Summarization 2 RPS 的一次失败为 `ServerDisconnectedError`，请求尚未收到 HTTP headers，服务始终健康。相同 workload 的补充轮次 106/106 成功，故暂记为不可复现的 host↔CVM 传输异常，不从主结果中删除。

## 饱和点服务端 metrics

补充轮次每 0.5 秒采样 vLLM `/metrics`，1,439/1,439 请求成功。

| point | max running | max waiting | max KV usage | prefix hit fraction |
|---|---:|---:|---:|---:|
| chat fixed c64 | 64 | 30 | 11.62% | 0% |
| agent coding fixed c64 | 64 | 57 | 16.10% | 0% |
| summarization fixed c64 | 64 | 53 | 16.67% | 0% |
| chat Poisson 4 RPS | 30 | 0 | 4.20% | 0% |
| agent coding Poisson 4 RPS | 83 | 21 | 19.02% | 0% |
| summarization Poisson 2 RPS | 30 | 2 | 11.95% | 0% |
| summarization Poisson 4 RPS | 137 | 37 | 26.51% | 0% |
| Mooncake trace | 22 | 0 | 8.49% | 97.58% |

这里的 KV usage 峰值不高，但长输入在 attention/prefill 计算上已经饱和，因此不能用 KV 是否接近 100% 判断服务是否还有 latency capacity。

## Timestamp trace

Mooncake conversation 结果：256/256 成功，duration 61.33 秒，4.17 RPS，名义 input throughput 41.68k token/s、output throughput 476.67 token/s，TTFT P95 380.96 ms，TPOT P95 26.31 ms，E2E P95 3,535.38 ms，goodput fraction 1.00。

`input tok/s` 按请求完整 prompt usage 计算，不会扣除 prefix cache 已复用的 token，因此它是业务吞吐，不是实际 prefill compute throughput。97.58% 的 prefix hit 正是该 trace 性能显著优于 cold-prefix agent/summarization 的主要原因。

## 新的 CPU Profile

Profile workload 含 12 个 cold-prefix 请求，总输入 48,141 token、输出 1,301 token。客户端观测 TTFT P95 677.01 ms、TPOT P95 13.87 ms、E2E P95 2,247.92 ms。

Rank 0 trace：

- 799,350 events，其中 796,694 `cpu_op`、2,645 `user_annotation`，无 CUDA activity。
- 14 个主要 prefill/chunked-prefill `execute_context`，CPU-visible 总计 3,128.34 ms；出现两次 8,192-token chunk。
- 1,291 个 decode `execute_context`，均值 5.291 ms、P95 6.688 ms、最大 7.992 ms。8 ranks 的均值范围为 3.929--5.575 ms。
- `vllm::all_gather` 1,305 次，CPU-visible 累计 326.84 ms。
- `vllm::sparse_attn_indexer` 294 次，累计 96.56 ms。
- 大于 1 ms 的 `aten::copy_` 共 1,300 次，全部为 `[1,1] → [1,1]`、`non_blocking=True` 的 int copy，累计 10,174.79 ms。这与逐 step sampled-token GPU→CPU copy 对齐，但仍不能解释为纯 PCIe D2H 时间。

Profiler trace 会显著改变时序；裸金属侧应使用同一 profile workload，再用 CUPTI/Nsight Systems 分解 CUDA kernel、H2D/D2H、NCCL 与 CPU synchronization。

## 裸金属复跑

不要再次执行 `prepare`。直接复用 `campaigns/dsv4-large-scale/plan.json` 和全部 `workloads/`：

```bash
cd /home/user007/input-bench

# 假设裸金属 vLLM 监听 8000；服务参数必须与本报告一致。
scripts/run_dsv4_campaign.sh baremetal performance http://127.0.0.1:8000
scripts/run_dsv4_campaign.sh baremetal profile http://127.0.0.1:8000
```

性能 suite 会为每个点额外保存 0.5 秒粒度的 `server-metrics.jsonl`。只复跑指定点可使用：

```bash
python3 scripts/large_scale_campaign.py run \
  --mode baremetal --suite performance \
  --base-url http://127.0.0.1:8000 \
  --point 'poisson-*-rps4p0' --point 'fixed-*-c64-r2'
```

Profile 的 server trace 输出位置由启动 vLLM 时的 `--profiler-config` 决定。当前 CVM trace 已复制到本 campaign 的 profile result 目录，虚拟机关闭不会丢失。
