# DeepSeek-V4-Flash vLLM 裸金属性能与 Profile 报告

执行时间：2026-09-03 UTC
测试端：裸金属本机 `http://127.0.0.1:8000`
对照端：同一冻结 workload 的 H800 机密虚拟机结果

## 结论

- 裸金属 performance campaign 完成 37 个正式测试点、4,609 个请求，失败 0；另有 24 请求的 preflight，结果不纳入正式比较。
- profile workload 完成 12/12 请求，失败 0。8 个 TP rank 均生成了包含 CUDA activity 的完整 trace，总大小约 1.2 GiB；CVM trace 因 CC 环境禁止 CUPTI，仅有 CPU/user annotation。
- 固定并发 r2 中，裸金属 input-token throughput 相比 CVM 高 `7.2%--46.0%`。差异随 agent/summarization 并发升高而收窄，但 chat 各并发仍有约 `40%` 的差距。
- profile-mix 中，裸金属相对 CVM 的 p95 TTFT、TPOT、E2E 分别降低 `25.8%`、`30.1%`、`31.8%`。
- 这轮结果不能解释成纯 CC overhead：裸金属与 CVM 的驱动、kernel、CPU/NUMA 暴露和系统编译链并未完全对齐；CVM 请求还经过 host 到 guest 的本地端口转发，而裸金属是直接 loopback。它首先是一组可复跑的端到端环境对比，也是后续收敛变量的基线。

## 产物位置

- 裸金属性能结果：`results/20260903T164113Z-baremetal-performance/`
- 裸金属 profile 结果：`results/20260903T172455Z-baremetal-profile/`
- 裸金属服务端 trace：`results/20260903T172455Z-baremetal-profile/server-traces/`
- CVM 性能对照：`results/20260903T125401Z-cvm-performance/`
- CVM profile 对照：`results/20260903T135600Z-cvm-profile/`
- 服务日志：`/data/vllm-logs/v025-baremetal-tmux-attempt2.log`
- campaign 日志：`baremetal-performance.log`、`baremetal-profile.log`

两次 run 的冻结 plan SHA-256 均为：

```text
f61d1d6b47114f5de17d3939b9505e5843d364988d820fa051d2cc80e6a8a0fa
```

## 服务与环境

| 项目 | 裸金属值 |
|---|---|
| 主机 | `huaxiyun` |
| GPU | 8 × NVIDIA H800，GPU 间均为 NV8 |
| Driver | 595.71.05 |
| Kernel | Ubuntu `7.0.0-30-generic` |
| CPU | 2 × Xeon Platinum 8558，192 logical CPU，4 NUMA node |
| venv | `/data/venvs/vllm025` |
| Python | 3.12.14 |
| vLLM | 0.25.0 |
| PyTorch | 2.11.0+cu130 |
| CUDA toolkit/runtime | nvcc 13.0.88 / runtime 13.0.96 |
| Transformers | 5.16.1 |
| NCCL | 2.28.9 |
| FlashInfer / Triton / TileLang | 0.6.13 / 3.6.0 / 0.1.9 |
| glibc / host GCC | 2.43 / 15.2.0 |

模型关键文件 SHA-256 与 CVM 完全一致。服务只使用 vLLM，没有使用 TensorRT-LLM。关键配置为 TP=8、EP enabled、FP8 KV、block size 256、max model length 32768、DeepGEMM linear、Marlin MoE。

服务日志确认：

- `enforce_eager=False`，CUDA Graph 模式为 `FULL_AND_PIECEWISE`；
- graph capture size 最大 512，每卡估计 graph memory 7.38 GiB；
- prefix caching 与 chunked prefill 开启；
- `speculative_config=None`，本轮没有启用 MTP/speculative decoding。

### 裸金属兼容处理

裸金属系统的 `/usr/local/cuda` 为 CUDA 13.1，而 CVM 对照环境使用 CUDA 13.0。为避免 JIT 在两套 toolkit 之间漂移，启动脚本现在显式设置：

```text
CUDA_HOME=/data/venvs/vllm025/lib/python3.12/site-packages/nvidia/cu13
CUDA_PATH=/data/venvs/vllm025/lib/python3.12/site-packages/nvidia/cu13
```

此外，glibc 2.43 的 `rsqrt/rsqrtf` exception specification 与 CUDA 13.0 header 冲突。本轮在隔离 venv 的 `nvidia/cu13/include/crt/math_functions.h` 中给两个声明增加了 `noexcept(true)`。这是使 DeepGEMM JIT 在当前裸金属系统上可编译的兼容修补，不涉及模型或 vLLM 源码，但它仍属于 CVM/裸金属环境差异，必须在严格归因时记录。

## 大规模性能结果

每个 fixed 点为 128 请求，r1/r2 各一遍。由于 `VLLM_DEEP_GEMM_WARMUP=skip`，preflight 和部分 chat r1 包含明显的首次 shape/JIT 开销，因此下面用 r2 做主要比较。

| workload | concurrency | 裸金属 input tok/s | 相对 CVM | 裸金属 p95 TTFT ms | p95 TPOT ms | p95 E2E ms |
|---|---:|---:|---:|---:|---:|---:|
| chat | 1 | 949.4 | +39.8% | 174.7 | 8.10 | 1,209.0 |
| chat | 8 | 4,001.8 | +46.0% | 383.9 | 28.12 | 2,556.1 |
| chat | 32 | 8,176.3 | +42.6% | 1,034.9 | 50.14 | 4,872.2 |
| chat | 64 | 12,190.9 | +41.8% | 1,965.6 | 115.96 | 6,714.6 |
| agent coding | 1 | 7,916.5 | +37.6% | 522.1 | 8.12 | 1,479.3 |
| agent coding | 8 | 19,077.2 | +21.0% | 1,036.0 | 50.70 | 5,529.9 |
| agent coding | 32 | 23,375.1 | +15.6% | 8,278.8 | 173.61 | 17,897.7 |
| agent coding | 64 | 24,511.3 | +7.2% | 16,716.0 | 299.01 | 33,851.0 |
| summarization | 1 | 5,449.7 | +36.0% | 609.6 | 8.13 | 1,632.0 |
| summarization | 8 | 16,657.9 | +22.5% | 1,229.0 | 30.44 | 4,428.4 |
| summarization | 32 | 21,995.2 | +11.9% | 7,484.3 | 86.31 | 16,346.1 |
| summarization | 64 | 23,648.5 | +8.4% | 15,727.8 | 156.44 | 33,976.2 |

Poisson/trace 的主要观察：

- chat 4 RPS：goodput fraction 从 CVM 的 0.810 提升到 1.000，p95 E2E 降低 58.2%；
- agent coding 4 RPS：两端都进入严重排队区，goodput 从 0.004 提升到 0.030，仍远低于可服务区；
- summarization 4 RPS：两端 goodput 都为 0，说明 offered load 已越过当前 SLO 下的容量边界；
- Mooncake timestamp trace：裸金属 p95 TTFT/TPOT/E2E 分别为 229.7/13.67/1,857.2 ms，相比 CVM 降低 39.7%/48.0%/47.5%。

全部 38 行及服务端 running/waiting/KV-cache 指标见裸金属结果目录的 `summary.csv`。本轮裸金属每个测试点都采集了 `server-metrics.jsonl`。

## Profile 结果

profile-mix 输入 48,141 token、输出 1,301 token，串行发送 12 个请求。客户端指标如下：

| 指标 | CVM | 裸金属 | 裸金属变化 |
|---|---:|---:|---:|
| measurement duration | 21.917 s | 15.097 s | -31.1% |
| input throughput | 2,196.5 tok/s | 3,188.8 tok/s | +45.2% |
| output throughput | 59.36 tok/s | 86.18 tok/s | +45.2% |
| p95 TTFT | 677.0 ms | 502.4 ms | -25.8% |
| p95 TPOT | 13.87 ms | 9.70 ms | -30.1% |
| p95 E2E | 2,247.9 ms | 1,533.8 ms | -31.8% |
| goodput fraction | 1.000 | 1.000 | unchanged |

`/stop_profile` 在请求测量结束后约花 109 秒聚合并压缩 8-rank trace；该时间不计入上表请求 latency。

### rank 0 trace 初步拆解

以下是 rank 0 的流式聚合。`sum duration` 是同类 event 的累加时间，可能跨 stream 重叠，不能当作端到端 wall-time 占比。

- prefill/context annotation：14 次，覆盖 48,141 context token，累计 2,117.8 ms；14 而非 12 是因为长 prompt 发生 chunked prefill。
- decode/generation annotation：1,291 次，累计 8,775.6 ms。
- `cudaGraphLaunch`：1,335 次；数量与 decode step 接近，印证 decode 主要走 CUDA Graph。
- GPU kernel event：2,689,787 个，累计 16,735.9 ms。
- `_all_gather_base` annotation：1,305 次；trace 中还能直接看到 `cross_device_reduce` 和 `multimem_all_reduce` kernel。

按 kernel 名称启发式聚类：

| kernel group | event count | aggregate duration | 占 aggregate kernel time |
|---|---:|---:|---:|
| collective/reduce | 114,840 | 3,696.7 ms | 22.1% |
| DeepGEMM | 395,286 | 2,689.4 ms | 16.1% |
| attention/MLA | 223,802 | 2,079.4 ms | 12.4% |
| MoE/Marlin | 112,230 | 2,018.1 ms | 12.1% |
| quantization | 421,680 | 1,165.8 ms | 7.0% |
| MHC | 224,460 | 1,155.1 ms | 6.9% |
| other | 1,197,489 | 3,931.5 ms | 23.5% |

设备拷贝 activity：

| direction | event count | bytes | aggregate duration | 平均每次 |
|---|---:|---:|---:|---:|
| H2D pinned→device | 20,894 | 73.90 MiB | 32.57 ms | 1.56 us |
| D2H device→pinned | 1,305 | 5.1 KiB | 3.09 ms | 2.37 us |
| D2D device→device | 11,927 | 80.72 GiB | 64.82 ms | 5.44 us |

D2D 只表示 device memory copy，不能等同于 GPU-to-GPU collective。跨 GPU 路径应结合 collective kernel、NCCL annotation，以及下一轮的 NCCL/NVLink 计数器单独分析。

CUDA runtime 累加时间最大的两项是 `cudaEventSynchronize`（3,923 次、13.65 s）和 `cudaGraphLaunch`（1,335 次、4.89 s）。这些是 CPU API duration 且可能包含等待与 profiler 扰动，目前只能作为进一步定位同步边界的线索，不能直接宣称它们占用了相同比例的请求 wall time。

## 下一轮建议

1. 先对齐 driver、kernel、CPU affinity/NUMA、power/clock policy，再复跑相同冻结 plan，收敛纯 CVM overhead。
2. 将 profile workload 分成纯 prefill、纯 decode steady-state 和 mixed 三套；当前混合 trace 适合找热点，但不适合直接归因阶段差异。
3. 裸金属上增加 Nsight Systems 或 CUPTI counter 实验，重点核对 `cross_device_reduce`、multimem all-reduce、`_all_gather_base` 与 NVLink 流量；PyTorch trace 已经提供可对齐的时间轴锚点。
4. 单独做 `enforce_eager=False/True` A/B，随后再启用 MTP；本轮 MTP 关闭，不能把任何收益或开销归因到 speculative decoding。

## 当前状态

实验完成后已优雅停止 `vllm-server`。APIServer、EngineCore 和 8 个 worker 均已退出，端口 8000 已释放，8 张 H800 显存均回到 `0 MiB`。性能结果和 profile trace 已独立保存，不依赖服务继续运行。
