# 输入建模与 benchmark 调研

## 结论

只复用 ShareGPT 内容或只扫固定长度都不够。对 CVM A/B，至少要冻结六个维度：`(ISL, OSL)` 的联合分布、到达过程、并发/会话、prefix 复用、正文类别、采样/EOS。前四项决定调度和 KV 行为；正文与采样还会改变 MoE 路由以及 MTP/DSpark acceptance。

## 论文和标准如何测

| 来源 | workload | 主要指标 | 值得复用的设计 |
|---|---|---|---|
| Orca, OSDI'22 | 合成生成请求，扫 batch/到达负载 | latency-throughput | iteration-level batching 的早期基线 |
| vLLM/PagedAttention, SOSP'23 | 从 ShareGPT 采样输入/输出长度 | request throughput at matched latency | 用真实长度分布测试 continuous batching/KV 管理 |
| DistServe, OSDI'24 | ShareGPT chat、HumanEval code、LongBench summarization；Poisson arrival | TTFT、TPOT、在双 SLO 下的 per-GPU goodput | 按应用分别给 SLO，不能只报无限 request-rate throughput |
| Sarathi-Serve, OSDI'24 | ShareGPT 与 arXiv summarization | median TTFT、p99 TBT、goodput | 同时覆盖 decode-heavy 与 prefill-heavy，并观察 chunked prefill 干扰 |
| Vidur, MLSys'24 | Azure/Splitwise 等 trace 与可配置 arrival/length | TTFT、TBT/TPOT、E2E、throughput、operator time | workload、scheduler、operator profile 三层分离 |
| Mooncake, FAST'25 | 生产 trace：timestamp、ISL、OSL、prefix block hash | throughput、cache hit、TTFT/TPOT | prefix locality 是独立维度，不能从长度分布推断 |
| BurstGPT, KDD'25 | 5M+ Azure 请求，保留 timestamp、ISL、OSL、session | burst/concurrency、稳定性、吞吐/延迟 | 重放非平稳和 bursty arrival，而非只用 Poisson |
| MLPerf Inference | Offline + Server；Server 使用 Poisson arrival | output tok/s，受 p99 TTFT/TPOT SLO 约束 | 用 SLO-constrained throughput/goodput 做最终服务指标 |

主要资料：[Orca](https://www.usenix.org/conference/osdi22/presentation/yu)、[vLLM paper](https://arxiv.org/abs/2309.06180)、[DistServe](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf)、[Sarathi-Serve](https://www.usenix.org/system/files/osdi24-agrawal.pdf)、[Vidur](https://proceedings.mlsys.org/paper_files/paper/2024/hash/b74a8de47d2b3c928360e0a011f48351-Abstract-Conference.html)、[Mooncake](https://www.usenix.org/conference/fast25/presentation/qin)、[BurstGPT](https://github.com/HPMLL/BurstGPT)、[MLPerf Llama 2](https://mlcommons.org/2024/03/mlperf-llama2-70b/)。

## 主流框架的 benchmark 口径

| 框架 | 自带 workload | 输出指标/特点 |
|---|---|---|
| vLLM | ShareGPT、random、BurstGPT、prefix repetition、HF、multimodal | request/input/output throughput；TTFT/TPOT/ITL/E2E percentile；open-loop request rate 或 closed-loop concurrency |
| SGLang | ShareGPT、random/random-ids、generated shared prefix、Mooncake、SPEED-Bench、multimodal | 与 vLLM 类似，额外覆盖 prefix locality 和 speculative decoding entropy |
| TensorRT-LLM | 固定或正态 ISL/OSL JSON dataset；online server | request、output/total/per-user/per-GPU tok/s；TTFT、ITL、request latency percentiles |
| DeepSpeed-FastGen | 正态 ISL/OSL，client 1–32 | latency-throughput curve；满足 prompt latency 与 generation EMA SLA 的 effective throughput |

官方资料：[vLLM bench](https://docs.vllm.ai/en/stable/benchmarking/cli/)、[SGLang bench](https://github.com/sgl-project/sglang/blob/main/docs/docs/developer_guide/bench_serving.mdx)、[TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/developer-guide/perf-benchmarking.html)、[DeepSpeed-FastGen](https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/deepspeed-fastgen/README.md)。

## 指标定义

- TTFT：client 发出请求到收到第一个 token；包含排队、调度、H2D 和 prefill。另报 `service TTFT` 才能与排队分开。
- ITL：相邻 token 到达间隔的逐 token 分布；p99 ITL 能看到 stall。
- TPOT：每请求 `(E2E - TTFT)/(实际输出 token - 1)`，再按请求等权聚合。不要和 token-weighted ITL 混称。
- output tok/s：测量窗口内实际输出 token / 秒；另报 input tok/s，禁止用 total tok/s 掩盖 decode。
- goodput：同时满足 TTFT 与 TPOT/ITL SLO 的请求率。建议报告最大 sustainable RPS，而非只报 `request-rate=inf`。
- acceptance：MTP/DSpark 的 accepted tokens / drafted tokens、acceptance length、每 position acceptance；必须使用有语义正文并固定采样参数。
- fairness/stability：按短/长请求和 session 分组的 slowdown、SLO violation、p99/max ITL；至少 10 分钟 trace 或足够覆盖 ramp/steady/drain。

## H800 最小实验矩阵

第一阶段先建立机制面：ISL `{128,1K,4K,8K,32K}` × OSL `{1,32,256,2K}` × concurrency `{1,8,32,64}`。`OSL=1` 近似 prefill，短 ISL/长 OSL 放大 decode；每点 warmup 后至少 3 次，bare/CVM 交替，并以 run median 配对。

第二阶段用三类输入：冻结 ShareGPT（chat）、长文 summarization、代码/推理。分别做 Poisson open-loop 的 30/50/70/85% capacity，以及 BurstGPT/Mooncake trace replay。prefix caching on/off 要作为独立实验，不和 CUDA Graph/MTP 一次同时变化。

第三阶段做内容敏感优化：DSpark off/on，draft depth `{1,3,5,7}`（以实际框架支持为准），记录 acceptance 和 verifier forward 次数。随机 token 仅用于 kernel/shape 对照，不能用于推断生产加速。

所有结果必须保存：精确 model/tokenizer revision、框架/torch/CUDA/driver/NCCL、完整 server argv、环境变量、GPU clocks/power、实际 ISL/OSL、到达时间、失败/取消请求，以及 graph capture 完成后的稳态区间。
