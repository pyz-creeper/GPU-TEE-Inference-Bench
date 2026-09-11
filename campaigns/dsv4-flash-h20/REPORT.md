# DeepSeek V4 Flash / H20 单流 Profile 报告

## 结论

本机历史结果已经整理到本目录。可作为基线的主结果是 2026-08-04 14:25:08 的同轮 TP=4/TP=8 对照：

| 配置 | 单流 decode 均值 | 范围 | 平均 TPOT | GPU 数量 | 相对 TP=4 |
|---|---:|---:|---:|---:|---:|
| TP=4 | 100.47 tok/s | 99.8–101.2 tok/s | 约 9.95 ms | 4 | 基线 |
| TP=8 | 106.19 tok/s | 105.5–107.0 tok/s | 约 9.42 ms | 8 | +5.69% |

两种配置都明显超过 50 tok/s 的单流 decode 目标。TP=8 多使用 4 张 H20，只换来约 5.7% decode 提升；如果重点是单流资源效率，TP=4 更合适。如果重点是该模型在本机的最高单流速度，TP=8 略快。

## 测试对象和配置

- 模型：`/data/model/DeepSeek-V4-Flash-0731`
- 推理框架：vLLM offline `LLM.generate`
- 硬件：本机 8×NVIDIA H20，每卡 97,871 MiB；当前拓扑显示 GPU 间为 NV18
- 并行方式：单机 TP=4 或 TP=8
- `max_model_len=65536`
- `gpu_memory_utilization=0.90`
- KV cache：FP8
- 单请求、batch=1、concurrency=1
- 每个输入固定生成 256 token，`temperature=0`，`ignore_eos=True`
- 只在最短输入上做过一次 warmup，之后每种长度只测一次

原始结果没有记录 vLLM、PyTorch、CUDA、驱动版本，也没有保存当时的完整启动日志。因此不能用当前环境版本倒推历史运行版本；报告不会把今天查询到的软件版本误写成当时版本。

## 输入口径

测试脚本重复一段固定英文文本，并按 `context_length × 4` 个字符截断。`context_length` 是生成器目标值，不是 tokenizer 后的真实 token 数。真实输入 token 数如下：

| 目标 context_length | 实际 prompt token |
|---:|---:|
| 512 | 330 |
| 1,024 | 660 |
| 2,048 | 1,328 |
| 4,096 | 2,655 |
| 8,192 | 5,312 |
| 16,384 | 10,626 |
| 32,768 | 21,261 |

因此分析性能时应优先看 `prompt_tokens`，不能把 `context_length` 直接当作输入 token 数。

## 主基线：20260804_142508

| 实际输入 token | TP=4 TTFT | TP=4 decode | TP=4 total | TP=8 TTFT | TP=8 decode | TP=8 total | TP=8 decode 提升 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 330 | 66.5 ms | 101.2 tok/s | 2,586 ms | 67.2 ms | 105.6 tok/s | 2,483 ms | 4.35% |
| 660 | 67.1 ms | 101.2 tok/s | 2,587 ms | 66.2 ms | 107.0 tok/s | 2,451 ms | 5.73% |
| 1,328 | 392.0 ms | 101.0 tok/s | 2,917 ms | 389.0 ms | 106.9 tok/s | 2,774 ms | 5.84% |
| 2,655 | 418.6 ms | 100.1 tok/s | 2,966 ms | 402.8 ms | 106.2 tok/s | 2,804 ms | 6.09% |
| 5,312 | 376.0 ms | 100.0 tok/s | 2,926 ms | 273.2 ms | 105.9 tok/s | 2,681 ms | 5.90% |
| 10,626 | 675.3 ms | 100.0 tok/s | 3,225 ms | 500.1 ms | 106.2 tok/s | 2,902 ms | 6.20% |
| 21,261 | 1,380.5 ms | 99.8 tok/s | 3,934 ms | 1,187.2 ms | 105.5 tok/s | 3,602 ms | 5.71% |

TP=4 的干净复测 `20260805_150746` 得到 100.87 tok/s decode 均值，范围 100.0–102.1 tok/s，与主基线吻合。

## 8 月 10 日两轮异常复测

这两轮的 decode 仍然稳定，但若干长度的 TTFT 出现秒级停顿，不能作为稳定 prefill/TTFT 基线：

| 运行 | TP | decode 均值 | 明显异常的 TTFT |
|---|---:|---:|---|
| `20260810_131024` | 4 | 98.13 tok/s | 1,328 token: 30.41 s；2,655 token: 19.01 s；10,626 token: 3.64 s |
| `20260810_144607` | 4 | 99.37 tok/s | 1,328 token: 25.63 s；2,655 token: 31.20 s；5,312 token: 3.22 s |

原始数据没有 profiler trace 或服务日志，无法事后证明停顿来自 JIT/图编译、CPU 调度、I/O 或其他系统干扰。由于脚本仅预热最短 shape，并且每个长度只取一个样本，shape 首次执行开销是合理嫌疑，但只能作为推测。两轮数据仍归档，供后续诊断和对照使用，不参与主基线结论。

## 指标怎么算

- TTFT：vLLM request metrics 的 `first_token_latency`。
- TPOT：`(last_token_ts - first_token_ts) / (num_generation_tokens - 1)`。
- decode tok/s：`1 / TPOT`。
- 脚本中的 `prefill_ms`：`TTFT - TPOT`，是近似拆分，不是 kernel 级精确 prefill 时间。
- `total_ms`：在 `torch.cuda.synchronize()` 包围下测得的整次 `LLM.generate` 墙钟时间。
- `avg_gpu_mem_mb`：请求结束后各参与 GPU 的显存快照均值，并非请求全过程的时间平均或峰值。

## 结果能说明什么、不能说明什么

能说明：在这台 H20 机器、这个模型和上述单请求口径下，decode 约为 100–106 tok/s；TP=8 对 TP=4 的单流 decode 收益较小。

不能说明：多并发服务吞吐、复杂会话 workload 的吞吐、P50/P95/P99 延迟、CUDA kernel 分项耗时、NCCL 通信耗时，或 CVM/TEE 开销。它也不能与 conversation、tool-agent 或 timestamp-trace campaign 的 aggregate throughput 直接横向比较。

要得到可发表的稳定 TTFT/prefill 数据，后续应对每个输入长度独立 warmup，再做多次重复并报告 median/P95；同时固定并记录软件版本、GPU 时钟/功耗和后台负载。要拆分 kernel 与 TP 通信耗时，则需要 Nsight Systems/NCCL trace 或 PyTorch profiler，当前这些原始文件不包含该信息。

## 原始文件

- 主 TP4/TP8 对照：`profile_results/20260804_142508/`
- TP4 干净复测：`profile_results/20260805_150746/`
- TP4 异常诊断轮次：`profile_results/20260810_131024/`、`profile_results/20260810_144607/`

每个完整目录中的 `full_results.json` 是结构化记录，CSV 便于画图和二次统计，`report.txt` 是当时脚本自动生成的原始文本。原始文件按字节复制，没有修正其中的显示格式或重新计算字段。
