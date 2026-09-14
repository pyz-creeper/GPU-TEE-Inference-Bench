# DSV4 Flash H20 Profile

本目录归档本机 8×NVIDIA H20 上 DeepSeek V4 Flash 的历史 vLLM 单流 profile 结果。

- [REPORT.md](REPORT.md)：测试口径、主结果、复测结果与使用限制
- [SOURCE_INVENTORY.md](SOURCE_INVENTORY.md)：`/root/phase0/profile_results` 全量来源审计及排除原因
- `profile_results/`：4 轮确认属于 DSV4 的原始 CSV、JSON 和生成报告

主基线是 `profile_results/20260804_142508/`，它是唯一在同一轮内同时包含 TP=4 和 TP=8 的完整对照。单流 decode 均值分别为 100.47 tok/s 和 106.19 tok/s；TP=8 相对 TP=4 平均提升 5.69%。

这里的“profile”是端到端请求计时，不是 CUDA kernel trace。不要将这些单请求结果直接当作 GPU-TEE-Inference-Bench 复杂 workload 的在线吞吐。
