#!/usr/bin/env python3
"""Build the machine-readable summary and Chinese benchmark report."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path("/root/glm53-flash")
SGLANG_RUN = ROOT / "results/sglang/20260906T033021Z"


def metric_row(point: dict) -> dict:
    summary_path = SGLANG_RUN / point["name"] / "summary.json"
    summary = json.loads(summary_path.read_text())
    latency = summary["latency_ms"]
    throughput = summary["token_throughput_per_s"]
    return {
        "backend": "sglang",
        "point": point["name"],
        "concurrency": point["concurrency"],
        "input_tokens_per_request": point["input_tokens"],
        "output_tokens_per_request": point["output_tokens"],
        "requests": point["requests"],
        "completed": summary["counts"]["completed"],
        "failed": summary["counts"]["failed"],
        "duration_s": summary["measurement"]["duration_s"],
        "request_throughput_rps": summary["rps"]["completed"],
        "input_throughput_tok_s": throughput["input"],
        "output_throughput_tok_s": throughput["output"],
        "total_throughput_tok_s": throughput["total"],
        "ttft_p50_ms": latency["ttft"]["p50"],
        "ttft_p95_ms": latency["ttft"]["p95"],
        "tpot_p50_ms": latency["tpot"]["p50"],
        "tpot_p95_ms": latency["tpot"]["p95"],
        "e2e_p50_ms": latency["e2e"]["p50"],
        "e2e_p95_ms": latency["e2e"]["p95"],
        "source_summary": str(summary_path.relative_to(ROOT)),
    }


def fmt(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}"


def main() -> None:
    matrix = json.loads((ROOT / "configs/benchmark-matrix.json").read_text())
    rows = [metric_row(point) for point in matrix["points"]]

    csv_path = ROOT / "performance-summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "status": "complete_with_vllm_correctness_blocker",
        "model": {
            "path": matrix["model"],
            "served_name": matrix["served_model_name"],
            "checkpoint_revision": "04c4e9e95c5da8862dced7e5056455116f83a7e0",
        },
        "hardware": {"gpu_count": 8, "gpu": "NVIDIA H20 96 GB", "compute_capability": "9.0"},
        "method": {
            "seed": matrix["seed"],
            "stream": matrix["stream"],
            "temperature": matrix["temperature"],
            "top_p": matrix["top_p"],
            "ignore_eos": matrix["ignore_eos"],
            "warmup": "one full concurrency batch per point, minimum 2",
            "input_bench_commit": "9218661dee92e1c3b839fb78074b09a6542b81ea",
        },
        "sglang": {
            "status": "passed",
            "successful_requests": sum(row["completed"] for row in rows),
            "failed_requests": sum(row["failed"] for row in rows),
            "run": str(SGLANG_RUN.relative_to(ROOT)),
            "metrics": rows,
        },
        "vllm": {
            "status": "abandoned_after_correctness_failure",
            "formal_performance_benchmark": False,
            "host_conda_environment": {
                "path": "/opt/miniforge3/envs/glm53-vllm",
                "status": "insufficient_not_used",
                "vllm": "0.25.1+cu129",
                "flashinfer": "0.6.13",
            },
            "reason": "The API returned HTTP 200, but generated a degenerate repeated 'lock' token for both raw completion and chat-template smoke tests.",
            "image_manifest": "vllm/vllm-openai:glm53-flash",
            "image_platform_digest": "sha256:2e771fa615452282cc331eb418b3ef21636fce355bea0491fca89e6d362ab703",
            "vllm_version": "0.1.dev20051+g487ecf187",
        },
    }
    (ROOT / "performance-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )

    table_rows = []
    for row in rows:
        table_rows.append(
            "| {point} | {concurrency} | {input_tokens_per_request}/{output_tokens_per_request} | "
            "{completed}/{requests} | {output} | {input_tp} | {rps} | {ttft50}/{ttft95} | "
            "{tpot50}/{tpot95} | {e2e50}/{e2e95} |".format(
                **row,
                output=fmt(row["output_throughput_tok_s"]),
                input_tp=fmt(row["input_throughput_tok_s"]),
                rps=fmt(row["request_throughput_rps"], 3),
                ttft50=fmt(row["ttft_p50_ms"]),
                ttft95=fmt(row["ttft_p95_ms"]),
                tpot50=fmt(row["tpot_p50_ms"], 2),
                tpot95=fmt(row["tpot_p95_ms"], 2),
                e2e50=fmt(row["e2e_p50_ms"]),
                e2e95=fmt(row["e2e_p95_ms"]),
            )
        )

    report = """# GLM-5.3-Flash 本机部署与推理性能报告

生成日期：2026-09-06  
主机：8 × NVIDIA H20 96 GB（SM90，NVLink）  
模型：`/data/model/GLM-5.3-Flash`，原生 FP8，约 306 GiB，320B 总参数 / 18B 激活参数

## 结论

- **SGLang 部署成功且结果有效。** OpenAI 兼容接口通过健康检查和文本烟测；正式稳态矩阵 **448/448 请求成功、0 错误**。
- **vLLM 环境完成配置，但本轮放弃正式性能测试。** 官方 GLM-5.3-Flash day-0 镜像能加载模型并返回 HTTP 200，但同一检查点在 completion 和 chat 请求中都从第一个 token 起持续输出 `lock`。这种结果不具备推理正确性，因此没有把其强制定长吞吐写入对比表。
- SGLang 代表值：128→512、C1 为 **199.0 输出 tok/s**；128→512、C16 为 **949.9 输出 tok/s**；1024→256、C32 为 **1295.3 输出 tok/s**；8192→128、C16 为 **10640.9 输入 tok/s**。

## SGLang 稳态结果

| 场景 | 并发 | 输入/输出 token | 成功/请求 | 输出 tok/s | 输入 tok/s | 请求/s | TTFT p50/p95 ms | TPOT p50/p95 ms | E2E p50/p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{table}

以上是闭环客户端结果。并发请求等待信号量的时间记录为 `client_queue_delay`，不计入 TTFT；TTFT 从请求实际发出开始计时。`output tok/s` 是整段测量窗口内的聚合输出吞吐，TPOT 为逐请求 `(E2E - TTFT) / (输出 token - 1)`。

## SGLang 配置与资源

- 环境：`/opt/miniforge3/envs/glm53-sglang`；SGLang `0.0.0.dev1+gc767511ea`；PyTorch `2.13.0+cu130`；Transformers `5.12.1`；FlashInfer `0.6.18`。
- 该 Conda 环境是 editable 安装；源码树在本任务开始前已有 `load_model_utils.py` 的本地修改。本任务未改动它，状态已记录在 `system/sglang-worktree-status.txt`，因此严格跨机复现还需同步这项既有改动。
- 并行与内核：TP=8、EP=8；FP8 权重、BF16 KV；DSA TileLang prefill/decode；KDA Triton；DeepGEMM MoE；自适应 EAGLE MTP（5 steps、top-k 1、6 draft tokens）。
- 限制：32 个运行请求、32K context、8192 chunked prefill、`mem-fraction-static=0.70`。
- 冷启动：权重加载 28.00 s；scheduler 端到端 260.43 s；tokenizer/API 端到端 270.24 s。首次 DeepGEMM JIT 和 MTP target-verify 图捕获占主要时间。
- 服务报告的 KV token 池为 1,292,096 tokens；启动完成后可用 GPU 显存约 24.29 GiB/卡。

## vLLM 状态与诊断

使用官方专用镜像 `vllm/vllm-openai:glm53-flash` 的 amd64 平台镜像，固定 digest `sha256:2e771fa615452282cc331eb418b3ef21636fce355bea0491fca89e6d362ab703`。镜像内 vLLM 为 `0.1.dev20051+g487ecf187`，PyTorch `2.13.0+cu130`，Transformers `5.15.1`，FlashInfer `0.6.17`。

主机侧已建立 `/opt/miniforge3/envs/glm53-vllm`，但公开包目前只能得到 vLLM `0.25.1+cu129` 与 FlashInfer `0.6.13`，低于模型配方要求，故没有拿它启动服务。官方配方明确要求正式集成前使用专用 Docker；实际诊断均在上述固定镜像中完成。

已依次排除/尝试：

1. MTP=5 和 MTP=1：均在 sampling warm-up 阶段触发 CUDA scatter/gather 越界断言，服务不能就绪。
2. 关闭 MTP：服务可以就绪，图模式模型加载约 54 s，engine 初始化约 208 s，KV 池约 2,596,192 tokens，但输出退化为重复 `lock`。
3. 应用官方 Hopper 覆盖项 `--no-enable-flashinfer-autotune`。
4. 禁用 FlashInfer block-FP8 线性候选，改用纯 DeepGEMM。
5. 同时关闭 CUDA Graph、自定义 all-reduce，并强制 DeepGEMM linear + MoE。

第 3–5 项均未改变错误输出。上游 GLM-5.3 支持 PR 在 2026-09-03 才以 38 个提交合入；当前专用镜像是较早的 day-0 快照。上游同一 PR 的 H20 报告明确提到短序列 indexer 的 garbage-output/crash 修复，并另报 >4K prefill 的 H20 崩溃边界。因此这里把 vLLM 结果定性为**版本/实现正确性阻塞**，而不是模型文件损坏（同一权重在 SGLang 正常）。

`scripts/start_vllm_docker.sh` 保留最后一组诊断配置，但默认拒绝启动；仅设置 `GLM53_ALLOW_KNOWN_BAD_VLLM=1` 才可复现。建议等含完整 #53906 修复的新版官方镜像或 vLLM 0.29.0+ 正式包，再重跑本目录的相同冻结负载。

## 测试方法

- 基准工具：`/root/GPU-TEE-Inference-Bench`，commit `9218661dee92e1c3b839fb78074b09a6542b81ea`。
- 没有发现仓库文档预期的 `/data/benchmarks`，因此用其 ServeGen 适配器生成可审计的固定 token 工作负载；每个 workload 都有 SHA-256 manifest。
- 7 个点覆盖 decode-heavy、balanced、prefill-heavy，C1/C16/C32；固定 seed `20260906`、stream、temperature=0、top_p=1、ignore_eos=true。
- 每个点先预热至少一个完整并发批次；最终 SGLang 结果目录为 `results/sglang/20260906T033021Z`。早期未充分预热与缺少 parquet 依赖的试跑保留作诊断，不用于结论。

## 目录索引

- `performance-summary.csv` / `performance-summary.json`：机器可读汇总。
- `configs/`：运行时和基准矩阵。
- `workloads/`：冻结输入、源记录、manifest。
- `results/sglang/20260906T033021Z/`：正式逐点 `events.jsonl`、`summary.json`、`results.parquet`。
- `logs/`：SGLang 服务/烟测/基准日志，以及每种 vLLM 失败模式的完整日志。
- `system/`：GPU、拓扑、OS、Conda、Docker 镜像和模型文件哈希快照。
- `scripts/`：部署、负载生成、运行与报告脚本。

## 上游参考

- vLLM 官方 GLM-5.3-Flash 配方：https://recipes.vllm.ai/zai-org/GLM-5.3-Flash
- vLLM GLM-5.3 支持 PR：https://github.com/vllm-project/vllm/pull/53906
- H20 短序列/长 prefill 问题报告：https://github.com/vllm-project/vllm/pull/53906#issuecomment-5438954729
- 模型卡：https://huggingface.co/zai-org/GLM-5.3-Flash
""".format(table="\n".join(table_rows))
    (ROOT / "REPORT.md").write_text(report)


if __name__ == "__main__":
    main()
