#!/usr/bin/env python3
"""Run the frozen benchmark matrix with Input Bench against one backend."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/glm53-flash")
BENCH_REPO = Path("/root/GPU-TEE-Inference-Bench")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("sglang", "vllm"), required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="GLM-5.3-Flash")
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="minimum warm-up requests; each point warms at least one full concurrency batch",
    )
    args = parser.parse_args()

    matrix = json.loads((ROOT / "configs/benchmark-matrix.json").read_text())
    env = os.environ.copy()
    env["PYTHONPATH"] = str(BENCH_REPO / "src")
    env.setdefault("OPENAI_API_KEY", "EMPTY")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result_root = ROOT / "results" / args.backend / stamp
    result_root.mkdir(parents=True, exist_ok=False)
    (result_root / "run-config.json").write_text(
        json.dumps(
            {
                "backend": args.backend,
                "base_url": args.base_url,
                "model": args.model,
                "minimum_warmup": args.warmup,
                "warmup_policy": "max(minimum_warmup, point concurrency)",
                "matrix": matrix,
                "input_bench_repo": str(BENCH_REPO),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    for point in matrix["points"]:
        workload = ROOT / "workloads" / point["name"] / "workload.jsonl"
        output = result_root / point["name"]
        point_warmup = max(args.warmup, point["concurrency"])
        command = [
            sys.executable,
            "-m",
            "input_bench.cli",
            "run",
            "--workload",
            str(workload),
            "--backend",
            args.backend,
            "--base-url",
            args.base_url,
            "--model",
            args.model,
            "--tokenizer",
            matrix["model"],
            "--output-dir",
            str(output),
            "--max-concurrency",
            str(point["concurrency"]),
            "--pool-size",
            str(max(64, point["concurrency"] * 2)),
            "--timeout",
            "900",
            "--warmup",
            str(point_warmup),
        ]
        print(f"running {args.backend} {point['name']}", flush=True)
        subprocess.run(command, cwd=BENCH_REPO, env=env, check=True)
    print(f"results: {result_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
