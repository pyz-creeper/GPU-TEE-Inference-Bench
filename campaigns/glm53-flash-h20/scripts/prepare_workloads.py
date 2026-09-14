#!/usr/bin/env python3
"""Compile deterministic, tokenizer-exact Input Bench workloads for both servers."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path("/root/glm53-flash")
BENCH_REPO = Path("/root/GPU-TEE-Inference-Bench")
MATRIX_PATH = ROOT / "configs/benchmark-matrix.json"
WORKLOAD_ROOT = ROOT / "workloads"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    matrix = json.loads(MATRIX_PATH.read_text())
    env = os.environ.copy()
    env["PYTHONPATH"] = str(BENCH_REPO / "src")
    for point in matrix["points"]:
        target = WORKLOAD_ROOT / point["name"]
        target.mkdir(parents=True, exist_ok=True)
        source = target / "servegen-source.jsonl"
        with source.open("w", encoding="utf-8") as handle:
            for index in range(point["requests"]):
                row = {
                    "request_id": f"{point['name']}-{index:05d}",
                    "timestamp": 0.0,
                    "input_tokens": point["input_tokens"],
                    "output_tokens": point["output_tokens"],
                }
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")

        workload = target / "workload.jsonl"
        command = [
            sys.executable,
            "-m",
            "input_bench.cli",
            "compile",
            "--adapter",
            "servegen",
            "--source",
            str(source),
            "--arrival",
            "fixed-concurrency",
            "--concurrency",
            str(point["concurrency"]),
            "--max-samples",
            str(point["requests"]),
            "--tokenizer",
            matrix["model"],
            "--seed",
            str(matrix["seed"]),
            "--output",
            str(workload),
        ]
        subprocess.run(command, cwd=BENCH_REPO, env=env, check=True)

        # Input Bench forwards arbitrary sampling fields. Force every request to
        # its requested output length so throughput points are shape-stable.
        rows = []
        for line in workload.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            row["sampling"]["ignore_eos"] = bool(matrix["ignore_eos"])
            rows.append(row)
        with workload.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

        manifest_path = target / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["workload_sha256"] = sha256_file(workload)
        manifest["benchmark_point"] = point
        manifest["sampling_overrides"] = {"ignore_eos": bool(matrix["ignore_eos"])}
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        print(f"prepared {point['name']}: {point['requests']} requests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
