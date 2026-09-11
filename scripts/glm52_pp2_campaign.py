#!/usr/bin/env python3
"""Prepare and replay a model-serving campaign from Mooncake FAST'25 traces."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from input_bench.adapters import MooncakeTraceAdapter
from input_bench.arrivals import FixedConcurrencyPolicy, TimestampTracePolicy
from input_bench.compiler import CompileOptions, compile_workload, sha256_file
from input_bench.tokenizer import load_tokenizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "scenarios" / "glm52_pp2.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def read_trace(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def eligible(row: dict[str, Any], spec: dict[str, Any], context: int, output_cap: int) -> bool:
    input_tokens = int(row["input_length"])
    output_tokens = int(row["output_length"])
    capped_output = min(output_tokens, output_cap)
    return (
        int(spec["min_input_tokens"]) <= input_tokens <= int(spec["max_input_tokens"])
        and output_tokens >= int(spec["min_output_tokens"])
        and input_tokens + capped_output <= context
    )


def random_subset(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    if len(rows) < count:
        raise RuntimeError(f"only {len(rows)} eligible trace rows; need {count}")
    indices = sorted(random.Random(seed).sample(range(len(rows)), count))
    return [rows[index] for index in indices]


def local_trace_subset(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    """Select a nearby ordered region so burst and prefix locality are retained."""
    if len(rows) < count:
        raise RuntimeError(f"only {len(rows)} eligible trace rows; need {count}")
    maximum_start = len(rows) - count
    start = random.Random(seed).randrange(maximum_start + 1)
    selected = rows[start:start + count]
    if float(selected[-1]["timestamp"]) == float(selected[0]["timestamp"]):
        for candidate in range(maximum_start + 1):
            trial = rows[candidate:candidate + count]
            if float(trial[-1]["timestamp"]) > float(trial[0]["timestamp"]):
                return trial
        raise RuntimeError("cannot find a trace subset with a non-zero timestamp span")
    return selected


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def force_ignore_eos(workload: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Set the generation-length override without depending on compiler API version."""
    rows = [json.loads(line) for line in workload.read_text().splitlines() if line.strip()]
    with workload.open("w") as handle:
        for row in rows:
            row.setdefault("sampling", {})["ignore_eos"] = True
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    manifest["sampling_overrides"] = {"ignore_eos": True}
    manifest["workload_sha256"] = sha256_file(workload)
    write_json(workload.with_name("manifest.json"), manifest)
    return manifest


def compile_one(
    *, cfg: dict[str, Any], spec: dict[str, Any], subset: Path, output: Path,
    tokenizer: Any, arrival: Any,
) -> dict[str, Any]:
    manifest = compile_workload(
        MooncakeTraceAdapter(subset, tokenizer, seed=int(cfg["seed"])),
        tokenizer,
        str(cfg["tokenizer"]),
        arrival,
        output,
        CompileOptions(
            seed=int(cfg["seed"]),
            min_input_tokens=int(spec["min_input_tokens"]),
            max_input_tokens=int(spec["max_input_tokens"]),
            min_output_tokens=int(spec["min_output_tokens"]),
            max_output_tokens=int(cfg["output_cap"]),
            default_max_output_tokens=int(cfg["output_cap"]),
            bucket_boundaries=(2048, 8192, 16384, 24576),
        ),
        source_paths=[subset],
        adapter_parameters={
            "mode": "shape_only",
            "timestamp_unit": "milliseconds",
            "upstream_source": str((Path(cfg["data_root"]) / spec["source"]).resolve()),
        },
    )
    return force_ignore_eos(output, manifest)


def prepare(config_path: Path) -> Path:
    cfg = read_json(config_path)
    data_root = Path(cfg["data_root"])
    campaign = Path(cfg["campaign_root"])
    derived = campaign / "derived-data"
    workloads = campaign / "workloads"
    derived.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(str(cfg["tokenizer"]))
    context = int(cfg["context_length"])
    output_cap = int(cfg["output_cap"])
    seed = int(cfg["seed"])
    plan: list[dict[str, Any]] = []

    for offset, spec in enumerate(cfg["fixed"]):
        source = data_root / spec["source"]
        rows = [row for row in read_trace(source) if eligible(row, spec, context, output_cap)]
        selected = random_subset(rows, int(spec["requests"]), seed + offset)
        subset = derived / f"{spec['name']}-fixed.jsonl"
        write_jsonl(subset, selected)
        workload = workloads / f"fixed-{spec['name']}" / "workload.jsonl"
        manifest = compile_one(
            cfg=cfg, spec=spec, subset=subset, output=workload,
            tokenizer=tokenizer, arrival=FixedConcurrencyPolicy(1),
        )
        for concurrency in spec["concurrency"]:
            plan.append({
                "label": f"fixed-{spec['name']}-c{concurrency}",
                "kind": "fixed",
                "source": spec["name"],
                "workload": str(workload),
                "concurrency": int(concurrency),
                "requests": manifest["counts"]["after_filter"],
                "workload_sha256": manifest["workload_sha256"],
            })

    spec = cfg["trace"]
    source = data_root / spec["source"]
    rows = [row for row in read_trace(source) if eligible(row, spec, context, output_cap)]
    selected = local_trace_subset(rows, int(spec["requests"]), seed + 100)
    subset = derived / f"{spec['name']}.jsonl"
    write_jsonl(subset, selected)
    raw_span_s = (float(selected[-1]["timestamp"]) - float(selected[0]["timestamp"])) * 1e-3
    target_duration_s = float(spec["target_duration_s"])
    time_scale = raw_span_s / target_duration_s
    workload = workloads / spec["name"] / "workload.jsonl"
    manifest = compile_one(
        cfg=cfg, spec=spec, subset=subset, output=workload, tokenizer=tokenizer,
        arrival=TimestampTracePolicy(time_scale=time_scale, max_concurrency=int(spec["max_concurrency"])),
    )
    plan.append({
        "label": spec["name"],
        "kind": "timestamp-trace",
        "source": "toolagent",
        "workload": str(workload),
        "concurrency": int(spec["max_concurrency"]),
        "requests": manifest["counts"]["after_filter"],
        "raw_span_s": raw_span_s,
        "target_duration_s": target_duration_s,
        "time_scale": time_scale,
        "workload_sha256": manifest["workload_sha256"],
    })

    source_metadata = {}
    for relative in sorted({item["source"] for item in cfg["fixed"]} | {spec["source"]}):
        path = data_root / relative
        source_metadata[str(path)] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    snapshot = dict(cfg)
    snapshot["source_files"] = source_metadata
    snapshot["plan"] = plan
    write_json(campaign / "config.snapshot.json", snapshot)
    write_json(campaign / "plan.json", plan)
    print(json.dumps({"campaign": str(campaign), "points": plan}, indent=2))
    return campaign


def no_proxy_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def endpoint_request(url: str, method: str = "GET") -> str:
    request = urllib.request.Request(url, method=method)
    if method == "POST":
        request.add_header("Content-Type", "application/json")
        request.data = b"{}"
    with no_proxy_opener().open(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def run(config_path: Path, only: set[str]) -> Path:
    cfg = read_json(config_path)
    campaign = Path(cfg["campaign_root"])
    plan_path = campaign / "plan.json"
    if not plan_path.exists():
        prepare(config_path)
    plan = read_json(plan_path)
    base_url = str(cfg["endpoint"]["base_url"]).rstrip("/")
    endpoint_request(base_url + "/health")
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    result_root = campaign / "results" / run_id
    result_root.mkdir(parents=True)
    backend = str(cfg["endpoint"]["backend"])
    info_paths = ["/server_info"] if backend == "sglang" else ["/version", "/v1/models"]
    server_info: dict[str, Any] = {}
    for path in info_paths:
        try:
            server_info[path] = json.loads(endpoint_request(base_url + path))
        except Exception as exc:
            server_info[path] = {"error": str(exc)}
    write_json(result_root / "server_info.json", server_info)
    commands: list[str] = []
    summaries: list[dict[str, Any]] = []
    for point in plan:
        if only and point["label"] not in only:
            continue
        if backend == "sglang":
            try:
                endpoint_request(base_url + "/flush_cache", method="POST")
            except Exception as exc:
                print(f"warning: cache flush failed: {exc}", file=sys.stderr)
        output_dir = result_root / point["label"]
        cmd = [
            sys.executable, "-m", "input_bench.cli", "run",
            "--workload", point["workload"],
            "--backend", str(cfg["endpoint"]["backend"]),
            "--base-url", base_url,
            "--model", str(cfg["endpoint"]["model"]),
            "--tokenizer", str(cfg["tokenizer"]),
            "--output-dir", str(output_dir),
            "--max-concurrency", str(point["concurrency"]),
            "--pool-size", str(max(32, point["concurrency"])),
            "--timeout", "1200",
            "--ttft-slo-ms", str(cfg["slo_ms"]["ttft"]),
            "--tpot-slo-ms", str(cfg["slo_ms"]["tpot"]),
            "--e2e-slo-ms", str(cfg["slo_ms"]["e2e"]),
        ]
        if backend == "vllm":
            # The normal vLLM API does not expose the dev-only reset endpoint.
            # A unique per-point salt prevents cross-point cache hits while
            # retaining prefix reuse among requests inside the same point.
            cmd.extend(["--cache-salt", f"input-bench:{run_id}:{point['label']}"])
        commands.append(" ".join(cmd))
        print(f"\n=== {point['label']} ===", flush=True)
        subprocess.run(cmd, check=True, cwd=ROOT)
        summary = read_json(output_dir / "summary.json")
        summaries.append({"label": point["label"], "point": point, "summary": summary})
    (result_root / "commands.txt").write_text("\n".join(commands) + "\n")
    write_json(result_root / "campaign-summary.json", summaries)
    print(f"\nresults: {result_root}")
    return result_root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "all"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--only", action="append", default=[], help="run only the named point")
    args = parser.parse_args()
    if args.action in {"prepare", "all"}:
        prepare(args.config)
    if args.action in {"run", "all"}:
        run(args.config, set(args.only))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
