#!/usr/bin/env python3
"""Prepare and run a deterministic SWE-bench request-latency benchmark."""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from input_bench.benchmark import aggregate_request_benchmark
from input_bench.cli import main as input_bench_main
from input_bench.compiler import load_workload, sha256_file


DEFAULT_CONFIG = ROOT / "scenarios" / "swebench_request_bench.example.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def resolve_path(value: str, config_path: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def tokenizer_reference(value: str, config_path: Path) -> str:
    """Resolve local tokenizer paths while preserving Hugging Face IDs."""
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return str(candidate.resolve())
    relative = (config_path.parent / candidate).resolve()
    return str(relative) if relative.exists() else value


def positive_int(cfg: dict[str, Any], key: str, default: int) -> int:
    value = int(cfg.get(key, default))
    if value < 1:
        raise ValueError(f"{key} must be at least 1")
    return value


def paths(cfg: dict[str, Any], config_path: Path) -> tuple[Path, Path, Path]:
    source = resolve_path(str(cfg["source"]), config_path)
    output_root = resolve_path(str(cfg["output_root"]), config_path)
    workload = output_root / "workload" / "workload.jsonl"
    return source, output_root, workload


@contextmanager
def isolated_input_bench_environment():
    """Prevent unrelated portable-config variables from changing this bench."""
    names = [
        "INPUT_BENCH_CONFIG",
        "INPUT_BENCH_DATA_ROOT",
        "INPUT_BENCH_TOKENIZER",
        "INPUT_BENCH_WORKLOAD",
        "INPUT_BENCH_RESULTS_ROOT",
        "INPUT_BENCH_BACKEND",
        "INPUT_BENCH_BASE_URL",
        "INPUT_BENCH_MODEL",
    ]
    saved = {name: os.environ[name] for name in names if name in os.environ}
    try:
        for name in names:
            os.environ.pop(name, None)
        yield
    finally:
        for name in names:
            os.environ.pop(name, None)
        os.environ.update(saved)


def checked_input_bench(argv: list[str]) -> None:
    with isolated_input_bench_environment():
        status = input_bench_main(argv)
    if status != 0:
        raise RuntimeError(f"input-bench exited with status {status}: {' '.join(argv)}")


def validate_existing_workload(workload: Path) -> None:
    checked_input_bench(["validate", str(workload)])


def prepare(config_path: Path, *, force: bool = False) -> Path:
    cfg = read_json(config_path)
    source, _, workload = paths(cfg, config_path)
    concurrency = positive_int(cfg, "concurrency", 1)
    examples = positive_int(cfg, "examples", 10)
    max_output_tokens = positive_int(cfg, "max_output_tokens", 128)
    if workload.exists() and not force:
        validate_existing_workload(workload)
        print(f"reusing frozen workload: {workload}")
        return workload
    if not source.is_file():
        raise FileNotFoundError(f"SWE-bench source not found: {source}")

    argv = [
        "compile",
        "--adapter", "swebench-verified",
        "--source", str(source),
        "--arrival", "fixed-concurrency",
        "--concurrency", str(concurrency),
        "--seed", str(int(cfg.get("seed", 1))),
        "--tokenizer", tokenizer_reference(str(cfg["tokenizer"]), config_path),
        "--max-samples", str(examples),
        "--max-output-tokens", str(max_output_tokens),
        "--default-max-output-tokens", str(max_output_tokens),
        "--output", str(workload),
    ]
    if cfg.get("include_repo", True):
        argv.append("--include-repo")
    if cfg.get("include_base_commit", True):
        argv.append("--include-base-commit")
    if cfg.get("include_hints", False):
        argv.append("--include-hints")
    checked_input_bench(argv)
    print(f"prepared frozen workload: {workload}")
    return workload


def redacted_config(cfg: dict[str, Any]) -> dict[str, Any]:
    snapshot = json.loads(json.dumps(cfg))
    headers = snapshot.get("endpoint", {}).get("headers", {})
    for key in list(headers):
        if key.lower() in {"authorization", "x-api-key", "api-key"}:
            headers[key] = "<redacted>"
    return snapshot


def run(config_path: Path, *, run_id: str | None = None) -> Path:
    cfg = read_json(config_path)
    _, output_root, workload = paths(cfg, config_path)
    if not workload.is_file():
        raise FileNotFoundError(f"frozen workload not found; run prepare first: {workload}")
    validate_existing_workload(workload)

    endpoint = cfg["endpoint"]
    tokenizer = tokenizer_reference(str(cfg["tokenizer"]), config_path)
    repeats = positive_int(cfg, "repeats", 3)
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    result_root = output_root / "results" / run_id
    if result_root.exists():
        raise FileExistsError(f"result run already exists: {result_root}")
    result_root.mkdir(parents=True)

    requests = load_workload(workload)
    if not requests:
        raise ValueError(f"frozen workload contains no requests: {workload}")
    concurrency = positive_int(cfg, "concurrency", 1)
    pool_size = positive_int(cfg, "pool_size", max(16, concurrency))
    summaries: list[dict[str, Any]] = []
    for repeat in range(1, repeats + 1):
        output_dir = result_root / f"repeat-{repeat:02d}"
        argv = [
            "run",
            "--workload", str(workload),
            "--backend", str(endpoint.get("backend", "vllm")),
            "--base-url", str(endpoint["base_url"]),
            "--model", str(endpoint["model"]),
            "--tokenizer", str(tokenizer),
            "--output-dir", str(output_dir),
            "--max-concurrency", str(concurrency),
            "--pool-size", str(pool_size),
            "--timeout", str(float(cfg.get("timeout_s", 900))),
            "--response-max-chars", str(int(cfg.get("response_max_chars", 10000))),
            "--retries", str(int(cfg.get("retries", 0))),
        ]
        argv.append("--stream" if cfg.get("stream", True) else "--no-stream")
        argv.append("--tls-verify" if endpoint.get("tls_verify", True) else "--no-tls-verify")
        argv.extend(["--api-key-env", str(endpoint.get("api_key_env", "OPENAI_API_KEY"))])
        for key, value in endpoint.get("headers", {}).items():
            argv.extend(["--header", f"{key}={value}"])
        if int(cfg.get("warmup", 0)):
            argv.extend(["--warmup", str(int(cfg["warmup"]))])

        print(f"running repeat {repeat}/{repeats}: {output_dir}", flush=True)
        checked_input_bench(argv)
        summaries.append(read_json(output_dir / "summary.json"))

    report = aggregate_request_benchmark(summaries)
    report.update({
        "benchmark": "swebench-request-total-time",
        "target_label": str(cfg.get("target_label", "ours")),
        "backend": str(endpoint.get("backend", "vllm")),
        "base_url": str(endpoint["base_url"]),
        "model": str(endpoint["model"]),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "workload": str(workload),
        "workload_sha256": sha256_file(workload),
        "request_count": len(requests),
        "request_ids": [request.request_id for request in requests],
        "result_directories": [str(result_root / f"repeat-{i:02d}") for i in range(1, repeats + 1)],
    })
    write_json(result_root / "benchmark-summary.json", report)
    write_json(result_root / "config.snapshot.json", redacted_config(cfg))
    print(json.dumps(report["total_duration_s"], indent=2))
    print(f"benchmark results: {result_root}")
    return result_root


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("action", choices=("prepare", "run", "all"))
    value.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    value.add_argument("--force", action="store_true", help="rebuild an existing frozen workload")
    value.add_argument("--run-id", help="explicit result directory name for run/all")
    return value


def main() -> int:
    args = parser().parse_args()
    config_path = args.config.expanduser().resolve()
    try:
        if args.action in {"prepare", "all"}:
            prepare(config_path, force=args.force)
        if args.action in {"run", "all"}:
            run(config_path, run_id=args.run_id)
        return 0
    except (KeyError, ValueError, RuntimeError, OSError) as exc:
        print(f"swebench-request-bench: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
