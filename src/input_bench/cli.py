from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from input_bench.adapters import ADAPTERS
from input_bench.arrivals import FixedConcurrencyPolicy, PoissonPolicy, TimestampTracePolicy
from input_bench.compiler import CompileOptions, compile_workload, load_workload, sha256_file
from input_bench.config import (SUPPORTED_BACKENDS, load_runtime_config,
                                portable_tokenizer_reference,
                                resolve_tokenizer_reference)
from input_bench.metrics import summarize_events
from input_bench.recorder import read_events, write_results
from input_bench.sender import BenchmarkCancelled, BenchmarkSender, SenderConfig
from input_bench.tokenizer import load_tokenizer

SOURCE_LAYOUT = {
 "sharegpt": "chat/sharegpt-v3/ShareGPT_V3_unfiltered_cleaned_split.json",
 "swe-agent": "coding/swe-agent-trajectories/data",
 "swebench-verified": "coding/swe-bench-verified/data/test-00000-of-00001.parquet",
 "thoughtworks": "coding/thoughtworks-agentic-trajectories/sessions.parquet",
 "arxiv": "summarization/arxiv/document",
 "longbench": "summarization/longbench/data.zip",
 "mooncake": "traces/mooncake/conversation_trace.jsonl",
}


def _config_options(parser: argparse.ArgumentParser, *, data: bool = False) -> None:
    parser.add_argument("--config", help="portable JSON runtime config (or INPUT_BENCH_CONFIG)")
    if data:
        parser.add_argument("--data-root", help="dataset root; overrides config and INPUT_BENCH_DATA_ROOT")


def _common_adapter(parser: argparse.ArgumentParser) -> None:
    _config_options(parser, data=True)
    parser.add_argument("--adapter", choices=sorted(ADAPTERS), required=True)
    parser.add_argument("--source", help="dataset file/directory (defaults below configured data root)")
    parser.add_argument("--mode", choices=["single", "independent_turn", "session_replay", "shape_only"])
    parser.add_argument("--subset"); parser.add_argument("--split")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="input-bench", description="Compile and replay deterministic LLM workloads")
    sub = parser.add_subparsers(dest="command", required=True)
    datasets = sub.add_parser("datasets", help="list supported adapters and local dataset availability")
    _config_options(datasets, data=True)
    inspect = sub.add_parser("inspect", help="show a few semantic adapter records"); _common_adapter(inspect)
    inspect.add_argument("--limit", type=int, default=3); inspect.add_argument("--tokenizer")
    compile_p = sub.add_parser("compile", help="compile an immutable workload JSONL"); _common_adapter(compile_p)
    compile_p.add_argument("--tokenizer"); compile_p.add_argument("--tokenizer-revision")
    compile_p.add_argument("--output"); compile_p.add_argument("--seed", type=int, default=1)
    compile_p.add_argument("--arrival", choices=["fixed-concurrency", "poisson", "timestamp-trace"], required=True)
    compile_p.add_argument("--concurrency", type=int, default=1); compile_p.add_argument("--request-rate", type=float)
    compile_p.add_argument("--duration", type=float); compile_p.add_argument("--time-scale", type=float, default=1.0)
    compile_p.add_argument("--start-offset", type=float, default=0.0); compile_p.add_argument("--window-start", type=float)
    compile_p.add_argument("--window-end", type=float); compile_p.add_argument("--max-concurrency", type=int)
    compile_p.add_argument("--max-samples", type=int); compile_p.add_argument("--min-input-tokens", type=int)
    compile_p.add_argument("--max-input-tokens", type=int); compile_p.add_argument("--min-output-tokens", type=int)
    compile_p.add_argument("--max-output-tokens", type=int); compile_p.add_argument("--default-max-output-tokens", type=int, default=256)
    compile_p.add_argument("--bucket-boundary", action="append", type=int, default=[])
    compile_p.add_argument("--timestamp-unit", choices=["seconds", "milliseconds", "microseconds"], default="milliseconds")
    compile_p.add_argument("--include-repo", action="store_true"); compile_p.add_argument("--include-base-commit", action="store_true")
    compile_p.add_argument("--include-hints", action="store_true"); compile_p.add_argument("--max-turns", type=int)
    validate = sub.add_parser("validate", help="validate workload schema/order/hash"); validate.add_argument("workload")
    run = sub.add_parser("run", help="replay a compiled workload against an OpenAI-compatible endpoint")
    _config_options(run)
    run.add_argument("--workload"); run.add_argument("--backend", choices=SUPPORTED_BACKENDS)
    run.add_argument("--base-url"); run.add_argument("--model")
    run.add_argument("--tokenizer"); run.add_argument("--tokenizer-revision")
    run.add_argument("--output-dir"); run.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--api-key-env"); run.add_argument("--header", action="append")
    run.add_argument("--timeout", type=float, default=300); run.add_argument("--pool-size", type=int, default=100)
    run.add_argument("--max-concurrency", type=int); run.add_argument("--tls-verify", action=argparse.BooleanOptionalAction, default=None)
    run.add_argument("--response-max-chars", type=int, default=10000); run.add_argument("--retries", type=int, default=0)
    run.add_argument("--warmup", type=int, default=0); run.add_argument("--ttft-slo-ms", type=float)
    run.add_argument("--e2e-slo-ms", type=float); run.add_argument("--tpot-slo-ms", type=float)
    summarize = sub.add_parser("summarize", help="recompute summary from events JSONL")
    summarize.add_argument("events"); summarize.add_argument("--output")
    return parser


def _default_source(adapter: str, data_root: str | Path) -> Path | None:
    relative = SOURCE_LAYOUT.get(adapter)
    return Path(data_root) / relative if relative else None


def _require(value: Any, option: str) -> Any:
    if value is None or value == "":
        raise ValueError(f"{option} is required (pass it on the CLI or set it in --config)")
    return value


def _apply_runtime_config(args: argparse.Namespace) -> None:
    if not hasattr(args, "config"):
        return
    cfg = load_runtime_config(args.config)
    args.runtime_config = cfg
    if hasattr(args, "data_root"):
        args.data_root = Path(args.data_root).expanduser().resolve() if args.data_root else cfg.data_root
    if args.command == "inspect":
        args.tokenizer = args.tokenizer or cfg.tokenizer or "whitespace"
    elif args.command == "compile":
        args.tokenizer = _require(args.tokenizer or cfg.tokenizer, "--tokenizer")
        args.tokenizer_revision = args.tokenizer_revision or cfg.tokenizer_revision
        args.output = str(_require(args.output or cfg.workload, "--output"))
    elif args.command == "run":
        args.workload = str(_require(args.workload or cfg.workload, "--workload"))
        args.backend = args.backend or cfg.backend
        args.base_url = args.base_url or cfg.base_url
        args.model = _require(args.model or cfg.model, "--model")
        args.tokenizer = args.tokenizer or cfg.tokenizer
        args.tokenizer_revision = args.tokenizer_revision or cfg.tokenizer_revision
        if not args.output_dir:
            root = _require(cfg.results_root, "--output-dir or config results.root")
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            args.output_dir = str(Path(root) / f"{stamp}-{args.backend}")
        args.api_key_env = args.api_key_env or cfg.api_key_env
        config_headers = [f"{key}={value}" for key, value in cfg.headers.items()]
        args.header = config_headers + (args.header or [])
        args.tls_verify = cfg.tls_verify if args.tls_verify is None else args.tls_verify


def _adapter(args: argparse.Namespace, tokenizer: Any) -> tuple[Any, Path, dict[str, Any]]:
    raw_source = args.source or _default_source(args.adapter, args.data_root)
    if not raw_source: raise ValueError(f"--source is required for {args.adapter}")
    source = Path(raw_source)
    kwargs = {"source": source, "tokenizer": tokenizer, "seed": getattr(args, "seed", 1)}
    for key in ["mode", "subset", "split", "timestamp_unit", "max_turns", "include_repo", "include_base_commit", "include_hints"]:
        if hasattr(args, key) and getattr(args, key) is not None: kwargs[key] = getattr(args, key)
    adapter = ADAPTERS[args.adapter](**kwargs)
    params = {k: v for k, v in kwargs.items() if k not in {"source", "tokenizer"}}
    return adapter, source, params


def _source_paths(adapter: Any, source: Path) -> list[Path]:
    if hasattr(adapter, "paths"): return list(adapter.paths)
    return [source] if source.is_file() else []


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    requests = load_workload(args.workload)
    manifest_path = Path(args.workload).with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    tokenizer_id = args.tokenizer or manifest.get("tokenizer", {}).get("id")
    if not tokenizer_id: raise ValueError("--tokenizer is required when manifest is unavailable")
    tokenizer_id = resolve_tokenizer_reference(str(tokenizer_id), manifest_path.parent)
    revision = args.tokenizer_revision if args.tokenizer else manifest.get("tokenizer", {}).get("revision")
    tokenizer = load_tokenizer(tokenizer_id, revision)
    headers = dict(value.split("=", 1) for value in args.header)
    default_concurrency = manifest.get("arrival_parameters", {}).get("concurrency") or manifest.get("arrival_parameters", {}).get("max_concurrency") or 100
    config = SenderConfig(args.base_url, args.model, args.stream, os.getenv(args.api_key_env), headers,
        args.timeout, args.pool_size, args.max_concurrency or default_concurrency, args.tls_verify,
        args.response_max_chars, args.retries, args.backend)
    sender = BenchmarkSender(config, tokenizer)
    report_config = asdict(config)
    report_config["api_key"] = "<set>" if config.api_key else None
    report_config["headers"] = {k: ("<redacted>" if k.lower() in {"authorization", "x-api-key", "api-key"} else v)
                                for k, v in config.headers.items()}
    if args.warmup:
        warm = [replace(request, scheduled_offset_s=None) for request in requests[:args.warmup]]
        # Warmup uses immediate dispatch and is intentionally absent from formal events/metrics.
        await sender.run(warm)
    try:
        events, bounds = await sender.run(requests)
    except BenchmarkCancelled as exc:
        events, bounds = exc.events, exc.bounds
        slos = {"ttft_ms": args.ttft_slo_ms, "e2e_ms": args.e2e_slo_ms, "tpot_ms": args.tpot_slo_ms}
        write_results(args.output_dir, events, bounds,
                      {"workload": str(Path(args.workload).resolve()), "sender": report_config,
                       "warmup": args.warmup, "interrupted": True}, slos)
        raise
    slos = {"ttft_ms": args.ttft_slo_ms, "e2e_ms": args.e2e_slo_ms, "tpot_ms": args.tpot_slo_ms}
    return write_results(args.output_dir, events, bounds,
                         {"workload": str(Path(args.workload).resolve()), "sender": report_config, "warmup": args.warmup}, slos)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _apply_runtime_config(args)
        if args.command == "datasets":
            for name in sorted(ADAPTERS):
                source = _default_source(name, args.data_root)
                print(f"{name:20} {source or '(explicit source required)'} {'OK' if source and source.exists() else ''}")
        elif args.command == "inspect":
            tokenizer = load_tokenizer(args.tokenizer); adapter, _, _ = _adapter(args, tokenizer)
            for sample in adapter.iter_samples(args.limit):
                value = asdict(sample); value["reference_output"] = (value.get("reference_output") or "")[:200]
                print(json.dumps(value, ensure_ascii=False, default=str))
            print(json.dumps({"skip_reasons": adapter.skip_reasons}, default=dict), file=sys.stderr)
        elif args.command == "compile":
            tokenizer = load_tokenizer(args.tokenizer, args.tokenizer_revision); adapter, source, params = _adapter(args, tokenizer)
            if args.arrival == "fixed-concurrency": arrival = FixedConcurrencyPolicy(args.concurrency)
            elif args.arrival == "poisson":
                if args.request_rate is None: raise ValueError("--request-rate is required for poisson")
                arrival = PoissonPolicy(args.request_rate, args.seed, args.duration, args.max_concurrency)
            else: arrival = TimestampTracePolicy(args.time_scale, args.start_offset, args.window_start, args.window_end, args.max_concurrency)
            opts = CompileOptions(args.seed, args.max_samples, args.min_input_tokens, args.max_input_tokens,
                args.min_output_tokens, args.max_output_tokens, args.default_max_output_tokens, tuple(sorted(args.bucket_boundary)))
            tokenizer_reference = portable_tokenizer_reference(args.tokenizer, Path(args.output))
            manifest = compile_workload(adapter, tokenizer, tokenizer_reference, arrival, args.output, opts,
                tokenizer_revision=args.tokenizer_revision, source_paths=_source_paths(adapter, source), adapter_parameters=params)
            print(json.dumps(manifest["counts"], indent=2))
        elif args.command == "validate":
            requests = load_workload(args.workload); manifest_path = Path(args.workload).with_name("manifest.json")
            if manifest_path.exists():
                expected = json.loads(manifest_path.read_text()).get("workload_sha256")
                if expected and sha256_file(Path(args.workload)) != expected: raise ValueError("workload SHA-256 differs from manifest")
            print(f"valid: {len(requests)} requests")
        elif args.command == "run": print(json.dumps(asyncio.run(_run(args)), indent=2))
        elif args.command == "summarize":
            summary = summarize_events(read_events(args.events)); text = json.dumps(summary, indent=2, sort_keys=True)
            if args.output: Path(args.output).write_text(text + "\n")
            else: print(text)
        return 0
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"input-bench: error: {exc}", file=sys.stderr); return 2


if __name__ == "__main__": raise SystemExit(main())
