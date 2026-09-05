#!/usr/bin/env python3
"""Prepare and replay the deterministic DSV4 CVM/bare-metal campaign.

Preparation is intentionally separate from replay.  Run ``prepare`` once while
the source datasets are mounted, then reuse the frozen workloads byte-for-byte
for every environment under comparison.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from input_bench.adapters import (
    ArxivSummarizationAdapter,
    MooncakeTraceAdapter,
    ShareGPTAdapter,
    ThoughtworksAgenticAdapter,
)
from input_bench.adapters.base import iter_json_array
from input_bench.arrivals import FixedConcurrencyPolicy, TimestampTracePolicy
from input_bench.backends import get_backend, parse_server_metrics
from input_bench.compiler import CompileOptions, compile_workload, load_workload, sha256_file
from input_bench.config import (SUPPORTED_BACKENDS, load_runtime_config,
                                portable_tokenizer_reference)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "scenarios" / "dsv4_large_scale.json"
DEFAULT_CAMPAIGN = ROOT / "campaigns" / "dsv4-large-scale"
DATASET_LAYOUT = {
    "chat": "chat/sharegpt-v3/ShareGPT_V3_unfiltered_cleaned_split.json",
    "agent_coding": "coding/thoughtworks-agentic-trajectories/sessions.parquet",
    "summarization": "summarization/arxiv/document",
    "mooncake": "traces/mooncake/conversation_trace.jsonl",
}


def dataset_paths(data_root: Path) -> dict[str, Path]:
    return {key: data_root / relative for key, relative in DATASET_LAYOUT.items()}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def resolve_from_root(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_encoding(path: Path):
    spec = importlib.util.spec_from_file_location("input_bench_dsv4_encoding", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import DeepSeek-V4 encoding from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DeepSeekV4Tokenizer:
    """HF tokenizer plus the exact vLLM 0.25 DeepSeek-V4 message renderer."""

    def __init__(self, tokenizer_dir: Path, encoding_path: Path):
        self.base = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=False)
        self.encoding = load_encoding(encoding_path)
        self.name_or_path = str(tokenizer_dir.resolve())
        self.chat_template = f"vllm-0.25:{sha256_file(encoding_path)}"

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def encode(self, text: str, add_special_tokens: bool = False, **kwargs: Any) -> list[int]:
        return self.base.encode(text, add_special_tokens=add_special_tokens, **kwargs)

    def decode(self, ids: list[int], skip_special_tokens: bool = False, **kwargs: Any) -> str:
        return self.base.decode(ids, skip_special_tokens=skip_special_tokens, **kwargs)

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **_: Any,
    ) -> str | list[int]:
        # vLLM's DSV4 renderer decides the generation suffix from the final role.
        # All benchmark samples end immediately before an assistant response.
        if not add_generation_prompt:
            raise ValueError("DSV4 benchmark tokenizer requires add_generation_prompt=True")
        rendered = self.encoding.encode_messages(
            messages, thinking_mode="chat", drop_thinking=True
        )
        return self.encode(rendered, add_special_tokens=False) if tokenize else rendered


class ReservoirAdapter:
    """Deterministically pre-sample semantic records before costly tokenization."""

    def __init__(self, adapter: Any, count: int, seed: int):
        self.adapter = adapter
        self.count = count
        self.seed = seed
        self.name = adapter.name
        self.skip_reasons = adapter.skip_reasons
        self.manifest_metadata = {
            **adapter.manifest_metadata,
            "pre_tokenization_sampling": {"method": "reservoir", "count": count, "seed": seed},
        }

    def iter_samples(self, limit: int | None = None) -> Iterable[Any]:
        rng = random.Random(self.seed)
        selected: list[tuple[int, Any]] = []
        for seen, sample in enumerate(self.adapter.iter_samples(), 1):
            value = (seen, sample)
            if seen <= self.count:
                selected.append(value)
            else:
                slot = rng.randrange(seen)
                if slot < self.count:
                    selected[slot] = value
        selected.sort(key=lambda value: value[0])
        rows = [sample for _, sample in selected]
        yield from rows if limit is None else rows[:limit]


def reservoir_json_array(source: Path, count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    sample: list[dict[str, Any]] = []
    for seen, (_, row) in enumerate(iter_json_array(source), 1):
        if seen <= count:
            sample.append(row)
        else:
            slot = rng.randrange(seen)
            if slot < count:
                sample[slot] = row
    return sample


def sample_parquet(paths: list[Path], count: int, seed: int) -> pa.Table:
    rng = random.Random(seed)
    sample: list[dict[str, Any]] = []
    seen = 0
    # Loading the full Thoughtworks string columns can exceed Arrow's 2 GiB
    # offset limit.  Reservoir-sample record batches instead and only
    # materialize the selected rows.
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=256):
            for row in batch.to_pylist():
                seen += 1
                if seen <= count:
                    sample.append(row)
                else:
                    slot = rng.randrange(seen)
                    if slot < count:
                        sample[slot] = row
    return pa.Table.from_pylist(sample)


def sample_jsonl_preserve_order(source: Path, count: int, seed: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    indices = sorted(random.Random(seed).sample(range(len(rows)), min(count, len(rows))))
    return [rows[index] for index in indices]


def prepare_subsets(
    cfg: dict[str, Any], campaign: Path, datasets: dict[str, Path]
) -> dict[str, Path]:
    seed = int(cfg["seed"])
    counts = cfg["derived_records"]
    out = campaign / "derived-data"
    out.mkdir(parents=True, exist_ok=True)

    chat = out / "sharegpt-random.json"
    chat.write_text(json.dumps(
        reservoir_json_array(datasets["chat"], int(counts["chat"]), seed),
        ensure_ascii=False,
    ))

    coding = out / "thoughtworks-random.parquet"
    pq.write_table(sample_parquet([datasets["agent_coding"]], int(counts["agent_coding"]), seed + 1), coding)

    summary = out / "arxiv-random.parquet"
    arxiv_paths = sorted(datasets["summarization"].glob("*.parquet"))
    pq.write_table(sample_parquet(arxiv_paths, int(counts["summarization"]), seed + 2), summary)

    mooncake = out / "mooncake-conversation-random.jsonl"
    trace_rows = sample_jsonl_preserve_order(datasets["mooncake"], int(counts["mooncake"]), seed + 3)
    mooncake.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in trace_rows))
    return {"chat": chat, "agent_coding": coding, "summarization": summary, "mooncake": mooncake}


def compile_bases(
    cfg: dict[str, Any], campaign: Path, subsets: dict[str, Path], tokenizer: DeepSeekV4Tokenizer
) -> dict[str, Path]:
    seed = int(cfg["seed"])
    limits = cfg["length_limits"]
    counts = cfg["base_requests"]
    max_output = int(cfg["max_output_tokens"])
    base_root = campaign / "base-workloads"
    specs = {
        "chat": ShareGPTAdapter(subsets["chat"], mode="independent_turn"),
        "agent_coding": ThoughtworksAgenticAdapter(subsets["agent_coding"], mode="independent_turn"),
        "summarization": ArxivSummarizationAdapter(subsets["summarization"]),
    }
    paths: dict[str, Path] = {}
    for offset, (workload_class, adapter) in enumerate(specs.items()):
        adapter = ReservoirAdapter(
            adapter,
            count=max(int(counts[workload_class]) * 3, int(counts[workload_class])),
            seed=seed + 100 + offset,
        )
        target = base_root / workload_class / "workload.jsonl"
        low, high = map(int, limits[workload_class])
        compile_workload(
            adapter,
            tokenizer,
            tokenizer.name_or_path,
            FixedConcurrencyPolicy(1),
            target,
            CompileOptions(
                seed=seed + offset,
                max_samples=int(counts[workload_class]),
                min_input_tokens=low,
                max_input_tokens=high,
                max_output_tokens=max_output,
                default_max_output_tokens=max_output,
                bucket_boundaries=(512, 2048, 8192, 16384),
            ),
            source_paths=[subsets[workload_class]],
            adapter_parameters={"mode": "independent_turn" if workload_class != "summarization" else "single"},
        )
        paths[workload_class] = target

    trace_cfg = cfg["trace"]
    trace_target = base_root / "mooncake" / "workload.jsonl"
    trace_rows = [json.loads(line) for line in subsets["mooncake"].read_text().splitlines() if line.strip()]
    timestamps = [float(row["timestamp"]) * 1e-3 for row in trace_rows]
    raw_span = max(timestamps) - min(timestamps)
    time_scale = max(raw_span / float(trace_cfg["target_duration_s"]), 1.0)
    low, high = map(int, limits["mooncake"])
    compile_workload(
        MooncakeTraceAdapter(subsets["mooncake"], tokenizer, seed=seed),
        tokenizer,
        tokenizer.name_or_path,
        TimestampTracePolicy(time_scale=time_scale, max_concurrency=int(trace_cfg["max_concurrency"])),
        trace_target,
        CompileOptions(
            seed=seed + 3,
            max_samples=int(trace_cfg["requests"]),
            min_input_tokens=low,
            max_input_tokens=high,
            max_output_tokens=max_output,
            default_max_output_tokens=max_output,
            bucket_boundaries=(2048, 8192, 16384, 30000),
        ),
        source_paths=[subsets["mooncake"]],
        adapter_parameters={"mode": "shape_only", "timestamp_unit": "milliseconds"},
    )
    paths["mooncake"] = trace_target
    return paths


def nonce_request(request, point: str, sequence: int, tokenizer: DeepSeekV4Tokenizer):
    marker = f"[IBENCH-{point}-{sequence:04d}]\n"
    messages = None
    prompt = None
    if request.messages is not None:
        messages = [dict(message) for message in request.messages]
        if not messages:
            raise ValueError(f"empty messages for {request.request_id}")
        first = dict(messages[0])
        content = first.get("content", "")
        first["content"] = marker + (content if isinstance(content, str) else json.dumps(content, ensure_ascii=False))
        messages[0] = first
        input_tokens = len(tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        ))
    else:
        prompt = marker + (request.prompt or "")
        input_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    metadata = dict(request.metadata)
    metadata.update({"campaign_point": point, "base_request_id": request.request_id, "cold_prefix_nonce": marker.strip()})
    return replace(
        request,
        request_id=f"{point}:{request.request_id}",
        sequence_no=sequence,
        parent_request_id=None,
        scheduled_offset_s=None,
        messages=messages,
        prompt=prompt,
        input_tokens=input_tokens,
        metadata=metadata,
    )


def poisson_offsets(count: int, rate: float, duration: float, seed: int) -> list[float]:
    rng = random.Random(seed)
    values: list[float] = []
    now = 0.0
    for index in range(count):
        if index:
            now += rng.expovariate(rate)
        if now > duration:
            break
        values.append(now)
    return values


def stats(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "mean": None, "max": None}
    return {"min": min(values), "mean": mean(values), "max": max(values)}


def write_point(
    campaign: Path,
    label: str,
    requests: list[Any],
    *,
    kind: str,
    workload_class: str,
    concurrency: int,
    parameters: dict[str, Any],
    tokenizer: DeepSeekV4Tokenizer,
) -> dict[str, Any]:
    point_dir = campaign / "workloads" / label
    point_dir.mkdir(parents=True, exist_ok=True)
    workload = point_dir / "workload.jsonl"
    with workload.open("w") as handle:
        for request in requests:
            handle.write(json.dumps(request.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    manifest = {
        "schema_version": "1.0",
        "campaign_schema": "dsv4-large-scale-v1",
        "label": label,
        "kind": kind,
        "workload_class": workload_class,
        "concurrency": concurrency,
        "parameters": parameters,
        "counts": {"after_filter": len(requests)},
        "token_statistics": {
            "input": stats([request.input_tokens for request in requests]),
            "output_cap": stats([request.max_output_tokens for request in requests]),
        },
        "tokenizer": {
            "id": portable_tokenizer_reference(tokenizer.name_or_path, workload),
            "encoding_sha256": tokenizer.chat_template.split(":", 1)[1],
        },
        "workload_sha256": sha256_file(workload),
    }
    write_json(point_dir / "manifest.json", manifest)
    return {
        "label": label,
        "kind": kind,
        "workload_class": workload_class,
        "workload": str(workload.relative_to(campaign)),
        "requests": len(requests),
        "concurrency": concurrency,
        "parameters": parameters,
        "workload_sha256": manifest["workload_sha256"],
    }


def prepare_points(
    cfg: dict[str, Any], campaign: Path, bases: dict[str, Path], tokenizer: DeepSeekV4Tokenizer
) -> list[dict[str, Any]]:
    seed = int(cfg["seed"])
    fixed = cfg["fixed_concurrency"]
    poisson = cfg["poisson"]
    plan: list[dict[str, Any]] = []
    base_requests = {key: load_workload(path) for key, path in bases.items()}
    semantic_classes = ("chat", "agent_coding", "summarization")

    # A separate preflight warms the representative kernels but is excluded from results.
    warm: list[Any] = []
    for workload_class in semantic_classes:
        source = base_requests[workload_class]
        ordered = sorted(source, key=lambda request: request.input_tokens)
        chosen = [ordered[round(index * (len(ordered) - 1) / 7)] for index in range(8)]
        warm.extend(nonce_request(request, "preflight", len(warm), tokenizer) for request in chosen)
    plan.append(write_point(
        campaign, "preflight", warm, kind="preflight", workload_class="mixed",
        concurrency=8, parameters={}, tokenizer=tokenizer,
    ))

    per_point = int(fixed["requests_per_point"])
    for repeat in range(1, int(fixed["repeats"]) + 1):
        for workload_class in semantic_classes:
            source = base_requests[workload_class]
            if len(source) < per_point:
                raise RuntimeError(f"{workload_class} has only {len(source)} base requests")
            chosen = source[:per_point]
            for concurrency in map(int, fixed["concurrency"]):
                label = f"fixed-{workload_class}-c{concurrency}-r{repeat}"
                requests = [nonce_request(request, label, index, tokenizer) for index, request in enumerate(chosen)]
                plan.append(write_point(
                    campaign, label, requests, kind="fixed", workload_class=workload_class,
                    concurrency=concurrency, parameters={"repeat": repeat}, tokenizer=tokenizer,
                ))

    duration = float(poisson["duration_s"])
    max_concurrency = int(poisson["max_concurrency"])
    for workload_class in semantic_classes:
        source = base_requests[workload_class]
        for rate_index, rate_value in enumerate(poisson["rates"]):
            rate = float(rate_value)
            offsets = poisson_offsets(len(source), rate, duration, seed + rate_index)
            label_rate = str(rate).replace(".", "p")
            label = f"poisson-{workload_class}-rps{label_rate}"
            requests = []
            for index, (request, scheduled) in enumerate(zip(source, offsets)):
                value = nonce_request(request, label, index, tokenizer)
                requests.append(replace(value, scheduled_offset_s=scheduled))
            plan.append(write_point(
                campaign, label, requests, kind="poisson", workload_class=workload_class,
                concurrency=max_concurrency,
                parameters={"rate_rps": rate, "duration_s": duration}, tokenizer=tokenizer,
            ))

    trace_cfg = cfg["trace"]
    trace_label = "trace-mooncake-conversation"
    trace_requests = []
    for index, request in enumerate(base_requests["mooncake"]):
        metadata = dict(request.metadata)
        metadata["campaign_point"] = trace_label
        trace_requests.append(replace(
            request,
            request_id=f"{trace_label}:{request.request_id}",
            sequence_no=index,
            metadata=metadata,
        ))
    plan.append(write_point(
        campaign, trace_label, trace_requests, kind="trace", workload_class="long_context",
        concurrency=int(trace_cfg["max_concurrency"]),
        parameters={"target_duration_s": float(trace_cfg["target_duration_s"]), "source": "Mooncake conversation"},
        tokenizer=tokenizer,
    ))

    profile_cfg = cfg["profile"]
    profile: list[Any] = []
    for workload_class in semantic_classes:
        source = base_requests[workload_class]
        for request in source[: int(profile_cfg["requests_per_class"])]:
            profile.append(nonce_request(request, "profile-mix", len(profile), tokenizer))
    plan.append(write_point(
        campaign, "profile-mix", profile, kind="profile", workload_class="mixed",
        concurrency=int(profile_cfg["concurrency"]), parameters={}, tokenizer=tokenizer,
    ))
    return plan


def prepare(args: argparse.Namespace) -> None:
    cfg = read_json(args.config)
    runtime = load_runtime_config(args.runtime_config)
    datasets = dataset_paths(runtime.data_root)
    campaign = args.campaign.resolve()
    campaign.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, campaign / "config.snapshot.json")
    tokenizer_dir = Path(runtime.tokenizer) if runtime.tokenizer else resolve_from_root(cfg["tokenizer_dir"])
    tokenizer = DeepSeekV4Tokenizer(tokenizer_dir, resolve_from_root(cfg["deepseek_v4_encoding"]))
    subsets = prepare_subsets(cfg, campaign, datasets)
    bases = compile_bases(cfg, campaign, subsets, tokenizer)
    plan = prepare_points(cfg, campaign, bases, tokenizer)
    metadata = {
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "config_sha256": sha256_file(args.config),
        "source_datasets": {key: str(path) for key, path in datasets.items()},
        "subsets": {key: str(path.relative_to(campaign)) for key, path in subsets.items()},
        "points": plan,
    }
    write_json(campaign / "plan.json", metadata)
    print(f"prepared {len(plan)} points under {campaign}")


def http_json(
    url: str,
    method: str = "GET",
    body: dict[str, Any] | None = None,
    timeout: float = 10,
    headers: dict[str, str] | None = None,
) -> Any:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        raw = response.read()
        if not raw:
            return {"status": response.status}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # SGLang administrative endpoints may return plain text while
            # vLLM returns JSON. Preserve both without backend-specific hacks.
            return {"status": response.status, "text": raw.decode(errors="replace")}


def http_text(url: str, timeout: float = 10, headers: dict[str, str] | None = None) -> str:
    request = urllib.request.Request(url)
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        return response.read().decode()


class MetricsSampler:
    def __init__(self, base_url: str, backend: str, output: Path, interval_s: float,
                 headers: dict[str, str] | None = None):
        self.url = base_url.rstrip("/") + get_backend(backend).metrics_path
        self.backend = backend
        self.headers = headers or {}
        self.output = output
        self.interval_s = interval_s
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name=f"{backend}-metrics-sampler", daemon=True)

    def start(self) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=max(10, self.interval_s * 4))

    def _run(self) -> None:
        with self.output.open("w") as handle:
            while not self.stop_event.is_set():
                row: dict[str, Any] = {
                    "wall_time": datetime.now(timezone.utc).isoformat(),
                    "monotonic_ns": time.monotonic_ns(),
                }
                try:
                    row["metrics"] = parse_server_metrics(
                        http_text(self.url, timeout=3, headers=self.headers), self.backend
                    )
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
                self.stop_event.wait(self.interval_s)


def wait_healthy(base_url: str, backend: str, timeout: float = 300,
                 headers: dict[str, str] | None = None) -> None:
    health_url = base_url.rstrip("/") + get_backend(backend).health_path
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            http_json(health_url, timeout=3, headers=headers)
            return
        except Exception:
            time.sleep(2)
    raise RuntimeError(f"server did not become healthy within {timeout}s: {base_url}")


def run_point(
    campaign: Path,
    result_root: Path,
    point: dict[str, Any],
    args: argparse.Namespace,
    cfg: dict[str, Any],
) -> None:
    output = result_root / point["label"]
    summary_path = output / "summary.json"
    if summary_path.exists() and not args.force:
        summary = read_json(summary_path)
        counts = summary.get("counts", {})
        if counts.get("completed") == counts.get("offered") and counts.get("failed") == 0:
            print(f"SKIP complete {point['label']}", flush=True)
            return
    workload = campaign / point["workload"]
    command = [
        sys.executable, "-m", "input_bench.cli", "run",
        "--workload", str(workload),
        "--backend", args.backend,
        "--base-url", args.base_url,
        "--model", args.model,
        "--output-dir", str(output),
        "--stream",
        "--pool-size", str(max(512, int(point["concurrency"]))),
        "--max-concurrency", str(point["concurrency"]),
        "--timeout", str(args.request_timeout),
        "--response-max-chars", "256",
        "--ttft-slo-ms", str(cfg["slo_ms"]["ttft"]),
        "--tpot-slo-ms", str(cfg["slo_ms"]["tpot"]),
        "--e2e-slo-ms", str(cfg["slo_ms"]["e2e"]),
    ]
    if args.tokenizer:
        command.extend(["--tokenizer", args.tokenizer])
    if args.runtime_config:
        command.extend(["--config", str(args.runtime_config)])
    print("RUN", point["label"], "requests=", point["requests"], "concurrency=", point["concurrency"], flush=True)
    started = time.monotonic()
    sampler = MetricsSampler(
        args.base_url, args.backend, output / "server-metrics.jsonl",
        args.metrics_interval, args.endpoint_headers,
    )
    sampler.start()
    try:
        subprocess.run(command, cwd=ROOT, check=True)
    finally:
        sampler.stop()
    print(f"DONE {point['label']} wall_s={time.monotonic() - started:.1f}", flush=True)


def aggregate(result_root: Path, plan: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for point in plan:
        summary_path = result_root / point["label"] / "summary.json"
        if not summary_path.exists():
            continue
        summary = read_json(summary_path)
        latency = summary["latency_ms"]
        metric_rows = []
        metrics_path = result_root / point["label"] / "server-metrics.jsonl"
        if metrics_path.exists():
            metric_rows = [
                json.loads(line).get("metrics", {})
                for line in metrics_path.read_text().splitlines()
                if line.strip()
            ]
        def metric_max(name: str) -> float | None:
            values = [row[name] for row in metric_rows if name in row]
            return max(values) if values else None
        def metric_delta(name: str) -> float | None:
            values = [row[name] for row in metric_rows if name in row]
            return values[-1] - values[0] if len(values) >= 2 else None
        def metric_last(name: str) -> float | None:
            values = [row[name] for row in metric_rows if name in row]
            return values[-1] if values else None
        hit_delta = metric_delta("prefix_cache_hits_total")
        query_delta = metric_delta("prefix_cache_queries_total")
        rows.append({
            "label": point["label"],
            "kind": point["kind"],
            "workload_class": point["workload_class"],
            "concurrency": point["concurrency"],
            "offered_requests": summary["counts"]["offered"],
            "completed_requests": summary["counts"]["completed"],
            "failed_requests": summary["counts"]["failed"],
            "duration_s": summary["measurement"]["duration_s"],
            "completed_rps": summary["rps"]["completed"],
            "input_tokens_s": summary["token_throughput_per_s"]["input"],
            "output_tokens_s": summary["token_throughput_per_s"]["output"],
            "ttft_mean_ms": latency["ttft"]["mean"],
            "ttft_p50_ms": latency["ttft"]["p50"],
            "ttft_p95_ms": latency["ttft"]["p95"],
            "ttft_p99_ms": latency["ttft"]["p99"],
            "tpot_mean_ms": latency["tpot"]["mean"],
            "tpot_p95_ms": latency["tpot"]["p95"],
            "e2e_mean_ms": latency["e2e"]["mean"],
            "e2e_p95_ms": latency["e2e"]["p95"],
            "scheduler_lag_p95_ms": latency["scheduler_lag"]["p95"],
            "client_queue_p95_ms": latency["client_queue_delay"]["p95"],
            "goodput_rps": summary["goodput"]["rps"],
            "goodput_fraction": summary["goodput"]["fraction"],
            "server_metric_samples": len(metric_rows),
            "server_running_max": metric_max("running"),
            "server_waiting_max": metric_max("waiting"),
            "server_kv_cache_usage_max": metric_max("kv_cache_usage"),
            "server_prompt_tokens_delta": metric_delta("prompt_tokens_total"),
            "server_generation_tokens_delta": metric_delta("generation_tokens_total"),
            "server_prefix_hit_fraction": (
                hit_delta / query_delta if hit_delta is not None and query_delta and query_delta > 0
                else metric_last("prefix_hit_fraction")
            ),
        })
    if not rows:
        return
    with (result_root / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    columns = ["label", "completed_rps", "output_tokens_s", "ttft_p95_ms", "tpot_p95_ms", "e2e_p95_ms", "failed_requests"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] + ["---:"] * (len(columns) - 1)) + "|"]
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.3f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    (result_root / "SUMMARY.md").write_text("# Campaign summary\n\n" + "\n".join(lines) + "\n")


def select_points(plan: list[dict[str, Any]], suite: str) -> list[dict[str, Any]]:
    if suite == "performance":
        return [point for point in plan if point["kind"] in {"preflight", "fixed", "poisson", "trace"}]
    if suite == "profile":
        return [point for point in plan if point["kind"] == "profile"]
    raise ValueError(f"unknown suite: {suite}")


def run(args: argparse.Namespace) -> None:
    runtime = load_runtime_config(args.runtime_config)
    args.backend = args.backend or runtime.backend
    args.base_url = args.base_url or runtime.base_url
    args.model = args.model or runtime.model or read_json(args.config).get("model")
    if not args.model:
        raise ValueError("--model is required (or set endpoint.model in runtime config)")
    args.tokenizer = args.tokenizer or runtime.tokenizer
    if args.results_root is None:
        args.results_root = runtime.results_root
    args.endpoint_headers = dict(runtime.headers)
    api_key = os.getenv(runtime.api_key_env)
    if api_key and not any(key.lower() == "authorization" for key in args.endpoint_headers):
        args.endpoint_headers["Authorization"] = f"Bearer {api_key}"
    backend = get_backend(args.backend)
    campaign = args.campaign.resolve()
    metadata = read_json(campaign / "plan.json")
    cfg = read_json(campaign / "config.snapshot.json")
    plan = select_points(metadata["points"], args.suite)
    if args.point:
        plan = [point for point in plan if any(fnmatch.fnmatch(point["label"], pattern) for pattern in args.point)]
        if not plan:
            raise RuntimeError(f"no campaign points match: {args.point}")
    wait_healthy(args.base_url, args.backend, headers=args.endpoint_headers)
    models = http_json(
        args.base_url.rstrip("/") + backend.models_path, headers=args.endpoint_headers
    )
    server_info: dict[str, Any] = {}
    for path in backend.info_paths:
        try:
            server_info[path] = http_json(
                args.base_url.rstrip("/") + path, headers=args.endpoint_headers
            )
            break
        except Exception as exc:
            server_info[path] = {"error": f"{type(exc).__name__}: {exc}"}
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_root = Path(args.results_root).resolve() if args.results_root else campaign / "results"
    result_root = results_root / f"{run_id}-{args.mode}-{args.backend}-{args.suite}"
    result_root.mkdir(parents=True, exist_ok=True)
    write_json(result_root / "run-metadata.json", {
        "run_id": run_id,
        "mode": args.mode,
        "backend": args.backend,
        "suite": args.suite,
        "base_url": args.base_url,
        "model": args.model,
        "server_info": server_info,
        "models": models,
        "plan_sha256": sha256_file(campaign / "plan.json"),
        "started_at": datetime.now(timezone.utc).isoformat(),
    })
    profile_started = False
    try:
        if args.suite == "profile":
            write_json(result_root / "start-profile.json", http_json(
                args.base_url.rstrip("/") + backend.profile_start_path,
                method="POST", timeout=60, headers=args.endpoint_headers,
            ))
            profile_started = True
        for point in plan:
            run_point(campaign, result_root, point, args, cfg)
            aggregate(result_root, plan)
    finally:
        if profile_started:
            try:
                write_json(result_root / "stop-profile.json", http_json(
                    args.base_url.rstrip("/") + backend.profile_stop_path,
                    method="POST", body={}, timeout=600, headers=args.endpoint_headers,
                ))
            finally:
                aggregate(result_root, plan)
    print(f"results: {result_root}")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    value.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    sub = value.add_subparsers(dest="command", required=True)
    prepare_p = sub.add_parser("prepare")
    prepare_p.add_argument("--runtime-config")
    run_p = sub.add_parser("run")
    run_p.add_argument("--runtime-config")
    run_p.add_argument("--backend", choices=SUPPORTED_BACKENDS)
    run_p.add_argument("--base-url")
    run_p.add_argument("--model")
    run_p.add_argument("--tokenizer")
    run_p.add_argument("--results-root", type=Path)
    run_p.add_argument("--mode", choices=["cvm", "baremetal", "vm"], required=True)
    run_p.add_argument("--suite", choices=["performance", "profile"], default="performance")
    run_p.add_argument("--point", action="append", help="only run labels matching this glob; repeatable")
    run_p.add_argument("--run-id")
    run_p.add_argument("--request-timeout", type=float, default=900)
    run_p.add_argument("--metrics-interval", type=float, default=0.5)
    run_p.add_argument("--force", action="store_true")
    return value


def main() -> None:
    args = parser().parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
