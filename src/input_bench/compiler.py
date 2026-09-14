from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from input_bench import __version__
from input_bench.arrivals.trace import TimestampTracePolicy
from input_bench.schema import SCHEMA_VERSION, SemanticSample, WorkloadRequest
from input_bench.tokenizer import Tokenizer, count_input, tokenizer_metadata


@dataclass(slots=True)
class CompileOptions:
    seed: int = 1
    max_samples: int | None = None
    min_input_tokens: int | None = None
    max_input_tokens: int | None = None
    min_output_tokens: int | None = None
    max_output_tokens: int | None = None
    default_max_output_tokens: int = 256
    bucket_boundaries: tuple[int, ...] = ()
    ignore_eos: bool = False

    def __post_init__(self) -> None:
        if self.max_samples is not None and self.max_samples < 0:
            raise ValueError("max_samples cannot be negative")
        if self.default_max_output_tokens < 1:
            raise ValueError("default_max_output_tokens must be positive")
        if any(x <= 0 for x in self.bucket_boundaries):
            raise ValueError("bucket boundaries must be positive")
        if tuple(sorted(self.bucket_boundaries)) != self.bucket_boundaries:
            raise ValueError("bucket boundaries must be sorted")
        for low, high, label in [(self.min_input_tokens, self.max_input_tokens, "input"),
                                 (self.min_output_tokens, self.max_output_tokens, "output")]:
            if low is not None and high is not None and low > high:
                raise ValueError(f"minimum {label} tokens exceeds maximum")


def _percentile(values: list[int], q: float) -> float | None:
    if not values: return None
    ordered = sorted(values); pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo))


def _stats(values: list[int]) -> dict[str, float | int | None]:
    if not values: return {k: None for k in ["min", "mean", "p50", "p90", "p95", "p99", "max"]}
    return {"min": min(values), "mean": mean(values), "p50": _percentile(values, .5),
            "p90": _percentile(values, .9), "p95": _percentile(values, .95),
            "p99": _percentile(values, .99), "max": max(values)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""): digest.update(chunk)
    return digest.hexdigest()


def fingerprint_sources(paths: list[Path]) -> list[dict[str, Any]]:
    values = []
    for path in sorted(set(p.resolve() for p in paths)):
        stat = path.stat()
        values.append({"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                       "sha256": sha256_file(path)})
    return values


def compile_workload(adapter: Any, tokenizer: Tokenizer, tokenizer_id: str,
                     arrival: Any, output: str | Path, options: CompileOptions,
                     *, tokenizer_revision: str | None = None,
                     source_paths: list[Path] | None = None,
                     adapter_parameters: dict[str, Any] | None = None) -> dict[str, Any]:
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    reasons: Counter[str] = Counter(); prepared: list[tuple[SemanticSample, int, int]] = []
    before = 0
    for sample in adapter.iter_samples():
        before += 1
        message_dicts = [m.to_dict() for m in sample.messages] if sample.messages is not None else None
        try: input_count = count_input(tokenizer, prompt=sample.prompt, messages=message_dicts)
        except ValueError: reasons["tokenization_error"] += 1; continue
        if sample.requested_output_tokens is not None: output_count = sample.requested_output_tokens
        elif sample.reference_output:
            output_count = max(1, len(tokenizer.encode(sample.reference_output, add_special_tokens=False)))
        else: output_count = options.default_max_output_tokens
        checks = [(options.min_input_tokens, input_count, "input_too_short", lambda a,b: b < a),
                  (options.max_input_tokens, input_count, "input_too_long", lambda a,b: b > a),
                  (options.min_output_tokens, output_count, "output_too_short", lambda a,b: b < a)]
        rejected = False
        for bound, value, reason, predicate in checks:
            if bound is not None and predicate(bound, value): reasons[reason] += 1; rejected = True; break
        if rejected: continue
        if options.max_output_tokens is not None:
            output_count = min(output_count, options.max_output_tokens)
        prepared.append((sample, input_count, max(1, output_count)))
    if options.max_samples is not None and len(prepared) > options.max_samples:
        picked = sorted(random.Random(options.seed).sample(range(len(prepared)), options.max_samples))
        prepared = [prepared[i] for i in picked]
        reasons["seed_sampling"] += before - sum(reasons.values()) - len(prepared)
    samples = [x[0] for x in prepared]
    if isinstance(arrival, TimestampTracePolicy):
        selected = arrival.select(samples); ids = {id(x) for x in selected}
        if len(selected) < len(prepared): reasons["trace_window"] += len(prepared) - len(selected)
        prepared = [x for x in prepared if id(x[0]) in ids]; samples = selected
    offsets = arrival.offsets(samples)
    if len(offsets) < len(prepared):
        reasons["arrival_duration"] += len(prepared) - len(offsets); prepared = prepared[:len(offsets)]
    requests = []
    for sequence, ((sample, input_count, output_count), offset) in enumerate(zip(prepared, offsets)):
        messages = [m.to_dict() for m in sample.messages] if sample.messages is not None else None
        metadata = dict(sample.metadata)
        if options.bucket_boundaries:
            metadata["input_token_bucket_upper"] = next((x for x in options.bucket_boundaries if input_count <= x), None)
        sampling = {"temperature": 0.0, "top_p": 1.0, "seed": options.seed}
        if options.ignore_eos:
            sampling["ignore_eos"] = True
        requests.append(WorkloadRequest(sample.sample_id, sequence, sample.source,
            sample.workload_class, "chat" if messages is not None else "completions",
            input_count, output_count, session_id=sample.session_id,
            parent_request_id=sample.parent_sample_id, scheduled_offset_s=offset,
            messages=messages, prompt=sample.prompt,
            sampling=sampling, metadata=metadata))
    with output.open("w", encoding="utf-8") as handle:
        for request in requests:
            handle.write(json.dumps(request.to_dict(), ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")) + "\n")
        handle.flush(); os.fsync(handle.fileno())
    input_values, output_values = [r.input_tokens for r in requests], [r.max_output_tokens for r in requests]
    all_reasons = Counter(adapter.skip_reasons); all_reasons.update(reasons)
    manifest = {"schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(), "tool_version": __version__,
        "dataset_files": fingerprint_sources(source_paths or []), "adapter": adapter.name,
        "adapter_parameters": adapter_parameters or {}, "adapter_metadata": adapter.manifest_metadata,
        "tokenizer": tokenizer_metadata(tokenizer, tokenizer_id, tokenizer_revision),
        "seed": options.seed, "arrival_policy": arrival.name,
        "arrival_parameters": arrival.parameters(), "compiler_parameters": asdict(options),
        "counts": {"before_filter": before, "after_filter": len(requests),
                   "discard_reasons": dict(sorted(all_reasons.items()))},
        "token_statistics": {"input": _stats(input_values), "output": _stats(output_values)},
        "workload_sha256": sha256_file(output)}
    manifest_path = output.with_name("manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    return manifest


def load_workload(path: str | Path) -> list[WorkloadRequest]:
    requests = []
    with Path(path).open() as handle:
        for line_no, line in enumerate(handle, 1):
            try: requests.append(WorkloadRequest.from_dict(json.loads(line)))
            except Exception as exc: raise ValueError(f"{path}:{line_no}: {exc}") from exc
    if [x.sequence_no for x in requests] != list(range(len(requests))):
        raise ValueError("sequence_no must be contiguous and start at zero")
    seen: set[str] = set()
    for request in requests:
        if request.request_id in seen: raise ValueError(f"duplicate request_id {request.request_id}")
        if request.parent_request_id is not None and request.parent_request_id not in seen:
            raise ValueError(f"parent_request_id must refer to an earlier request: {request.request_id}")
        seen.add(request.request_id)
    scheduled = [x.scheduled_offset_s for x in requests if x.scheduled_offset_s is not None]
    if scheduled != sorted(scheduled): raise ValueError("scheduled offsets must be monotonic")
    return requests
