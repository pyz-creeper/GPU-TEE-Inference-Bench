"""Complete-session compiler; legacy turn/arrival compilation stays available."""
from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from input_bench.adapters.swe_agent import SWEAgentTrajectoriesAdapter, NORMALIZATION, canonical
from input_bench.compiler import load_workload, sha256_file, _stats
from input_bench.schema import WorkloadRequest
from input_bench.tokenizer import count_input, tokenizer_metadata


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def content_hash(manifest: dict) -> str:
    return hashlib.sha256(canonical({k: v for k, v in manifest.items()
                                    if k not in {"created_at", "content_sha256"}}).encode()).hexdigest()


def prepare_bundle(source: Path, output: Path, tokenizer: Any, tokenizer_id: str, *,
                   revision: str, tokenizer_revision: str | None = None, seed: int = 1,
                   max_sessions: int = 20, max_input_tokens: int = 8192,
                   output_cap: int = 256, candidate_sessions: int | None = None) -> dict:
    if not revision or revision in {"main", "latest", "unspecified"}:
        raise ValueError("a fixed dataset revision is required")
    if min(max_sessions, max_input_tokens, output_cap) < 1:
        raise ValueError("session and token limits must be positive")
    if output.exists():
        raise FileExistsError(f"bundle already exists; choose a new directory: {output}")
    adapter = SWEAgentTrajectoriesAdapter(source, revision=revision, normalization=NORMALIZATION)
    sampling_frame = None
    if candidate_sessions is not None:
        import pyarrow.parquet as pq
        if candidate_sessions < max_sessions or any(p.suffix != '.parquet' for p in adapter.paths):
            raise ValueError('candidate_sessions requires Parquet and must be >= max_sessions')
        sizes = [pq.ParquetFile(p).metadata.num_rows for p in adapter.paths]
        total_rows = sum(sizes)
        picked = set(random.Random(seed).sample(range(total_rows), min(candidate_sessions, total_rows)))
        offset = 0
        selected_rows = {}
        for path, size in zip(adapter.paths, sizes):
            selected_rows[path.relative_to(adapter.root).as_posix()] = sorted(
                n - offset + 1 for n in picked if offset <= n < offset + size)
            offset += size
        adapter.selected_rows = selected_rows
        sampling_frame = {'policy': 'seeded-parquet-candidate-pool-v1', 'total_source_sessions': total_rows,
                          'candidate_sessions': len(picked), 'outside_candidate_pool': total_rows - len(picked),
                          'selected_source_rows': selected_rows}
    files = [{"path": p.relative_to(adapter.root).as_posix(), "sha256": sha256_file(p),
              "size": p.stat().st_size} for p in adapter.paths]
    candidates = []
    excluded = []
    for samples in adapter.iter_sessions():
        try:
            # Frozen histories grow with each turn; reject oversized sessions before recounting every prefix.
            last_count = count_input(tokenizer, prompt=None, messages=[m.to_dict() for m in samples[-1].messages])
            if last_count > max_input_tokens:
                raise ValueError("input_too_long")
            counts = [count_input(tokenizer, prompt=None, messages=[m.to_dict() for m in s.messages]) for s in samples]
            if max(counts) > max_input_tokens:
                raise ValueError("input_too_long")
            candidates.append((samples, counts))
        except ValueError as exc:
            excluded.append({"session_id": samples[0].session_id, "reason": str(exc)})
    indices = sorted(random.Random(seed).sample(range(len(candidates)), min(max_sessions, len(candidates))))
    selected = set(indices)
    excluded.extend({"session_id": s[0].session_id, "reason": "seed_sampling"}
                    for i, (s, _) in enumerate(candidates) if i not in selected)
    requests, sessions = [], []
    for i in indices:
        samples, counts = candidates[i]
        sessions.append({"session_id": samples[0].session_id, "request_ids": [s.sample_id for s in samples],
                         "turn_count": len(samples), **samples[0].metadata})
        for sample, count in zip(samples, counts):
            requests.append(WorkloadRequest(sample.sample_id, len(requests), "swe-agent", "agent_coding", "chat",
                count, output_cap, session_id=sample.session_id, parent_request_id=sample.parent_sample_id,
                messages=[m.to_dict() for m in sample.messages],
                sampling={"temperature": 0.0, "top_p": 1.0, "seed": seed},
                metadata={**sample.metadata, "turn_id": sample.turn_id, "normalization": NORMALIZATION}))
            requests[-1].metadata['reference_output_tokens'] = len(tokenizer.encode(
                sample.reference_output or '', add_special_tokens=False))
    if not requests:
        raise ValueError(f"no eligible sessions; exclusions: {excluded + adapter.exclusions}")
    output.mkdir(parents=True)
    workload = output / "workload.jsonl"
    workload.write_text("".join(canonical(r.to_dict()) + "\n" for r in requests))
    write_json(output / "sessions.json", {"selected": sessions, "excluded": adapter.exclusions + excluded})
    manifest = {"bundle_version": "agentic-replay-v1", "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(), "dataset_revision": revision,
        "sampling_frame": sampling_frame,
        "dataset_files": files, "normalization": {"version": NORMALIZATION, "parameters": {}},
        "tokenizer": tokenizer_metadata(tokenizer, tokenizer_id, tokenizer_revision),
        "compiler_parameters": {"seed": seed, "max_sessions": max_sessions,
                                "max_input_tokens": max_input_tokens, "output_cap": output_cap,
                                "candidate_sessions": candidate_sessions},
        "counts": {"sessions": len(sessions), "requests": len(requests),
                   "requested_sessions": max_sessions, "session_shortfall": max(0, max_sessions - len(sessions)),
                   "excluded_sessions": len(excluded) + len(adapter.exclusions)},
        "token_statistics": {"reference_input": _stats([r.input_tokens for r in requests]),
                             "total_reference_input_tokens": sum(r.input_tokens for r in requests),
                             "recorded_reference_output": _stats([r.metadata['reference_output_tokens'] for r in requests]),
                             "logical_output_cap": output_cap, "total_output_cap": output_cap * len(requests)},
        "workload_sha256": sha256_file(workload), "sessions_sha256": sha256_file(output / "sessions.json")}
    manifest["content_sha256"] = content_hash(manifest)
    write_json(output / "manifest.json", manifest)
    validate_bundle(output)
    return manifest


def validate_bundle(bundle: Path) -> tuple[list[WorkloadRequest], dict, dict]:
    manifest = json.loads((bundle / "manifest.json").read_text())
    if manifest.get("bundle_version") != "agentic-replay-v1":
        raise ValueError("requires agentic-replay-v1 bundle; prepare again from source")
    for name in ("workload", "sessions"):
        path = bundle / ("workload.jsonl" if name == "workload" else "sessions.json")
        if sha256_file(path) != manifest[f"{name}_sha256"]:
            raise ValueError(f"{name} hash mismatch")
    if content_hash(manifest) != manifest["content_sha256"]:
        raise ValueError("manifest content hash mismatch")
    requests = load_workload(bundle / "workload.jsonl")
    sessions = json.loads((bundle / "sessions.json").read_text())
    grouped: dict[str, list[str]] = {}
    for r in requests:
        prior = grouped.setdefault(r.session_id, [])
        if not r.session_id or r.parent_request_id != (prior[-1] if prior else None):
            raise ValueError("broken session dependency chain")
        if r.metadata.get("turn_id") != len(prior) or r.scheduled_offset_s is not None:
            raise ValueError("invalid session turn/order")
        if r.endpoint_kind != "chat" or not r.messages or any(
            set(m) != {"role", "content"} or m["role"] not in {"system", "user", "assistant"}
            or not isinstance(m["content"], str) for m in r.messages):
            raise ValueError("bundle messages must be normalized text")
        if r.max_output_tokens != manifest["compiler_parameters"]["output_cap"]:
            raise ValueError("logical output cap mismatch")
        prior.append(r.request_id)
    expected = [{"session_id": sid, "request_ids": ids} for sid, ids in grouped.items()]
    if expected != [{k: s[k] for k in ("session_id", "request_ids")} for s in sessions["selected"]]:
        raise ValueError("session selection mismatch")
    if not requests or len(requests) != manifest["counts"]["requests"] or len(grouped) != manifest["counts"]["sessions"]:
        raise ValueError("bundle counts mismatch")
    return requests, manifest, sessions
