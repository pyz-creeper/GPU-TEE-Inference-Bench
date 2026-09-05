"""Portable runtime configuration for dataset compilation and replay."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SUPPORTED_BACKENDS = ("vllm", "sglang")
DEFAULT_BASE_URLS = {
    "vllm": "http://127.0.0.1:8000",
    "sglang": "http://127.0.0.1:30000",
}


def _expand(value: str) -> str:
    return os.path.expanduser(os.path.expandvars(value))


def _path(value: str | None, base: Path) -> Path | None:
    if value is None:
        return None
    path = Path(_expand(value))
    return path.resolve() if path.is_absolute() else (base / path).resolve()


@dataclass(slots=True)
class RuntimeConfig:
    source: Path | None = None
    data_root: Path = Path("/data/benchmarks")
    tokenizer: str | None = None
    tokenizer_revision: str | None = None
    workload: Path | None = None
    results_root: Path | None = None
    backend: str = "vllm"
    base_url: str = DEFAULT_BASE_URLS["vllm"]
    model: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    headers: dict[str, str] = field(default_factory=dict)
    tls_verify: bool = True


def load_runtime_config(path: str | Path | None = None) -> RuntimeConfig:
    """Load JSON config; environment variables override file values.

    Filesystem paths in the JSON are resolved relative to the config file.
    Tokenizer ``path`` is treated as a path, while tokenizer ``id`` remains a
    Hugging Face identifier.
    """
    selected = path or os.getenv("INPUT_BENCH_CONFIG")
    source = Path(_expand(str(selected))).resolve() if selected else None
    raw: dict[str, Any] = {}
    if source is not None:
        try:
            raw = json.loads(source.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON config {source}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"config root must be an object: {source}")
    base = source.parent if source else Path.cwd()
    data = raw.get("data", {})
    tokenizer = raw.get("tokenizer", {})
    workload = raw.get("workload", {})
    results = raw.get("results", {})
    endpoint = raw.get("endpoint", {})
    for name, value in {
        "data": data, "tokenizer": tokenizer, "workload": workload,
        "results": results, "endpoint": endpoint,
    }.items():
        if not isinstance(value, dict):
            raise ValueError(f"config field {name!r} must be an object")

    tokenizer_value: str | None
    if tokenizer.get("path") is not None:
        tokenizer_path = _path(str(tokenizer["path"]), base)
        tokenizer_value = str(tokenizer_path) if tokenizer_path else None
    else:
        tokenizer_value = tokenizer.get("id")

    backend = str(os.getenv("INPUT_BENCH_BACKEND", endpoint.get("backend", "vllm"))).lower()
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"endpoint.backend must be one of: {', '.join(SUPPORTED_BACKENDS)}")
    headers = endpoint.get("headers", {})
    if not isinstance(headers, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in headers.items()):
        raise ValueError("endpoint.headers must be an object of string keys and values")

    data_root_value = os.getenv("INPUT_BENCH_DATA_ROOT") or data.get("root") or "/data/benchmarks"
    workload_value = os.getenv("INPUT_BENCH_WORKLOAD") or workload.get("path")
    results_value = os.getenv("INPUT_BENCH_RESULTS_ROOT") or results.get("root")
    tokenizer_value = os.getenv("INPUT_BENCH_TOKENIZER") or tokenizer_value
    base_url = os.getenv("INPUT_BENCH_BASE_URL") or endpoint.get("base_url") or DEFAULT_BASE_URLS[backend]
    model = os.getenv("INPUT_BENCH_MODEL") or endpoint.get("model")
    return RuntimeConfig(
        source=source,
        data_root=_path(str(data_root_value), base) or Path("/data/benchmarks"),
        tokenizer=str(tokenizer_value) if tokenizer_value is not None else None,
        tokenizer_revision=tokenizer.get("revision"),
        workload=_path(str(workload_value), base) if workload_value else None,
        results_root=_path(str(results_value), base) if results_value else None,
        backend=backend,
        base_url=str(base_url),
        model=str(model) if model is not None else None,
        api_key_env=str(endpoint.get("api_key_env", "OPENAI_API_KEY")),
        headers=dict(headers),
        tls_verify=bool(endpoint.get("tls_verify", True)),
    )


def resolve_tokenizer_reference(identifier: str, manifest_dir: Path) -> str:
    """Resolve a local tokenizer stored relative to a workload manifest."""
    candidate = Path(_expand(identifier))
    if candidate.is_absolute():
        return str(candidate)
    relative = (manifest_dir / candidate).resolve()
    return str(relative) if relative.exists() else identifier


def portable_tokenizer_reference(identifier: str, output: Path) -> str:
    """Store local tokenizer paths relative to the generated manifest."""
    candidate = Path(_expand(identifier))
    if not candidate.exists():
        return identifier
    return os.path.relpath(candidate.resolve(), output.resolve().parent)
