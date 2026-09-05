"""Serving-backend capabilities shared by CLI and campaign runners."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BackendSpec:
    name: str
    chat_path: str = "/v1/chat/completions"
    completions_path: str = "/v1/completions"
    health_path: str = "/health"
    models_path: str = "/v1/models"
    metrics_path: str = "/metrics"
    profile_start_path: str = "/start_profile"
    profile_stop_path: str = "/stop_profile"
    info_paths: tuple[str, ...] = ()


BACKENDS = {
    "vllm": BackendSpec("vllm", info_paths=("/version",)),
    "sglang": BackendSpec("sglang", info_paths=("/server_info", "/get_model_info")),
}

METRIC_MAPS = {
    "vllm": {
        "vllm:num_requests_running": "running",
        "vllm:num_requests_waiting": "waiting",
        "vllm:kv_cache_usage_perc": "kv_cache_usage",
        "vllm:prompt_tokens_total": "prompt_tokens_total",
        "vllm:generation_tokens_total": "generation_tokens_total",
        "vllm:prefix_cache_hits_total": "prefix_cache_hits_total",
        "vllm:prefix_cache_queries_total": "prefix_cache_queries_total",
    },
    "sglang": {
        "sglang:num_running_reqs": "running",
        "sglang:num_queue_reqs": "waiting",
        "sglang:token_usage": "kv_cache_usage",
        "sglang:prompt_tokens_total": "prompt_tokens_total",
        "sglang:generation_tokens_total": "generation_tokens_total",
        "sglang:cache_hit_rate": "prefix_hit_fraction",
    },
}


def get_backend(name: str) -> BackendSpec:
    try:
        return BACKENDS[name]
    except KeyError as exc:
        raise ValueError(f"unsupported backend {name!r}; choose one of: {', '.join(BACKENDS)}") from exc


def parse_server_metrics(text: str, backend: str) -> dict[str, float]:
    """Normalize selected Prometheus metrics emitted by vLLM or SGLang."""
    get_backend(backend)
    metric_map = METRIC_MAPS[backend]
    values: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        metric = line.split("{", 1)[0].split(None, 1)[0]
        if metric not in metric_map:
            continue
        try:
            canonical = metric_map[metric]
            values[canonical] = values.get(canonical, 0.0) + float(line.rsplit(None, 1)[1])
        except (IndexError, ValueError):
            continue
    return values
