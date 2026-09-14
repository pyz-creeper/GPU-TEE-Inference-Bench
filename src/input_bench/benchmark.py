from __future__ import annotations

from statistics import mean
from typing import Any

from input_bench.metrics import percentile


def aggregate_request_benchmark(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate whole-workload elapsed time without mixing in API baselines.

    Each duration is the monotonic measurement window produced by
    ``BenchmarkSender.run``. It includes creation and teardown of the HTTP client
    session plus all request scheduling/queueing, but excludes workload loading,
    tokenizer loading, and result serialization.
    """
    if not summaries:
        raise ValueError("at least one run summary is required")

    durations: list[float] = []
    runs: list[dict[str, Any]] = []
    for index, summary in enumerate(summaries, 1):
        duration = summary.get("measurement", {}).get("duration_s")
        if not isinstance(duration, (int, float)) or duration < 0:
            raise ValueError(f"run {index} has no valid measurement.duration_s")
        counts = summary.get("counts", {})
        durations.append(float(duration))
        runs.append({
            "repeat": index,
            "total_duration_s": float(duration),
            "offered": int(counts.get("offered", 0)),
            "completed": int(counts.get("completed", 0)),
            "failed": int(counts.get("failed", 0)),
        })

    return {
        "scope": "whole-workload client elapsed time for one configured endpoint; no comparison baseline",
        "timing_boundary": (
            "BenchmarkSender monotonic start to end; excludes compile, workload/tokenizer "
            "loading, and result serialization"
        ),
        "repeat_count": len(runs),
        "all_requests_completed": all(
            run["failed"] == 0 and run["completed"] == run["offered"] for run in runs
        ),
        "total_duration_s": {
            "values": durations,
            "min": min(durations),
            "mean": mean(durations),
            "p50": percentile(durations, 0.50),
            "p90": percentile(durations, 0.90),
            "p95": percentile(durations, 0.95),
            "p99": percentile(durations, 0.99),
            "max": max(durations),
        },
        "runs": runs,
    }
