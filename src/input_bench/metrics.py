from __future__ import annotations

import math
from collections import Counter
from statistics import mean
from typing import Any


def percentile(values: list[float], q: float) -> float | None:
    if not values: return None
    ordered = sorted(values); pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def distribution(values: list[float]) -> dict[str, float | None]:
    return {"mean": mean(values) if values else None, "p50": percentile(values, .5),
            "p90": percentile(values, .9), "p95": percentile(values, .95),
            "p99": percentile(values, .99), "max": max(values) if values else None}


def summarize_events(events: list[dict[str, Any]], bounds: dict[str, Any] | None = None,
                     slos: dict[str, float | None] | None = None) -> dict[str, Any]:
    bounds, slos = bounds or {}, slos or {}
    start = bounds.get("monotonic_start_ns")
    end = bounds.get("monotonic_end_ns")
    if start is None:
        points = [e.get("request_start_ns") for e in events if e.get("request_start_ns") is not None]
        start = min(points) if points else 0
    if end is None:
        points = [e.get("request_end_ns") for e in events if e.get("request_end_ns") is not None]
        end = max(points) if points else start
    duration = max((end - start) / 1e9, 1e-12)
    started = [e for e in events if e.get("request_start_ns") is not None]
    completed = [e for e in events if not e.get("error_type") and e.get("request_end_ns") is not None]
    failed = [e for e in events if e.get("error_type")]
    def delta(e: dict[str, Any], a: str, b: str) -> float | None:
        return ((e[b] - e[a]) / 1e6 if e.get(a) is not None and e.get(b) is not None else None)
    queue = [x for e in events if (x := e.get("client_queue_delay_ns")) is not None]
    lag = [x for e in events if (x := e.get("scheduler_lag_ns")) is not None]
    ttft = [x for e in completed if e.get("streaming", True) and (x := delta(e, "request_start_ns", "first_content_ns")) is not None]
    e2e = [x for e in completed if (x := delta(e, "request_start_ns", "request_end_ns")) is not None]
    tpot, interchunk = [], []
    for e in completed:
        if e.get("streaming", True) and e.get("first_content_ns") is not None and e.get("request_end_ns") is not None and (e.get("output_tokens") or 0) > 0:
            tpot.append((e["request_end_ns"] - e["first_content_ns"]) / 1e6 / e["output_tokens"])
        times = e.get("chunk_times_ns") or []
        interchunk.extend((b - a) / 1e6 for a, b in zip(times, times[1:]))
    good = 0
    for e in completed:
        values = {"ttft_ms": delta(e, "request_start_ns", "first_content_ns") if e.get("streaming", True) else None,
                  "e2e_ms": delta(e, "request_start_ns", "request_end_ns")}
        own_tpot = None
        if e.get("streaming", True) and e.get("first_content_ns") is not None and (e.get("output_tokens") or 0) > 0:
            own_tpot = (e["request_end_ns"] - e["first_content_ns"]) / 1e6 / e["output_tokens"]
        values["tpot_ms"] = own_tpot
        if all(limit is None or (values[key] is not None and values[key] <= limit)
               for key, limit in slos.items()): good += 1
    input_tokens = sum(e.get("input_tokens", 0) for e in completed)
    output_tokens = (sum(e.get("output_tokens", 0) for e in completed)
                     if all(e.get("output_tokens", 0) is not None for e in completed) else None)
    return {"measurement": {**bounds, "duration_s": duration},
        "counts": {"offered": len(events), "started": len(started), "completed": len(completed), "failed": len(failed)},
        "rps": {"offered": len(events)/duration, "started": len(started)/duration,
                "completed": len(completed)/duration, "failed": len(failed)/duration},
        "token_throughput_per_s": {"input": input_tokens/duration, "output": output_tokens/duration if output_tokens is not None else None,
                                   "total": (input_tokens + output_tokens)/duration if output_tokens is not None else None},
        "latency_ms": {"scheduler_lag": distribution([x/1e6 for x in lag]),
            "client_queue_delay": distribution([x/1e6 for x in queue]), "ttft": distribution(ttft),
            "e2e": distribution(e2e), "tpot": distribution(tpot),
            "inter_chunk": distribution(interchunk)},
        "goodput": {"slo": slos, "count": good, "rps": good/duration,
                    "fraction": good/len(completed) if completed else 0.0},
        "errors": dict(sorted(Counter(e.get("error_type") for e in failed).items()))}
