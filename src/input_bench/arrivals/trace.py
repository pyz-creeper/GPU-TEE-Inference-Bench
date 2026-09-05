from __future__ import annotations
import math
from input_bench.schema import SemanticSample

class TimestampTracePolicy:
    name = "timestamp-trace"
    def __init__(self, time_scale: float = 1.0, start_offset: float = 0.0,
                 window_start: float | None = None, window_end: float | None = None,
                 max_concurrency: int | None = None):
        if time_scale <= 0: raise ValueError("time_scale must be positive")
        self.time_scale, self.start_offset = time_scale, start_offset
        self.window_start, self.window_end, self.max_concurrency = window_start, window_end, max_concurrency
    def select(self, samples: list[SemanticSample]) -> list[SemanticSample]:
        kept = []
        last = -math.inf
        for sample in samples:
            value = sample.metadata.get("timestamp_s")
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < last:
                raise ValueError(f"invalid or unordered trace timestamp for {sample.sample_id}")
            last = value
            if self.window_start is not None and value < self.window_start: continue
            if self.window_end is not None and value > self.window_end: continue
            kept.append(sample)
        return kept
    def offsets(self, samples: list[SemanticSample]) -> list[float]:
        if not samples: return []
        first = float(samples[0].metadata["timestamp_s"])
        return [self.start_offset + (float(x.metadata["timestamp_s"]) - first) / self.time_scale
                for x in samples]
    def parameters(self) -> dict[str, object]:
        return {"time_scale": self.time_scale, "start_offset": self.start_offset,
                "window_start": self.window_start, "window_end": self.window_end,
                "max_concurrency": self.max_concurrency}
