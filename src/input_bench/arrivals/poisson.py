from __future__ import annotations
import random
from input_bench.schema import SemanticSample

class PoissonPolicy:
    name = "poisson"
    def __init__(self, request_rate: float, seed: int = 1, duration: float | None = None,
                 max_concurrency: int | None = None):
        if request_rate <= 0: raise ValueError("request_rate must be positive")
        if duration is not None and duration < 0: raise ValueError("duration cannot be negative")
        self.request_rate, self.seed, self.duration = request_rate, seed, duration
        self.max_concurrency = max_concurrency
    def offsets(self, samples: list[SemanticSample]) -> list[float]:
        rng, now, values = random.Random(self.seed), 0.0, []
        for i in range(len(samples)):
            if i: now += rng.expovariate(self.request_rate)
            if self.duration is not None and now > self.duration: break
            values.append(now)
        return values
    def parameters(self) -> dict[str, object]:
        return {"request_rate": self.request_rate, "seed": self.seed,
                "duration": self.duration, "max_concurrency": self.max_concurrency}
