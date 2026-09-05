from __future__ import annotations
from input_bench.schema import SemanticSample

class FixedConcurrencyPolicy:
    name = "fixed-concurrency"
    def __init__(self, concurrency: int = 1):
        if concurrency < 1: raise ValueError("concurrency must be >= 1")
        self.concurrency = concurrency
    def offsets(self, samples: list[SemanticSample]) -> list[None]: return [None] * len(samples)
    def parameters(self) -> dict[str, object]: return {"concurrency": self.concurrency}
