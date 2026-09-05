from __future__ import annotations
from typing import Protocol
from input_bench.schema import SemanticSample

class ArrivalPolicy(Protocol):
    name: str
    def offsets(self, samples: list[SemanticSample]) -> list[float | None]: ...
    def parameters(self) -> dict[str, object]: ...
