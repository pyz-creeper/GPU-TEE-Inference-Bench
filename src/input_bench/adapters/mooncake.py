from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from input_bench.adapters.base import iter_jsonl
from input_bench.schema import SemanticSample
from input_bench.tokenizer import Tokenizer, exact_synthetic_text


class MooncakeTraceAdapter:
    name = "mooncake"
    def __init__(self, source: str | Path, tokenizer: Tokenizer, seed: int = 1,
                 timestamp_unit: str = "milliseconds", **_: Any):
        if timestamp_unit not in {"seconds", "milliseconds", "microseconds"}:
            raise ValueError("timestamp_unit must be seconds, milliseconds, or microseconds")
        self.source, self.tokenizer, self.seed, self.timestamp_unit = Path(source), tokenizer, seed, timestamp_unit
        self.skip_reasons = Counter()
        self.manifest_metadata = {"synthetic_method": "validated tokenizer-vocabulary blocks v1", "synthetic_seed": seed,
            "timestamp_unit": timestamp_unit,
            "timestamp_unit_basis": "Mooncake FAST'25 trace README: relative arrival time in milliseconds",
            "timestamp_unit_source": "https://github.com/kvcache-ai/Mooncake/blob/main/FAST25-release/README.md"}

    def iter_samples(self, limit: int | None = None) -> Iterator[SemanticSample]:
        factor = {"seconds": 1.0, "milliseconds": 1e-3, "microseconds": 1e-6}[self.timestamp_unit]
        last = -math.inf; emitted = 0
        for line_no, row in iter_jsonl(self.source):
            timestamp = row.get("timestamp")
            if not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp) or timestamp < 0:
                raise ValueError(f"{self.source}:{line_no}: invalid timestamp")
            if timestamp < last: raise ValueError(f"{self.source}:{line_no}: timestamp is unordered")
            last = timestamp
            input_length, output_length = int(row["input_length"]), int(row["output_length"])
            hash_ids = row.get("hash_ids", [])
            prompt = exact_synthetic_text(self.tokenizer, input_length, self.seed, hash_ids)
            actual = len(self.tokenizer.encode(prompt, add_special_tokens=False))
            if actual != input_length:
                raise ValueError(f"{self.source}:{line_no}: synthetic token count {actual} != {input_length}")
            stem = self.source.stem
            yield SemanticSample(f"mooncake:{stem}:{line_no}", "mooncake", "long_context", "shape_only",
                prompt=prompt, requested_output_tokens=max(1, output_length),
                metadata={"source_record_id": line_no, "source_row": line_no,
                    "original_timestamp": timestamp, "timestamp_s": timestamp * factor,
                    "input_length": input_length, "output_length": output_length,
                    "hash_ids": hash_ids})
            emitted += 1
            if limit is not None and emitted >= limit: return
