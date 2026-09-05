from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from input_bench.adapters.base import iter_jsonl
from input_bench.schema import SemanticSample
from input_bench.tokenizer import Tokenizer, exact_synthetic_text


class ServeGenBridge:
    name = "servegen"
    def __init__(self, source: str | Path | Iterable[Any], tokenizer: Tokenizer,
                 seed: int = 1, **_: Any):
        self.source, self.tokenizer, self.seed = source, tokenizer, seed
        self.skip_reasons = Counter()
        self.manifest_metadata = {"synthetic_method": "validated tokenizer-vocabulary blocks v1", "synthetic_seed": seed}

    def _rows(self) -> Iterator[tuple[int, dict[str, Any]]]:
        if not isinstance(self.source, (str, Path)):
            for i, request in enumerate(self.source, 1):
                data = dict(getattr(request, "data", {}) or {})
                yield i, {"timestamp": getattr(request, "timestamp"),
                          "request_id": getattr(request, "request_id", i), **data}
            return
        path = Path(self.source)
        if path.suffix == ".jsonl": yield from iter_jsonl(path)
        elif path.suffix == ".json":
            for i, row in enumerate(json.loads(path.read_text()), 1): yield i, row
        elif path.suffix == ".csv":
            with path.open(newline="") as handle:
                for i, row in enumerate(csv.DictReader(handle), 1): yield i, row
        else: raise ValueError("ServeGen bridge accepts iterable requests, .jsonl, .json, or .csv")

    def iter_samples(self, limit: int | None = None) -> Iterator[SemanticSample]:
        aliases_in = ("input_length", "input_tokens", "prompt_tokens", "isl")
        aliases_out = ("output_length", "output_tokens", "max_output_tokens", "osl")
        for emitted, (line_no, row) in enumerate(self._rows()):
            try:
                isl = int(next(row[k] for k in aliases_in if k in row))
                osl = int(next(row[k] for k in aliases_out if k in row))
            except (StopIteration, ValueError, TypeError):
                self.skip_reasons["missing_token_shape"] += 1; continue
            rid = str(row.get("request_id", line_no)); timestamp = float(row["timestamp"])
            prompt = exact_synthetic_text(self.tokenizer, isl, self.seed + line_no)
            known = {*aliases_in, *aliases_out, "request_id", "timestamp", "client_id"}
            yield SemanticSample(f"servegen:{rid}", "servegen", "long_context", "shape_only",
                prompt=prompt, requested_output_tokens=max(1, osl),
                metadata={"source_record_id": rid, "source_row": line_no,
                    "original_timestamp": timestamp, "timestamp_s": timestamp,
                    "client_id": row.get("client_id"),
                    "servegen_data": {k: v for k, v in row.items() if k not in known}})
            if limit is not None and emitted + 1 >= limit: return
