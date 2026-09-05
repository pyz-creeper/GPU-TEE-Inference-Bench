from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from input_bench.metrics import summarize_events
from input_bench.sender import RequestEvent


def write_results(output_dir: str | Path, events: list[RequestEvent], bounds: dict[str, Any],
                  config: dict[str, Any], slos: dict[str, float | None] | None = None) -> dict[str, Any]:
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    rows = [event.to_dict() for event in events]
    events_path = output / "events.jsonl"
    with events_path.open("w") as handle:
        for row in rows: handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = summarize_events(rows, bounds, slos); summary["experiment_config"] = config
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc: raise RuntimeError("pyarrow is required for results.parquet export") from exc
    pq.write_table(pa.Table.from_pylist(rows), output / "results.parquet")
    return summary


def read_events(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open() as handle: return [json.loads(line) for line in handle if line.strip()]
