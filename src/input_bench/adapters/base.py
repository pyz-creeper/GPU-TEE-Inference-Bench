from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol

from input_bench.schema import SemanticSample


class Adapter(Protocol):
    name: str
    skip_reasons: dict[str, int]
    manifest_metadata: dict[str, Any]
    def iter_samples(self, limit: int | None = None) -> Iterator[SemanticSample]: ...


class AdapterError(ValueError):
    pass


def iter_json_array(path: Path) -> Iterator[tuple[int, Any]]:
    """Incrementally decode a top-level JSON array without loading it in memory."""
    decoder, buf, pos, index, eof = json.JSONDecoder(), "", 0, 0, False
    with path.open(encoding="utf-8") as handle:
        while True:
            if pos >= len(buf) and not eof:
                buf, pos = handle.read(1 << 20), 0
                if not buf:
                    eof = True
            while pos < len(buf) and (buf[pos].isspace() or buf[pos] in "[,"):
                pos += 1
            if pos < len(buf) and buf[pos] == "]":
                return
            while True:
                try:
                    value, end = decoder.raw_decode(buf, pos)
                    index += 1
                    yield index, value
                    buf, pos = buf[end:], 0
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise AdapterError(f"{path}: malformed JSON near record {index + 1}")
                    chunk = handle.read(1 << 20)
                    # Detect EOF from the read itself.  Checking the accumulated
                    # buffer left a non-empty partial record spinning forever.
                    if not chunk:
                        eof = True
                    buf = buf[pos:] + chunk
                    pos = 0
                    # A chunk boundary can fall after a comma/newline.  Since
                    # raw_decode does not consume leading whitespace, normalize
                    # the newly accumulated buffer before retrying.
                    while pos < len(buf) and (buf[pos].isspace() or buf[pos] in "[,"):
                        pos += 1
                    if pos < len(buf) and buf[pos] == "]":
                        return


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                raise AdapterError(f"{path}:{line_no}: invalid JSON: {exc}") from exc


def iter_parquet(paths: list[Path]) -> Iterator[tuple[Path, int, dict[str, Any]]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to read Parquet sources") from exc
    for path in sorted(paths):
        row_no = 0
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=64):
            for row in batch.to_pylist():
                row_no += 1
                yield path, row_no, row


def parse_json_field(value: Any, *, source: Path, record_id: str, field: str) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise AdapterError(f"{source}: record {record_id}: invalid {field}: {exc}") from exc
    return value


def normalized_message(raw: dict[str, Any], role_map: dict[str, str]) -> dict[str, Any]:
    role_key = raw.get("role", raw.get("from", raw.get("type", "")))
    role = role_map.get(str(role_key).lower(), str(role_key).lower())
    content = raw.get("content", raw.get("value", raw.get("text", "")))
    extra = {k: v for k, v in raw.items()
             if k not in {"role", "from", "type", "content", "value", "text"}}
    return {"role": role, "content": content, **extra}
