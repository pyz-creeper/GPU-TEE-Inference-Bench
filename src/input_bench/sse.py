from __future__ import annotations

import json
from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass
from typing import Any


class SSEParseError(ValueError): pass


@dataclass(slots=True)
class SSEEvent:
    data: str
    event: str | None = None
    event_id: str | None = None


async def iter_sse(chunks: AsyncIterable[bytes]) -> AsyncIterator[SSEEvent]:
    buffer = ""
    async for chunk in chunks:
        try: buffer += chunk.decode("utf-8")
        except UnicodeDecodeError as exc: raise SSEParseError(f"invalid UTF-8: {exc}") from exc
        buffer = buffer.replace("\r\n", "\n").replace("\r", "\n")
        while "\n\n" in buffer:
            block, buffer = buffer.split("\n\n", 1)
            event = _parse_block(block)
            if event is not None: yield event
    if buffer.strip():
        event = _parse_block(buffer)
        if event is not None: yield event


def _parse_block(block: str) -> SSEEvent | None:
    data, event, event_id = [], None, None
    for line in block.split("\n"):
        if not line or line.startswith(":"): continue
        field, _, value = line.partition(":")
        if value.startswith(" "): value = value[1:]
        if field == "data": data.append(value)
        elif field == "event": event = value
        elif field == "id": event_id = value
    if not data: return None
    return SSEEvent("\n".join(data), event, event_id)


def parse_data_json(event: SSEEvent) -> dict[str, Any]:
    try: value = json.loads(event.data)
    except json.JSONDecodeError as exc: raise SSEParseError(f"invalid SSE data JSON: {exc}") from exc
    if not isinstance(value, dict): raise SSEParseError("SSE data must be a JSON object")
    return value
