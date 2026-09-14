from __future__ import annotations

import json
import codecs
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
    decoder = codecs.getincrementaldecoder("utf-8")()
    buffer = ""
    lines = []
    try:
        async for chunk in chunks:
            buffer += decoder.decode(chunk)
            # A trailing CR might be the first byte of a CRLF spanning chunks.
            trailing_cr = buffer.endswith("\r")
            body = buffer[:-1] if trailing_cr else buffer
            buffer = body.replace("\r\n", "\n").replace("\r", "\n") + ("\r" if trailing_cr else "")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.rstrip("\r")
                if not line:
                    event = _parse_block("\n".join(lines)); lines = []
                    if event is not None: yield event
                else:
                    lines.append(line)
        buffer += decoder.decode(b"", final=True)
    except UnicodeDecodeError as exc:
        raise SSEParseError("invalid UTF-8 in SSE stream") from exc
    if buffer: lines.append(buffer.rstrip("\r"))
    if lines:
        event = _parse_block("\n".join(lines))
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
