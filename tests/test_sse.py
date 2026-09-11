import asyncio
import pytest
from input_bench.sse import iter_sse

async def chunks():
    for value in [b"data: one\n", b"data: two\n\ndata: three\n\n"]: yield value

async def test_sse_multiline_and_multiple_events():
    assert [x.data async for x in iter_sse(chunks())] == ["one\ntwo", "three"]

async def test_utf8_and_crlf_split_across_bytes():
    async def source():
        for b in 'data: 中文\r\n\r\ndata: next\r\r'.encode(): yield bytes([b])
    assert [e.data async for e in iter_sse(source())]==['中文','next']
