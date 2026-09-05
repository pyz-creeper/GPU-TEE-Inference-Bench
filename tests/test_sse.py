import asyncio
import pytest
from input_bench.sse import iter_sse

async def chunks():
    for value in [b"data: one\n", b"data: two\n\ndata: three\n\n"]: yield value

async def test_sse_multiline_and_multiple_events():
    assert [x.data async for x in iter_sse(chunks())] == ["one\ntwo", "three"]
