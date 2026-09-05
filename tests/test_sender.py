import asyncio
import json

from aiohttp import web

from input_bench.schema import WorkloadRequest
from input_bench.sender import BenchmarkSender, SenderConfig
from input_bench.tokenizer import WhitespaceTokenizer


async def start_server(handler):
    app = web.Application()
    app.router.add_post("/v1/completions", handler)
    app.router.add_post("/v1/chat/completions", handler)
    runner = web.AppRunner(app); await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0); await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}"


def request(i, *, chat=False, scheduled=None):
    return WorkloadRequest(str(i), i, "test", "chat", "chat" if chat else "completions", 2, 4,
        scheduled_offset_s=scheduled, messages=[{"role": "user", "content": "hi"}] if chat else None,
        prompt=None if chat else "hi there")


async def test_stream_role_only_chunk_usage_and_split_boundaries():
    async def handler(req):
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"}); await response.prepare(req)
        payload = ('data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
                   'data: {"choices":[{"delta":{"content":"hello "}}]}\n\n'
                   'data: {"choices":[{"delta":{"content":"world"}}],"usage":{"completion_tokens":2}}\n\n'
                   'data: [DONE]\n\n').encode()
        for piece in [payload[:7], payload[7:61], payload[61:]]: await response.write(piece)
        await response.write_eof(); return response
    runner, url = await start_server(handler)
    try:
        events, _ = await BenchmarkSender(SenderConfig(url, "m", max_concurrency=2), WhitespaceTokenizer()).run([request(0, chat=True)])
    finally: await runner.cleanup()
    event = events[0]
    assert event.error_type is None and event.response_text == "hello world"
    assert event.first_content_ns == event.chunk_times_ns[0] and len(event.chunk_times_ns) == 2
    assert event.server_usage["completion_tokens"] == 2


async def test_reasoning_content_counts_as_generated_content():
    async def handler(req):
        body = await req.json()
        if body["stream"]:
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(req)
            await response.write(
                b'data: {"choices":[{"delta":{"reasoning":"think "}}]}\n\n'
                b'data: {"choices":[{"delta":{"content":"answer"}}],"usage":{"prompt_tokens":3,"completion_tokens":2}}\n\n'
                b'data: [DONE]\n\n')
            await response.write_eof()
            return response
        return web.json_response({
            "choices": [{"message": {"reasoning_content": "think ", "content": "answer"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        })

    runner, url = await start_server(handler)
    try:
        stream_events, _ = await BenchmarkSender(
            SenderConfig(url, "m"), WhitespaceTokenizer()).run([request(0, chat=True)])
        json_events, _ = await BenchmarkSender(
            SenderConfig(url, "m", stream=False), WhitespaceTokenizer()).run([request(0, chat=True)])
    finally:
        await runner.cleanup()

    for event in [stream_events[0], json_events[0]]:
        assert event.response_text == "think answer"
        assert event.input_tokens == 3
        assert event.output_tokens == 2
        assert event.first_content_ns is not None


async def test_nonstream_both_endpoints_and_timing_separation():
    async def handler(req):
        body = await req.json()
        choice = {"message": {"content": "chat answer"}} if "messages" in body else {"text": "plain answer"}
        return web.json_response({"choices": [choice], "usage": {"completion_tokens": 2}})
    runner, url = await start_server(handler)
    try:
        events, _ = await BenchmarkSender(SenderConfig(url, "m", stream=False), WhitespaceTokenizer()).run(
            [request(0, scheduled=0), request(1, chat=True, scheduled=.001)])
    finally: await runner.cleanup()
    assert [e.response_text for e in events] == ["plain answer", "chat answer"]
    assert all(e.scheduled_ns <= e.admitted_ns <= e.request_start_ns <= e.headers_ns <= e.request_end_ns for e in events)


async def test_fixed_concurrency_never_exceeded_and_failure_isolated():
    active = 0; peak = 0
    async def handler(req):
        nonlocal active, peak
        active += 1; peak = max(peak, active); await asyncio.sleep(.02); active -= 1
        body = await req.json()
        if body.get("prompt") == "bad": return web.Response(status=503, text="unavailable")
        return web.json_response({"choices": [{"text": "ok"}]})
    runner, url = await start_server(handler)
    rows = [request(i) for i in range(6)]; rows[2].prompt = "bad"
    try:
        events, _ = await BenchmarkSender(SenderConfig(url, "m", stream=False, max_concurrency=2), WhitespaceTokenizer()).run(rows)
    finally: await runner.cleanup()
    assert peak <= 2 and len(events) == 6
    assert events[2].error_type == "http_error" and sum(e.error_type is None for e in events) == 5
    assert events[-1].client_queue_delay_ns > 0


async def test_session_parent_finishes_before_child_starts():
    order = []
    async def handler(req):
        body = await req.json(); order.append((body["prompt"], "start"))
        await asyncio.sleep(.01); order.append((body["prompt"], "end"))
        return web.json_response({"choices": [{"text": "ok"}]})
    runner, url = await start_server(handler)
    parent, child = request(0, scheduled=0), request(1, scheduled=0)
    child.parent_request_id = parent.request_id
    try:
        events, _ = await BenchmarkSender(SenderConfig(url, "m", stream=False, max_concurrency=2), WhitespaceTokenizer()).run([parent, child])
    finally: await runner.cleanup()
    assert order == [("hi there", "start"), ("hi there", "end"),
                     ("hi there", "start"), ("hi there", "end")]
    assert events[0].request_end_ns <= events[1].request_start_ns


async def test_timeout_parse_error_and_truncated_stream():
    async def timeout_handler(req):
        await asyncio.sleep(.1); return web.json_response({"choices": [{"text": "late"}]})
    runner, url = await start_server(timeout_handler)
    try:
        events, _ = await BenchmarkSender(SenderConfig(url, "m", stream=False, timeout_s=.01), WhitespaceTokenizer()).run([request(0)])
    finally: await runner.cleanup()
    assert events[0].error_type == "timeout"

    async def broken(req):
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"}); await response.prepare(req)
        await response.write(b'data: {"choices": [}\n\n'); await response.write_eof(); return response
    runner, url = await start_server(broken)
    try:
        events, _ = await BenchmarkSender(SenderConfig(url, "m"), WhitespaceTokenizer()).run([request(0)])
    finally: await runner.cleanup()
    assert events[0].error_type == "parse_error"

    async def cutoff(req):
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"}); await response.prepare(req)
        await response.write(b'data: {"choices":[{"text":"partial"}]}\n\n'); await response.write_eof(); return response
    runner, url = await start_server(cutoff)
    try:
        events, _ = await BenchmarkSender(SenderConfig(url, "m"), WhitespaceTokenizer()).run([request(0)])
    finally: await runner.cleanup()
    assert events[0].error_type == "parse_error" and "before data: [DONE]" in events[0].error_message


async def test_retry_clears_prior_error_state():
    calls = 0
    async def handler(req):
        nonlocal calls
        calls += 1
        if calls == 1: return web.Response(status=503, text="retry me")
        return web.json_response({"choices": [{"text": "recovered"}]})
    runner, url = await start_server(handler)
    try:
        events, _ = await BenchmarkSender(SenderConfig(url, "m", stream=False, retries=1), WhitespaceTokenizer()).run([request(0)])
    finally: await runner.cleanup()
    assert calls == 2 and events[0].retries == 1
    assert events[0].error_type is None and events[0].response_text == "recovered"


def test_vllm_and_sglang_use_openai_compatible_routes():
    row = request(0)
    chat = request(1, chat=True)
    for backend in ("vllm", "sglang"):
        sender = BenchmarkSender(SenderConfig("http://server", "m", backend=backend), WhitespaceTokenizer())
        assert sender._payload(row)[0] == "/v1/completions"
        assert sender._payload(chat)[0] == "/v1/chat/completions"
