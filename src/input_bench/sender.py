from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from input_bench.backends import get_backend
from input_bench.schema import WorkloadRequest
from input_bench.sse import SSEParseError, iter_sse, parse_data_json
from input_bench.tokenizer import Tokenizer


@dataclass(slots=True)
class SenderConfig:
    base_url: str
    model: str
    stream: bool = True
    api_key: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    timeout_s: float = 300.0
    pool_size: int = 100
    max_concurrency: int = 100
    tls_verify: bool = True
    response_max_chars: int = 10000
    retries: int = 0
    backend: str = "vllm"


@dataclass(slots=True)
class RequestEvent:
    request_id: str
    sequence_no: int
    scheduled_ns: int | None = None
    scheduler_wakeup_ns: int | None = None
    admitted_ns: int | None = None
    request_start_ns: int | None = None
    headers_ns: int | None = None
    first_content_ns: int | None = None
    chunk_times_ns: list[int] = field(default_factory=list)
    request_end_ns: int | None = None
    client_queue_delay_ns: int | None = None
    scheduler_lag_ns: int | None = None
    http_status: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    retries: int = 0
    response_text: str = ""
    response_truncated: bool = False
    server_usage: dict[str, Any] | None = None
    output_tokens: int = 0
    input_tokens: int = 0
    max_output_tokens: int = 0

    def to_dict(self) -> dict[str, Any]: return asdict(self)


class BenchmarkCancelled(asyncio.CancelledError):
    def __init__(self, events: list[RequestEvent], bounds: dict[str, Any]):
        super().__init__("benchmark cancelled")
        self.events, self.bounds = events, bounds


class BenchmarkSender:
    def __init__(self, config: SenderConfig, tokenizer: Tokenizer):
        self.config, self.tokenizer = config, tokenizer

    async def run(self, requests: list[WorkloadRequest]) -> tuple[list[RequestEvent], dict[str, Any]]:
        try: import aiohttp
        except ImportError as exc: raise RuntimeError("aiohttp is required to run workloads") from exc
        wall_start = datetime.now(timezone.utc); epoch = time.monotonic_ns()
        semaphore = asyncio.Semaphore(self.config.max_concurrency)
        gates = {request.request_id: asyncio.Event() for request in requests}
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_s)
        connector = aiohttp.TCPConnector(limit=self.config.pool_size, ssl=self.config.tls_verify)
        headers = dict(self.config.headers)
        if self.config.api_key: headers["Authorization"] = f"Bearer {self.config.api_key}"
        async with aiohttp.ClientSession(timeout=timeout, connector=connector, headers=headers) as session:
            tasks = [asyncio.create_task(self._guarded_scheduled(
                         session, request, epoch, semaphore,
                         gates.get(request.parent_request_id) if request.parent_request_id else None,
                         gates[request.request_id]))
                     for request in requests]
            events: list[RequestEvent] = []
            try:
                for future in asyncio.as_completed(tasks): events.append(await future)
            except asyncio.CancelledError:
                for task in tasks: task.cancel()
                completed = await asyncio.gather(*tasks, return_exceptions=True)
                events.extend(x for x in completed if isinstance(x, RequestEvent) and x not in events)
                events.sort(key=lambda x: x.sequence_no)
                raise BenchmarkCancelled(events, {"wall_start_utc": wall_start.isoformat(),
                    "wall_end_utc": datetime.now(timezone.utc).isoformat(),
                    "monotonic_start_ns": epoch, "monotonic_end_ns": time.monotonic_ns()})
        events.sort(key=lambda x: x.sequence_no)
        bounds = {"wall_start_utc": wall_start.isoformat(),
                  "wall_end_utc": datetime.now(timezone.utc).isoformat(),
                  "monotonic_start_ns": epoch, "monotonic_end_ns": time.monotonic_ns()}
        return events, bounds

    async def _guarded_scheduled(self, session: Any, request: WorkloadRequest, epoch: int,
                                 semaphore: asyncio.Semaphore, parent_gate: asyncio.Event | None,
                                 own_gate: asyncio.Event) -> RequestEvent:
        try:
            return await self._scheduled(session, request, epoch, semaphore, parent_gate)
        finally:
            own_gate.set()

    async def _scheduled(self, session: Any, request: WorkloadRequest, epoch: int,
                         semaphore: asyncio.Semaphore,
                         parent_gate: asyncio.Event | None = None) -> RequestEvent:
        event = RequestEvent(request.request_id, request.sequence_no,
            input_tokens=request.input_tokens, max_output_tokens=request.max_output_tokens)
        enqueued = time.monotonic_ns()
        if request.scheduled_offset_s is not None:
            deadline = epoch + round(request.scheduled_offset_s * 1e9); event.scheduled_ns = deadline
            delay = (deadline - time.monotonic_ns()) / 1e9
            if delay > 0: await asyncio.sleep(delay)
            event.scheduler_wakeup_ns = time.monotonic_ns()
            event.scheduler_lag_ns = max(0, event.scheduler_wakeup_ns - deadline)
        if parent_gate is not None:
            await parent_gate.wait()
        async with semaphore:
            event.admitted_ns = time.monotonic_ns()
            baseline = event.scheduler_wakeup_ns if event.scheduler_wakeup_ns is not None else enqueued
            event.client_queue_delay_ns = max(0, event.admitted_ns - baseline)
            for attempt in range(self.config.retries + 1):
                event.retries = attempt
                if attempt: self._reset_attempt(event)
                try:
                    await self._send_once(session, request, event)
                    if event.error_type == "http_error" and attempt < self.config.retries:
                        await asyncio.sleep(min(.1 * 2**attempt, 1.0)); continue
                    break
                except asyncio.TimeoutError as exc:
                    event.error_type, event.error_message = "timeout", str(exc)
                except SSEParseError as exc:
                    event.error_type, event.error_message = "parse_error", str(exc)
                except Exception as exc:
                    # aiohttp transport exceptions intentionally remain classifiable.
                    event.error_type, event.error_message = "transport_error", f"{type(exc).__name__}: {exc}"
                if attempt < self.config.retries: await asyncio.sleep(min(.1 * 2**attempt, 1.0))
            if event.request_end_ns is None: event.request_end_ns = time.monotonic_ns()
            return event

    @staticmethod
    def _reset_attempt(event: RequestEvent) -> None:
        event.headers_ns = event.first_content_ns = event.request_end_ns = None
        event.chunk_times_ns = []; event.http_status = None
        event.error_type = event.error_message = None
        event.response_text = ""; event.response_truncated = False
        event.server_usage = None; event.output_tokens = 0

    def _payload(self, request: WorkloadRequest) -> tuple[str, dict[str, Any]]:
        backend = get_backend(self.config.backend)
        path = backend.chat_path if request.endpoint_kind == "chat" else backend.completions_path
        payload: dict[str, Any] = {"model": self.config.model, "stream": self.config.stream,
            "max_tokens": request.max_output_tokens, **request.sampling}
        if request.endpoint_kind == "chat": payload["messages"] = request.messages
        else: payload["prompt"] = request.prompt
        if self.config.stream: payload["stream_options"] = {"include_usage": True}
        return path, payload

    async def _send_once(self, session: Any, request: WorkloadRequest, event: RequestEvent) -> None:
        path, payload = self._payload(request)
        if event.request_start_ns is None: event.request_start_ns = time.monotonic_ns()
        async with session.post(self.config.base_url.rstrip("/") + path, json=payload) as response:
            event.headers_ns, event.http_status = time.monotonic_ns(), response.status
            if response.status < 200 or response.status >= 300:
                text = await response.text(); self._store_response(event, text)
                event.error_type, event.error_message = "http_error", f"HTTP {response.status}"
                event.request_end_ns = time.monotonic_ns(); return
            if self.config.stream: await self._read_stream(response, request, event)
            else: await self._read_json(response, request, event)
            self._apply_server_usage(event)
            event.request_end_ns = time.monotonic_ns()

    @staticmethod
    def _apply_server_usage(event: RequestEvent) -> None:
        """Prefer authoritative server token counts over client estimates."""
        if not isinstance(event.server_usage, dict):
            return
        prompt_tokens = event.server_usage.get("prompt_tokens")
        completion_tokens = event.server_usage.get("completion_tokens")
        if isinstance(prompt_tokens, int) and prompt_tokens >= 0:
            event.input_tokens = prompt_tokens
        if isinstance(completion_tokens, int) and completion_tokens >= 0:
            event.output_tokens = completion_tokens

    async def _read_stream(self, response: Any, request: WorkloadRequest, event: RequestEvent) -> None:
        pieces: list[str] = []; saw_done = False
        try:
            async for sse_event in iter_sse(response.content.iter_chunked(1024)):
                now = time.monotonic_ns()
                if sse_event.data.strip() == "[DONE]": saw_done = True; break
                data = parse_data_json(sse_event)
                if isinstance(data.get("usage"), dict): event.server_usage = data["usage"]
                choices = data.get("choices") or []
                if not choices: continue
                first = choices[0]
                if request.endpoint_kind == "chat":
                    delta = first.get("delta") or {}
                    # Reasoning models can stream their initial tokens separately
                    # from the final answer.  Both fields are generated content and
                    # therefore must contribute to TTFT, chunk timing, and TPOT.
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    content = reasoning + (delta.get("content") or "")
                else:
                    content = first.get("text")
                if content:
                    if event.first_content_ns is None: event.first_content_ns = now
                    event.chunk_times_ns.append(now); pieces.append(str(content))
        finally:
            text = "".join(pieces); self._store_response(event, text)
            event.output_tokens = len(self.tokenizer.encode(text, add_special_tokens=False))
        if not saw_done: raise SSEParseError("stream ended before data: [DONE]")

    async def _read_json(self, response: Any, request: WorkloadRequest, event: RequestEvent) -> None:
        try: data = await response.json()
        except (json.JSONDecodeError, ValueError) as exc: raise SSEParseError(f"invalid response JSON: {exc}") from exc
        choices = data.get("choices") or []
        if not choices: raise SSEParseError("response has no choices")
        first = choices[0]
        if request.endpoint_kind == "chat":
            message = first.get("message") or {}
            reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
            text = reasoning + (message.get("content") or "")
        else:
            text = first.get("text") or ""
        if text: event.first_content_ns = time.monotonic_ns(); event.chunk_times_ns.append(event.first_content_ns)
        event.server_usage = data.get("usage"); self._store_response(event, str(text))
        event.output_tokens = len(self.tokenizer.encode(str(text), add_special_tokens=False))

    def _store_response(self, event: RequestEvent, text: str) -> None:
        event.response_truncated = len(text) > self.config.response_max_chars
        event.response_text = text[:self.config.response_max_chars]
