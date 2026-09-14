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
from input_bench.targets import TargetCapabilities, api_url


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
    cache_salt: str | None = None
    target: TargetCapabilities | None = None
    session_concurrency: int | None = None
    stop_on_error: bool = False
    enforce_reported_output_cap: bool = False


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
    output_tokens: int | None = None
    input_tokens: int = 0
    max_output_tokens: int = 0
    session_id: str | None = None
    streaming: bool = True
    reference_input_tokens: int | None = None
    client_output_tokens: int | None = None
    output_token_source: str = "unknown"
    answer_text: str = ""
    reasoning_text: str = ""
    first_answer_ns: int | None = None
    first_reasoning_ns: int | None = None
    finish_reason: str | None = None
    server_request_id: str | None = None
    response_model: str | None = None
    backoff_s: float = 0.0
    attempt_history: list[dict[str, Any]] = field(default_factory=list)


    def to_dict(self) -> dict[str, Any]: return asdict(self)


class BenchmarkCancelled(asyncio.CancelledError):
    def __init__(self, events: list[RequestEvent], bounds: dict[str, Any]):
        super().__init__("benchmark cancelled")
        self.events, self.bounds = events, bounds


class BenchmarkSender:
    def __init__(self, config: SenderConfig, tokenizer: Tokenizer | None = None, on_event=None):
        self.config, self.tokenizer, self.on_event = config, tokenizer, on_event
        if config.max_concurrency < 1 or config.pool_size < 1 or config.timeout_s <= 0 or config.retries < 0:
            raise ValueError("invalid sender limits")
        if config.session_concurrency is not None and config.session_concurrency < 1:
            raise ValueError("session_concurrency must be positive")

    async def run(self, requests: list[WorkloadRequest]) -> tuple[list[RequestEvent], dict[str, Any]]:
        try: import aiohttp
        except ImportError as exc: raise RuntimeError("aiohttp is required to run workloads") from exc
        # Payload validation and session grouping are offline, before the timer.
        groups = {}
        seen = set()
        for request in requests:
            if request.request_id in seen or (request.parent_request_id and request.parent_request_id not in seen):
                raise ValueError("invalid request IDs/dependencies")
            seen.add(request.request_id)
            self._payload(request)
            group = groups.setdefault(request.session_id or request.request_id, [])
            if self.config.session_concurrency is not None:
                if request.scheduled_offset_s is not None or request.parent_request_id != (group[-1].request_id if group else None):
                    raise ValueError("session scheduler requires complete closed-loop chains")
            group.append(request)
        api_url(self.config.base_url)
        events: list[RequestEvent] = []
        wall_start = datetime.now(timezone.utc); epoch = time.monotonic_ns()
        semaphore = asyncio.Semaphore(self.config.max_concurrency)
        gates = {request.request_id: asyncio.Event() for request in requests}
        stopped = False
        cancelled = False

        def collect(event):
            nonlocal stopped
            events.append(event)
            if self.on_event is not None:
                self.on_event(event)
            if event.error_type == "cancelled" or (self.config.stop_on_error and event.error_type):
                stopped = True

        timeout = aiohttp.ClientTimeout(total=self.config.timeout_s)
        connector = aiohttp.TCPConnector(limit=self.config.pool_size, ssl=self.config.tls_verify)
        headers = dict(self.config.headers)
        if self.config.api_key: headers["Authorization"] = f"Bearer {self.config.api_key}"
        async with aiohttp.ClientSession(timeout=timeout, connector=connector, headers=headers,
                                        trust_env=True) as session:
            async def one(request):
                event = await self._guarded_scheduled(session, request, epoch, semaphore,
                    gates.get(request.parent_request_id), gates[request.request_id])
                collect(event)

            queue = iter(groups.values())
            async def worker():
                for group in queue:
                    if stopped: return
                    for request in group:
                        if stopped: return
                        await one(request)

            if self.config.session_concurrency is not None:
                tasks = [asyncio.create_task(worker()) for _ in range(min(len(groups), self.config.session_concurrency))]
            else:
                tasks = [asyncio.create_task(one(request)) for request in requests]
            try:
                await asyncio.gather(*tasks)
            except asyncio.CancelledError:
                cancelled = True
                for task in tasks: task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
            except Exception:
                for task in tasks: task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
        bounds = {"wall_start_utc": wall_start.isoformat(),
                  "wall_end_utc": datetime.now(timezone.utc).isoformat(),
                  "monotonic_start_ns": epoch, "monotonic_end_ns": time.monotonic_ns(),
                  "planned_requests": len(requests), "cancelled": cancelled, "stopped_on_error": stopped}
        finished = {e.request_id for e in events}
        for request in requests:
            if request.request_id not in finished:
                event = self._event(request)
                event.error_type = "cancelled" if cancelled else "not_sent"
                collect(event)
        events.sort(key=lambda x: x.sequence_no)
        if cancelled:
            raise BenchmarkCancelled(events, bounds)
        return events, bounds

    def _event(self, request):
        return RequestEvent(request.request_id, request.sequence_no,
            input_tokens=request.input_tokens, reference_input_tokens=request.input_tokens,
            max_output_tokens=request.max_output_tokens, session_id=request.session_id,
            streaming=self.config.stream)

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
        event = self._event(request)
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
                    if not (event.error_type == "http_error" and
                            (event.http_status == 429 or event.http_status >= 500)):
                        break
                except asyncio.CancelledError:
                    event.error_type = "cancelled"
                    event.request_end_ns = time.monotonic_ns()
                    return event
                except asyncio.TimeoutError as exc:
                    event.error_type, event.error_message = "timeout", str(exc)
                except SSEParseError as exc:
                    event.error_type, event.error_message = "parse_error", str(exc)
                except Exception as exc:
                    # aiohttp transport exceptions intentionally remain classifiable.
                    event.error_type, event.error_message = "transport_error", f"{type(exc).__name__}: {exc}"
                finally:
                    event.attempt_history.append({"attempt": attempt + 1, "http_status": event.http_status,
                                                  "error_type": event.error_type})
                if attempt < self.config.retries:
                    backoff = min(.1 * 2**attempt, 1.0)
                    event.backoff_s += backoff
                    await asyncio.sleep(backoff)
            if event.request_end_ns is None: event.request_end_ns = time.monotonic_ns()
            if event.error_message:
                event.error_message = self._redact(event.error_message)
            return event

    @staticmethod
    def _reset_attempt(event: RequestEvent) -> None:
        event.headers_ns = event.first_content_ns = event.request_end_ns = None
        event.chunk_times_ns = []; event.http_status = None
        event.error_type = event.error_message = None
        event.response_text = ""; event.response_truncated = False
        event.server_usage = None; event.output_tokens = None
        event.client_output_tokens = None; event.output_token_source = "unknown"
        event.answer_text = event.reasoning_text = ""
        event.first_answer_ns = event.first_reasoning_ns = None
        event.finish_reason = None
        event.server_request_id = event.response_model = None

    def _payload(self, request: WorkloadRequest) -> tuple[str, dict[str, Any]]:
        if self.config.target is not None:
            return "/v1/chat/completions", self.config.target.payload(request, self.config.model, self.config.stream)
        backend = get_backend(self.config.backend)
        path = backend.chat_path if request.endpoint_kind == "chat" else backend.completions_path
        payload: dict[str, Any] = {"model": self.config.model, "stream": self.config.stream,
            "max_tokens": request.max_output_tokens, **request.sampling}
        if self.config.cache_salt:
            payload["cache_salt"] = self.config.cache_salt
        if request.endpoint_kind == "chat": payload["messages"] = request.messages
        else: payload["prompt"] = request.prompt
        if self.config.stream: payload["stream_options"] = {"include_usage": True}
        return path, payload

    async def _send_once(self, session: Any, request: WorkloadRequest, event: RequestEvent) -> None:
        path, payload = self._payload(request)
        if event.request_start_ns is None: event.request_start_ns = time.monotonic_ns()
        async with session.post(api_url(self.config.base_url, path), json=payload, allow_redirects=False) as response:
            event.headers_ns, event.http_status = time.monotonic_ns(), response.status
            event.server_request_id = response.headers.get("x-request-id")
            if response.status < 200 or response.status >= 300:
                text = await response.text(); self._store_response(event, text)
                event.error_type, event.error_message = "http_error", f"HTTP {response.status}"
                event.request_end_ns = time.monotonic_ns(); return
            if self.config.stream: await self._read_stream(response, request, event)
            else: await self._read_json(response, request, event)
            self._apply_server_usage(event)
            if (self.config.enforce_reported_output_cap and event.output_tokens is not None
                    and event.output_tokens > request.max_output_tokens + 10):
                event.error_type = "reported_output_exceeds_cap"
                event.error_message = "provider completion usage exceeds logical output cap (10-token tolerance)"
            event.request_end_ns = time.monotonic_ns()

    @staticmethod
    def _apply_server_usage(event: RequestEvent) -> None:
        """Prefer authoritative server token counts over client estimates."""
        if not isinstance(event.server_usage, dict):
            return
        prompt_tokens = event.server_usage.get("prompt_tokens")
        completion_tokens = event.server_usage.get("completion_tokens")
        if type(prompt_tokens) is int and prompt_tokens >= 0:
            event.input_tokens = prompt_tokens
        if type(completion_tokens) is int and completion_tokens >= 0:
            event.output_tokens = completion_tokens
            event.output_token_source = "server_usage"

    async def _read_stream(self, response: Any, request: WorkloadRequest, event: RequestEvent) -> None:
        pieces: list[str] = []; saw_done = False
        try:
            async for sse_event in iter_sse(response.content.iter_chunked(1024)):
                now = time.monotonic_ns()
                if sse_event.data.strip() == "[DONE]": saw_done = True; break
                data = parse_data_json(sse_event)
                if isinstance(data.get("usage"), dict): event.server_usage = data["usage"]
                event.server_request_id = event.server_request_id or data.get("id")
                event.response_model = data.get("model") or event.response_model
                if data.get("error"):
                    raise SSEParseError("provider error in stream")
                choices = data.get("choices") or []
                if not choices: continue
                first = choices[0]
                if first.get("finish_reason") is not None: event.finish_reason = first["finish_reason"]
                if request.endpoint_kind == "chat":
                    delta = first.get("delta") or {}
                    # Reasoning models can stream their initial tokens separately
                    # from the final answer.  Both fields are generated content and
                    # therefore must contribute to TTFT, chunk timing, and TPOT.
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    answer = delta.get("content") or ""
                    self._parts(event, answer, reasoning, now)
                    if delta.get("tool_calls") or delta.get("function_call"):
                        event.error_type = "unexpected_tool_call"
                    content = reasoning + answer
                else:
                    content = first.get("text")
                if content:
                    if event.first_content_ns is None: event.first_content_ns = now
                    event.chunk_times_ns.append(now); pieces.append(str(content))
        finally:
            text = "".join(pieces); self._store_response(event, text)
            self._recount(event, text)
            self._apply_server_usage(event)
        if not saw_done: raise SSEParseError("stream ended before data: [DONE]")
        if self.config.target and event.finish_reason is None:
            raise SSEParseError("stream has no finish_reason")
        if self.config.target and not event.error_type:
            if event.finish_reason not in {"stop", "length"}:
                event.error_type = "unexpected_finish_reason"
            elif not pieces:
                event.error_type = "empty_response"

    async def _read_json(self, response: Any, request: WorkloadRequest, event: RequestEvent) -> None:
        try: data = await response.json()
        except (json.JSONDecodeError, ValueError) as exc: raise SSEParseError(f"invalid response JSON: {exc}") from exc
        choices = data.get("choices") or []
        if not choices: raise SSEParseError("response has no choices")
        first = choices[0]
        event.finish_reason = first.get("finish_reason")
        event.server_request_id = event.server_request_id or data.get("id")
        event.response_model = data.get("model")
        if request.endpoint_kind == "chat":
            message = first.get("message") or {}
            reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
            answer = message.get("content") or ""
            self._parts(event, answer, reasoning, None)
            if message.get("tool_calls") or message.get("function_call"):
                event.error_type = "unexpected_tool_call"
            text = reasoning + answer
        else:
            text = first.get("text") or ""
        if text: event.first_content_ns = time.monotonic_ns(); event.chunk_times_ns.append(event.first_content_ns)
        event.server_usage = data.get("usage"); self._store_response(event, str(text))
        self._recount(event, str(text))

    def _redact(self, text):
        for secret in [self.config.api_key, *self.config.headers.values()]:
            if secret: text = text.replace(secret, "<redacted>")
        return text

    def _parts(self, event, answer, reasoning, now):
        if answer and event.first_answer_ns is None: event.first_answer_ns = now
        if reasoning and event.first_reasoning_ns is None: event.first_reasoning_ns = now
        cap = self.config.response_max_chars
        event.answer_text = self._redact((event.answer_text + answer)[:cap])
        event.reasoning_text = self._redact((event.reasoning_text + reasoning)[:cap])

    def _recount(self, event, text):
        if self.tokenizer is not None:
            event.client_output_tokens = len(self.tokenizer.encode(text, add_special_tokens=False))
            event.output_tokens = event.client_output_tokens
            event.output_token_source = "client_recount"

    def _store_response(self, event: RequestEvent, text: str) -> None:
        event.response_truncated = len(text) > self.config.response_max_chars
        event.response_text = self._redact(text)[:self.config.response_max_chars]
