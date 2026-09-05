from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

SCHEMA_VERSION = "1.0"


@dataclass(slots=True)
class Message:
    role: str
    content: Any = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = {"role": self.role, "content": self.content}
        result.update(self.extra)
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Message":
        known = {"role", "content"}
        return cls(str(value.get("role", "")), value.get("content", ""),
                   {k: v for k, v in value.items() if k not in known})


@dataclass(slots=True)
class SemanticSample:
    sample_id: str
    source: str
    workload_class: str
    mode: str
    session_id: str | None = None
    turn_id: int | None = None
    parent_sample_id: str | None = None
    prompt: str | None = None
    messages: list[Message] | None = None
    reference_output: str | None = None
    requested_output_tokens: int | None = None
    think_time_s: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (self.prompt is None) == (self.messages is None):
            raise ValueError("exactly one of prompt and messages must be set")


@dataclass(slots=True)
class WorkloadRequest:
    request_id: str
    sequence_no: int
    source: str
    workload_class: str
    endpoint_kind: Literal["chat", "completions"]
    input_tokens: int
    max_output_tokens: int
    session_id: str | None = None
    parent_request_id: str | None = None
    scheduled_offset_s: float | None = None
    messages: list[dict[str, Any]] | None = None
    prompt: str | None = None
    sampling: dict[str, Any] = field(default_factory=lambda: {
        "temperature": 0.0, "top_p": 1.0, "seed": 1})
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version {self.schema_version!r}")
        if (self.prompt is None) == (self.messages is None):
            raise ValueError("exactly one of prompt and messages must be set")
        if self.sequence_no < 0 or self.input_tokens < 0 or self.max_output_tokens < 1:
            raise ValueError("invalid request sequence/token count")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "WorkloadRequest":
        return cls(**value)
