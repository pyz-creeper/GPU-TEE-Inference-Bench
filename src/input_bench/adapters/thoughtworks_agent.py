from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from input_bench.adapters.base import iter_parquet, normalized_message, parse_json_field
from input_bench.schema import Message, SemanticSample


class ThoughtworksAgenticAdapter:
    name = "thoughtworks"
    def __init__(self, source: str | Path, mode: str = "session_replay", **_: Any):
        if mode not in {"independent_turn", "session_replay"}: raise ValueError("invalid mode")
        root = Path(source); self.paths = sorted(root.glob("*.parquet")) if root.is_dir() else [root]
        self.mode, self.skip_reasons, self.manifest_metadata = mode, Counter(), {}

    def iter_samples(self, limit: int | None = None) -> Iterator[SemanticSample]:
        emitted = 0
        for path, row_no, row in iter_parquet(self.paths):
            rid = str(row.get("session_id", row_no))
            try: raw = parse_json_field(row.get("messages_json"), source=path, record_id=rid, field="messages_json")
            except ValueError: self.skip_reasons["malformed_messages_json"] += 1; continue
            if not isinstance(raw, list): self.skip_reasons["malformed_messages_json"] += 1; continue
            messages = [normalized_message(m, {"ai": "assistant", "human": "user",
                "assistant": "assistant", "user": "user", "system": "system",
                "tool": "tool", "function": "tool"}) for m in raw if isinstance(m, dict)]
            for message in messages:
                # The published data stores OpenAI structured fields as JSON strings.
                # Normalize them back while keeping every other extension untouched.
                for old, new in (("tool_calls_json", "tool_calls"), ("function_call_json", "function_call")):
                    if old in message:
                        value = message.pop(old)
                        if value is not None:
                            try: message[new] = parse_json_field(value, source=path, record_id=rid, field=old)
                            except ValueError: message[old] = value
            session_id, parent, turn = f"thoughtworks:{rid}", None, 0
            base_meta = {k: v for k, v in row.items() if k != "messages_json"}
            for i, msg in enumerate(messages):
                if msg["role"] != "assistant": continue
                if not messages[:i]: self.skip_reasons["assistant_without_context"] += 1; continue
                sid = f"{session_id}:turn:{turn}"
                yield SemanticSample(sid, "thoughtworks", "agent_coding", self.mode,
                    session_id=session_id, turn_id=turn,
                    parent_sample_id=parent if self.mode == "session_replay" else None,
                    messages=[Message.from_dict(x) for x in messages[:i]],
                    reference_output=str(msg.get("content", "")),
                    metadata={**base_meta, "source_record_id": rid, "source_row": row_no})
                emitted += 1; parent = sid; turn += 1
                if limit is not None and emitted >= limit: return
            if turn == 0: self.skip_reasons["no_assistant_turn"] += 1
