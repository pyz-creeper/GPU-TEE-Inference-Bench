from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from input_bench.adapters.base import iter_json_array, normalized_message
from input_bench.schema import Message, SemanticSample


class ShareGPTAdapter:
    name = "sharegpt"
    def __init__(self, source: str | Path, mode: str = "independent_turn",
                 max_turns: int | None = None, **_: Any):
        if mode not in {"independent_turn", "session_replay"}:
            raise ValueError("ShareGPT mode must be independent_turn or session_replay")
        self.source, self.mode, self.max_turns = Path(source), mode, max_turns
        self.skip_reasons: Counter[str] = Counter()
        self.manifest_metadata: dict[str, Any] = {}

    def iter_samples(self, limit: int | None = None) -> Iterator[SemanticSample]:
        emitted = 0
        for row_no, row in iter_json_array(self.source):
            record_id = str(row.get("id", row_no))
            raw_messages = row.get("conversations")
            if not isinstance(raw_messages, list):
                self.skip_reasons["missing_conversations"] += 1
                continue
            messages = [normalized_message(x, {"human": "user", "gpt": "assistant",
                                                "system": "system"}) for x in raw_messages]
            if any(m["role"] not in {"system", "user", "assistant"} for m in messages):
                self.skip_reasons["invalid_role"] += 1; continue
            if any(not isinstance(m["content"], str) or not m["content"].strip() for m in messages):
                self.skip_reasons["empty_content"] += 1; continue
            first_non_system = next((i for i, m in enumerate(messages) if m["role"] != "system"), len(messages))
            if any(m["role"] == "system" for m in messages[first_non_system:]):
                self.skip_reasons["invalid_role_order"] += 1; continue
            non_system = [m["role"] for m in messages if m["role"] != "system"]
            if not non_system or non_system[0] != "user" or any(
                    role != ("user" if i % 2 == 0 else "assistant")
                    for i, role in enumerate(non_system)):
                self.skip_reasons["invalid_role_order"] += 1; continue
            session_id = f"sharegpt:{record_id}"
            assistant_index, parent = 0, None
            for i, message in enumerate(messages):
                if message["role"] != "assistant": continue
                if self.max_turns is not None and assistant_index >= self.max_turns: break
                sid = f"{session_id}:turn:{assistant_index}"
                yield SemanticSample(sid, "sharegpt", "chat", self.mode,
                    session_id=session_id, turn_id=assistant_index,
                    parent_sample_id=parent if self.mode == "session_replay" else None,
                    messages=[Message.from_dict(x) for x in messages[:i]],
                    reference_output=message["content"],
                    metadata={"source_record_id": record_id, "source_row": row_no})
                emitted += 1; parent = sid; assistant_index += 1
                if limit is not None and emitted >= limit: return
