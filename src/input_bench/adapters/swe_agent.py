from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from input_bench.adapters.base import iter_parquet, normalized_message, parse_json_field
from input_bench.schema import Message, SemanticSample


class SWEAgentTrajectoriesAdapter:
    name = "swe-agent"
    def __init__(self, source: str | Path, mode: str = "session_replay", **_: Any):
        if mode not in {"independent_turn", "session_replay"}: raise ValueError("invalid mode")
        root = Path(source); self.paths = sorted(root.glob("*.parquet")) if root.is_dir() else [root]
        self.mode, self.skip_reasons, self.manifest_metadata = mode, Counter(), {}

    def iter_samples(self, limit: int | None = None) -> Iterator[SemanticSample]:
        emitted = 0
        for path, row_no, row in iter_parquet(self.paths):
            rid = str(row.get("instance_id", row_no))
            try: trajectory = parse_json_field(row.get("trajectory"), source=path, record_id=rid, field="trajectory")
            except ValueError: self.skip_reasons["malformed_trajectory"] += 1; continue
            if isinstance(trajectory, dict):
                trajectory = trajectory.get("messages", trajectory.get("trajectory", []))
            if not isinstance(trajectory, list): self.skip_reasons["malformed_trajectory"] += 1; continue
            context: list[dict[str, Any]] = []
            system_prompt = row.get("system_prompt")
            if system_prompt: context.append({"role": "system", "content": system_prompt})
            session_id, parent, turn = f"swe-agent:{rid}", None, 0
            step_metadata: list[dict[str, Any]] = []
            for step in trajectory:
                if not isinstance(step, dict): continue
                normalized = normalized_message(step, {"ai": "assistant", "assistant": "assistant",
                    "human": "user", "user": "user", "system": "system",
                    "observation": "tool", "tool": "tool"})
                content = normalized.get("content")
                if normalized["role"] == "system" and not content: content = step.get("system_prompt")
                msg = {"role": normalized["role"], "content": content or ""}
                step_metadata.append({k: v for k, v in step.items()
                                      if k not in {"role", "content", "text", "value"}})
                role = msg["role"]
                if role == "assistant":
                    if not context: self.skip_reasons["assistant_without_context"] += 1; continue
                    sid = f"{session_id}:turn:{turn}"
                    meta = {k: row.get(k) for k in ["mask", "cutoff_date", "system_prompt",
                        "target", "model_name", "exit_status"] if k in row}
                    meta.update({"source_record_id": rid, "source_row": row_no,
                                 "trajectory_step_metadata": step_metadata.copy()})
                    yield SemanticSample(sid, "swe-agent", "agent_coding", self.mode,
                        session_id=session_id, turn_id=turn,
                        parent_sample_id=parent if self.mode == "session_replay" else None,
                        messages=[Message.from_dict(x) for x in context],
                        reference_output=str(msg.get("content", "")), metadata=meta)
                    emitted += 1; parent = sid; turn += 1
                    if limit is not None and emitted >= limit: return
                context.append(msg)
            if turn == 0: self.skip_reasons["no_assistant_turn"] += 1
