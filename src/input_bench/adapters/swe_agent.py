from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from input_bench.adapters.base import iter_jsonl, iter_parquet, normalized_message, parse_json_field
from input_bench.schema import Message, SemanticSample

NORMALIZATION = "recorded-tools-as-text-v1"


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def text_message(message: dict[str, Any]) -> dict[str, str]:
    """Freeze recorded tools as inert text. Unknown/non-text content rejects the session."""
    role, content = message["role"], message.get("content") or ""
    if not isinstance(content, str):
        raise ValueError("non_text_content")
    if role in {"tool", "function"}:
        origin = {k: message[k] for k in ("name", "tool_call_id") if message.get(k)}
        return {"role": "user", "content": f"[recorded {role} result {canonical(origin)}]\n{content}"}
    if role not in {"system", "user", "assistant"}:
        raise ValueError("invalid_role")
    for key in ("tool_calls", "function_call", "tools", "functions"):
        if message.get(key):
            content += f"\n[recorded {key}]\n{canonical(message[key])}"
    return {"role": role, "content": content}


class SWEAgentTrajectoriesAdapter:
    name = "swe-agent"

    def __init__(self, source: str | Path, mode: str = "session_replay",
                 revision: str = "unspecified", normalization: str | None = None,
                 selected_rows: dict[str, list[int]] | None = None, **_: Any):
        if mode not in {"independent_turn", "session_replay"}:
            raise ValueError("invalid mode")
        if normalization not in {None, NORMALIZATION}:
            raise ValueError("unsupported normalization")
        root = Path(source)
        self.root = root if root.is_dir() else root.parent
        self.paths = sorted(root.glob("*.parquet")) if root.is_dir() else [root]
        self.mode, self.revision, self.normalization = mode, revision, normalization
        self.selected_rows = selected_rows
        self.skip_reasons: Counter = Counter()
        self.manifest_metadata = {"identity_version": "trajectory-attempt-v1", "revision": revision,
                                  "normalization": normalization}
        self.exclusions: list[dict[str, Any]] = []

    def _rows(self, path: Path):
        """Read selected Parquet row groups only; row positions stay one-based."""
        if self.selected_rows is None:
            yield from iter_parquet([path])
            return
        import pyarrow.parquet as pq
        wanted = set(self.selected_rows.get(path.relative_to(self.root).as_posix(), []))
        if not wanted:
            return
        parquet = pq.ParquetFile(path)
        offset = 0
        for group in range(parquet.num_row_groups):
            size = parquet.metadata.row_group(group).num_rows
            positions = sorted(n for n in wanted if offset < n <= offset + size)
            if positions:
                table = parquet.read_row_group(group).take([n - offset - 1 for n in positions])
                for n, row in zip(positions, table.to_pylist()):
                    yield path, n, row
            offset += size

    def iter_sessions(self) -> Iterator[list[SemanticSample]]:
        for path in self.paths:
            rows = ((path, n, row) for n, row in iter_jsonl(path)) if path.suffix == ".jsonl" else self._rows(path)
            for path, row_no, row in rows:
                relative = path.relative_to(self.root).as_posix()
                digest = hashlib.sha256(canonical(row).encode()).hexdigest()
                identity = canonical([self.revision, relative, row_no, digest])
                session_id = "swe-agent:" + hashlib.sha256(identity.encode()).hexdigest()
                metadata = {k: row.get(k) for k in ("instance_id", "model_name", "target", "exit_status")}
                metadata.update(source_record_id=str(row.get("instance_id", row_no)), source_row=row_no,
                                source_file=relative, source_row_sha256=digest, dataset_revision=self.revision)
                try:
                    trajectory = parse_json_field(row.get("trajectory"), source=path,
                                                  record_id=session_id, field="trajectory")
                    if not isinstance(trajectory, list):
                        raise ValueError("malformed_trajectory")
                    context: list[dict[str, Any]] = []
                    if row.get("system_prompt"):
                        context.append({"role": "system", "content": row["system_prompt"]})
                    # Tool definitions are historical descriptions, never executable payload fields.
                    if self.normalization:
                        for key in ("tools", "functions"):
                            if row.get(key):
                                context.append({"role": "system", "content": f"[recorded {key}]\n{canonical(row[key])}"})
                    samples: list[SemanticSample] = []
                    parent = None
                    for step in trajectory:
                        if not isinstance(step, dict):
                            raise ValueError("malformed_step")
                        msg = normalized_message(step, {"ai": "assistant", "human": "user", "observation": "tool"})
                        if msg["role"] == "system" and not msg.get("content"):
                            msg["content"] = step.get("system_prompt") or ""
                        if self.normalization:
                            msg = text_message(msg)
                        if msg["role"] not in {"system", "user", "assistant", "tool", "function"}:
                            raise ValueError("invalid_role")
                        if msg["role"] == "assistant":
                            if not any(m["role"] == "user" for m in context):
                                raise ValueError("assistant_without_context")
                            turn = len(samples)
                            sid = f"{session_id}:turn:{turn}"
                            samples.append(SemanticSample(sid, self.name, "agent_coding", self.mode,
                                session_id=session_id, turn_id=turn,
                                parent_sample_id=parent if self.mode == "session_replay" else None,
                                messages=[Message.from_dict(m) for m in context],
                                reference_output=str(msg.get("content", "")), metadata=dict(metadata)))
                            parent = sid
                        context.append(msg)
                    if not samples:
                        raise ValueError("no_assistant_turn")
                    yield samples
                except ValueError as exc:
                    reason = str(exc) if str(exc) in {"non_text_content", "invalid_role", "malformed_step",
                        "assistant_without_context", "no_assistant_turn", "malformed_trajectory"} else "malformed_trajectory"
                    self.skip_reasons[reason] += 1
                    self.exclusions.append({"session_id": session_id, **metadata, "reason": reason})

    def iter_samples(self, limit: int | None = None) -> Iterator[SemanticSample]:
        emitted = 0
        for samples in self.iter_sessions():
            for sample in samples:
                yield sample
                emitted += 1
                if limit is not None and emitted >= limit:
                    return
