from __future__ import annotations

import hashlib
import json
import zipfile
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from input_bench.schema import SemanticSample

SUMMARY_SUBSETS = {"gov_report", "qmsum", "multi_news"}
TEMPLATES = {
    "gov_report": "Write a concise summary of the following government report.\n\n{context}",
    "qmsum": "Given the meeting transcript, answer the query with a focused summary.\nQuery: {input}\n\nTranscript:\n{context}",
    "multi_news": "Summarize the following collection of news articles.\n\n{context}",
    "default": "Use the context to answer the task.\nTask: {input}\n\nContext:\n{context}",
}


class LongBenchAdapter:
    name = "longbench"
    def __init__(self, source: str | Path, subset: str | None = None, **_: Any):
        self.source, self.subset = Path(source), subset
        self.skip_reasons = Counter()
        names = self.list_subsets()
        if subset is not None and subset not in names:
            raise ValueError(f"LongBench subset {subset!r} not found; available: {', '.join(names)}")
        selected = [subset] if subset else names
        template_hashes = {x: hashlib.sha256(TEMPLATES.get(x.removesuffix("_e"), TEMPLATES["default"]).encode()).hexdigest()
                           for x in selected}
        self.manifest_metadata = {"subsets": selected, "template_name": "longbench_v1",
                                  "template_sha256": template_hashes}

    def _members(self) -> dict[str, str]:
        with zipfile.ZipFile(self.source) as archive:
            result = {}
            for name in sorted(archive.namelist()):
                if name.endswith(".jsonl"):
                    result[Path(name).stem] = name
            return result

    def list_subsets(self) -> list[str]: return sorted(self._members())

    def iter_samples(self, limit: int | None = None) -> Iterator[SemanticSample]:
        emitted, members = 0, self._members()
        selected = [self.subset] if self.subset else sorted(members)
        with zipfile.ZipFile(self.source) as archive:
            for subset in selected:
                base_subset = subset.removesuffix("_e")
                template = TEMPLATES.get(base_subset, TEMPLATES["default"])
                with archive.open(members[subset]) as raw:
                    for line_no, encoded in enumerate(raw, 1):
                        try: row = json.loads(encoded)
                        except json.JSONDecodeError: self.skip_reasons["malformed_json"] += 1; continue
                        context, task = str(row.get("context", "")), str(row.get("input", ""))
                        if not context: self.skip_reasons["empty_context"] += 1; continue
                        prompt = template.format(context=context, input=task)
                        answers = row.get("answers") or []
                        reference = str(answers[0]) if answers else None
                        rid = str(row.get("id", line_no))
                        meta = {k: row.get(k) for k in ["length", "dataset", "language", "all_classes"] if k in row}
                        meta.update({"subset": subset, "source_record_id": rid, "source_row": line_no,
                                     "task_type": "summarization" if base_subset in SUMMARY_SUBSETS else "long_context"})
                        yield SemanticSample(f"longbench:{subset}:{rid}", "longbench",
                            "summarization" if base_subset in SUMMARY_SUBSETS else "long_context",
                            "single", prompt=prompt, reference_output=reference, metadata=meta)
                        emitted += 1
                        if limit is not None and emitted >= limit: return
