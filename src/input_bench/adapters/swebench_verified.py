from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from input_bench.adapters.base import iter_parquet
from input_bench.schema import SemanticSample

DEFAULT_TEMPLATE = "You are fixing a software repository. Resolve the following issue:\n\n{problem_statement}"


class SWEBenchVerifiedAdapter:
    name = "swebench-verified"
    def __init__(self, source: str | Path, template: str = DEFAULT_TEMPLATE,
                 include_repo: bool = False, include_base_commit: bool = False,
                 include_hints: bool = False, **_: Any):
        root = Path(source); self.paths = sorted(root.glob("*.parquet")) if root.is_dir() else [root]
        self.template, self.include_repo = template, include_repo
        self.include_base_commit, self.include_hints = include_base_commit, include_hints
        self.skip_reasons = Counter()
        self.manifest_metadata = {"template_name": "swebench_issue_v1",
            "template_sha256": hashlib.sha256(template.encode()).hexdigest(),
            "include_repo": include_repo, "include_base_commit": include_base_commit,
            "include_hints": include_hints}

    def iter_samples(self, limit: int | None = None) -> Iterator[SemanticSample]:
        emitted = 0
        for path, row_no, row in iter_parquet(self.paths):
            rid, problem = str(row.get("instance_id", row_no)), row.get("problem_statement")
            if not problem: self.skip_reasons["empty_problem_statement"] += 1; continue
            pieces = [self.template.format(problem_statement=problem)]
            if self.include_repo: pieces.append(f"Repository: {row.get('repo', '')}")
            if self.include_base_commit: pieces.append(f"Base commit: {row.get('base_commit', '')}")
            if self.include_hints and row.get("hints_text"): pieces.append(f"Hints: {row['hints_text']}")
            protected = {k: row.get(k) for k in
                         ["patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS"] if k in row}
            yield SemanticSample(f"swebench-verified:{rid}", "swebench-verified",
                "agent_coding", "single", prompt="\n\n".join(pieces),
                metadata={"source_record_id": rid, "source_row": row_no,
                          "reference": protected})
            emitted += 1
            if limit is not None and emitted >= limit: return
