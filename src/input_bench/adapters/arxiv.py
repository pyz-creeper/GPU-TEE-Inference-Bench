from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from input_bench.adapters.base import iter_parquet
from input_bench.schema import SemanticSample

DEFAULT_TEMPLATE = "Summarize the following academic article concisely and accurately.\n\n{article}"


class ArxivSummarizationAdapter:
    name = "arxiv"
    def __init__(self, source: str | Path, split: str | None = None,
                 template: str = DEFAULT_TEMPLATE, **_: Any):
        root = Path(source)
        if root.is_dir():
            pattern = f"{split}-*.parquet" if split else "*.parquet"
            self.paths = sorted(root.glob(pattern))
        else: self.paths = [root]
        self.template, self.skip_reasons = template, Counter()
        self.manifest_metadata = {"template_name": "arxiv_summary_v1",
            "template_sha256": hashlib.sha256(template.encode()).hexdigest(), "split": split}

    def iter_samples(self, limit: int | None = None) -> Iterator[SemanticSample]:
        emitted = 0
        for path, row_no, row in iter_parquet(self.paths):
            article, abstract = row.get("article"), row.get("abstract")
            if not article or not abstract:
                self.skip_reasons["empty_article_or_abstract"] += 1; continue
            rid = str(row.get("id", row_no)); split = path.name.split("-", 1)[0]
            yield SemanticSample(f"arxiv:{split}:{rid}", "arxiv", "summarization", "single",
                prompt=self.template.format(article=article), reference_output=str(abstract),
                metadata={"source_record_id": rid, "source_row": row_no, "split": split})
            emitted += 1
            if limit is not None and emitted >= limit: return
