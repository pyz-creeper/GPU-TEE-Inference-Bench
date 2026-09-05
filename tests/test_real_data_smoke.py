"""Bounded smoke checks against mounted source data; skipped on developer machines."""
import os
from pathlib import Path

import pytest

from input_bench.adapters import (ArxivSummarizationAdapter, LongBenchAdapter,
    MooncakeTraceAdapter, ShareGPTAdapter, SWEAgentTrajectoriesAdapter,
    SWEBenchVerifiedAdapter, ThoughtworksAgenticAdapter)
from input_bench.tokenizer import WhitespaceTokenizer

ROOT = Path(os.getenv("INPUT_BENCH_DATA_ROOT", "/data/benchmarks"))
pytestmark = pytest.mark.skipif(not ROOT.exists(), reason="benchmark mount unavailable")


@pytest.mark.parametrize("adapter", [
    lambda: ShareGPTAdapter(ROOT / "chat/sharegpt-v3/ShareGPT_V3_unfiltered_cleaned_split.json"),
    lambda: SWEAgentTrajectoriesAdapter(ROOT / "coding/swe-agent-trajectories/data/train-00000-of-00012.parquet"),
    lambda: SWEBenchVerifiedAdapter(ROOT / "coding/swe-bench-verified/data/test-00000-of-00001.parquet"),
    lambda: ThoughtworksAgenticAdapter(ROOT / "coding/thoughtworks-agentic-trajectories/sessions.parquet"),
    lambda: ArxivSummarizationAdapter(ROOT / "summarization/arxiv/document", split="validation"),
    lambda: LongBenchAdapter(ROOT / "summarization/longbench/data.zip", subset="gov_report"),
])
def test_real_semantic_source_reads_one_record(adapter):
    sample = next(adapter().iter_samples(limit=1))
    assert sample.sample_id and (sample.prompt is not None) != (sample.messages is not None)


@pytest.mark.parametrize("name", ["conversation_trace", "toolagent_trace", "synthetic_trace"])
def test_real_mooncake_trace_reads_one_record(name):
    adapter = MooncakeTraceAdapter(ROOT / f"traces/mooncake/{name}.jsonl", WhitespaceTokenizer())
    sample = next(adapter.iter_samples(limit=1))
    assert len(WhitespaceTokenizer().encode(sample.prompt)) == sample.metadata["input_length"]
