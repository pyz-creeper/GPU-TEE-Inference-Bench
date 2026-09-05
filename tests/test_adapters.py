import json
import zipfile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from input_bench.adapters import (ArxivSummarizationAdapter, LongBenchAdapter,
    MooncakeTraceAdapter, ServeGenBridge, ShareGPTAdapter,
    SWEAgentTrajectoriesAdapter, SWEBenchVerifiedAdapter, ThoughtworksAgenticAdapter)
from input_bench.tokenizer import WhitespaceTokenizer
from input_bench.adapters.base import AdapterError, iter_json_array


def write_parquet(path, rows): pq.write_table(pa.Table.from_pylist(rows), path)


def test_sharegpt_turns_roles_and_stable_ids(tmp_path):
    path = tmp_path / "share.json"
    path.write_text(json.dumps([{"id": "a", "conversations": [
        {"from": "system", "value": "rules"}, {"from": "human", "value": "hi"},
        {"from": "gpt", "value": "hello"}, {"from": "human", "value": "next"},
        {"from": "gpt", "value": "done"}]},
        {"id": "bad", "conversations": [{"from": "gpt", "value": "oops"}]}]))
    adapter = ShareGPTAdapter(path, mode="session_replay")
    rows = list(adapter.iter_samples())
    assert [x.sample_id for x in rows] == ["sharegpt:a:turn:0", "sharegpt:a:turn:1"]
    assert [m.role for m in rows[0].messages] == ["system", "user"]
    assert rows[1].parent_sample_id == rows[0].sample_id
    assert rows[0].reference_output == "hello"
    assert adapter.skip_reasons["invalid_role_order"] == 1


def test_json_array_partial_record_at_eof_raises(tmp_path):
    source = tmp_path / "truncated.json"
    source.write_text('[{"id": 1}')
    with pytest.raises(AdapterError, match="malformed JSON"):
        list(iter_json_array(source))


def test_json_array_chunk_starts_with_whitespace(tmp_path, monkeypatch):
    source = tmp_path / "chunked.json"
    source.write_text('[{"id": 1},\n {"id": 2}]')
    # Exercise the same parser state using a record sequence whose separator
    # contains leading whitespace before the next value.
    assert [row["id"] for _, row in iter_json_array(source)] == [1, 2]


def test_arxiv_and_swebench_reference_isolation(tmp_path):
    arxiv = tmp_path / "validation-00000.parquet"
    write_parquet(arxiv, [{"id": "p", "article": "long article", "abstract": "gold"}])
    sample = next(ArxivSummarizationAdapter(arxiv).iter_samples())
    assert sample.reference_output == "gold" and "gold" not in sample.prompt
    swe = tmp_path / "swe.parquet"
    write_parquet(swe, [{"instance_id": "i", "problem_statement": "fix bug", "repo": "r",
                         "patch": "SECRET", "test_patch": "TEST", "FAIL_TO_PASS": "[]", "PASS_TO_PASS": "[]"}])
    sample = next(SWEBenchVerifiedAdapter(swe).iter_samples())
    assert "SECRET" not in sample.prompt and sample.metadata["reference"]["patch"] == "SECRET"


def test_coding_trajectory_adapters(tmp_path):
    swe = tmp_path / "swe.parquet"
    write_parquet(swe, [{"instance_id": "i", "model_name": "m", "target": "t",
        "trajectory": json.dumps([{"role": "user", "content": "issue"},
          {"role": "ai", "content": "cmd"}, {"role": "observation", "content": "result"},
          {"role": "ai", "content": "answer"}]), "exit_status": "ok"}])
    rows = list(SWEAgentTrajectoriesAdapter(swe).iter_samples())
    assert len(rows) == 2 and rows[1].messages[-1].role == "tool"
    tw = tmp_path / "tw.parquet"
    write_parquet(tw, [{"session_id": "s", "source_dataset": "x", "messages_json": json.dumps([
        {"role": "user", "content": "q", "custom": 1},
        {"role": "assistant", "content": "a", "tool_calls": [{"id": "1"}]}])}])
    row = next(ThoughtworksAgenticAdapter(tw).iter_samples())
    assert row.messages[0].extra["custom"] == 1 and row.metadata["source_dataset"] == "x"


def test_longbench_reads_zip_and_distinguishes_extended(tmp_path):
    path = tmp_path / "data.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("data/gov_report.jsonl", json.dumps({"context": "report", "input": "", "answers": ["sum"]}) + "\n")
        z.writestr("data/gov_report_e.jsonl", json.dumps({"context": "extended", "input": "", "answers": []}) + "\n")
    adapter = LongBenchAdapter(path)
    rows = list(adapter.iter_samples())
    assert adapter.list_subsets() == ["gov_report", "gov_report_e"]
    assert rows[0].sample_id != rows[1].sample_id and rows[0].workload_class == "summarization"


def test_shape_bridges_are_exact_and_preserve_trace(tmp_path):
    moon = tmp_path / "trace.jsonl"
    moon.write_text(json.dumps({"timestamp": 40, "input_length": 4, "output_length": 2, "hash_ids": [1]}) + "\n")
    tok = WhitespaceTokenizer(); row = next(MooncakeTraceAdapter(moon, tok).iter_samples())
    assert len(tok.encode(row.prompt)) == 4 and row.metadata["timestamp_s"] == .04
    serve = tmp_path / "serve.jsonl"
    serve.write_text(json.dumps({"request_id": "r", "timestamp": 1.5, "isl": 3, "osl": 7, "client_id": "c"}) + "\n")
    row = next(ServeGenBridge(serve, tok).iter_samples())
    assert len(tok.encode(row.prompt)) == 3 and row.requested_output_tokens == 7
    assert row.metadata["client_id"] == "c"


def test_mooncake_equal_hash_blocks_generate_equal_prefixes(tmp_path):
    moon = tmp_path / "trace.jsonl"
    moon.write_text("\n".join(json.dumps(x) for x in [
        {"timestamp": 0, "input_length": 600, "output_length": 1, "hash_ids": [7, 8]},
        {"timestamp": 1, "input_length": 600, "output_length": 1, "hash_ids": [7, 9]},
    ]) + "\n")
    rows = list(MooncakeTraceAdapter(moon, WhitespaceTokenizer()).iter_samples())
    assert rows[0].prompt.split()[:512] == rows[1].prompt.split()[:512]
