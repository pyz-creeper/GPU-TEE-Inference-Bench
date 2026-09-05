import json
import pyarrow.parquet as pq

from input_bench.recorder import write_results
from input_bench.sender import RequestEvent

def test_recorder_writes_all_three_artifacts(tmp_path):
    event = RequestEvent("r", 0, request_start_ns=0, headers_ns=1,
        first_content_ns=2, chunk_times_ns=[2], request_end_ns=10,
        http_status=200, response_text="ok", server_usage={"completion_tokens": 1},
        output_tokens=1, input_tokens=2, max_output_tokens=3)
    summary = write_results(tmp_path, [event],
        {"monotonic_start_ns": 0, "monotonic_end_ns": 100}, {"model": "m"})
    assert (tmp_path / "events.jsonl").exists() and (tmp_path / "summary.json").exists()
    assert pq.read_table(tmp_path / "results.parquet").num_rows == 1
    assert json.loads((tmp_path / "summary.json").read_text())["counts"]["completed"] == 1
    assert summary["experiment_config"]["model"] == "m"
