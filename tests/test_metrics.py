from input_bench.metrics import summarize_events

def test_metrics_counts_latency_throughput_and_goodput():
    event = {"request_start_ns": 0, "headers_ns": 10, "first_content_ns": 100_000_000,
             "chunk_times_ns": [100_000_000, 150_000_000], "request_end_ns": 200_000_000,
             "client_queue_delay_ns": 10_000_000, "scheduler_lag_ns": 5_000_000,
             "error_type": None, "input_tokens": 10, "output_tokens": 2}
    summary = summarize_events([event], {"monotonic_start_ns": 0, "monotonic_end_ns": 1_000_000_000},
                               {"ttft_ms": 110, "e2e_ms": 250, "tpot_ms": 60})
    assert summary["counts"]["completed"] == 1 and summary["rps"]["completed"] == 1
    assert summary["latency_ms"]["ttft"]["p50"] == 100
    assert summary["token_throughput_per_s"]["total"] == 12
    assert summary["goodput"]["count"] == 1
