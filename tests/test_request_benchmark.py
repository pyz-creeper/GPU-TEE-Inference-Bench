import pytest

from input_bench.benchmark import aggregate_request_benchmark


def summary(duration, offered=3, completed=3, failed=0):
    return {
        "measurement": {"duration_s": duration},
        "counts": {"offered": offered, "completed": completed, "failed": failed},
    }


def test_request_benchmark_aggregates_only_whole_run_duration():
    report = aggregate_request_benchmark([summary(3.0), summary(1.0), summary(2.0)])

    assert report["repeat_count"] == 3
    assert report["all_requests_completed"] is True
    assert report["total_duration_s"]["values"] == [3.0, 1.0, 2.0]
    assert report["total_duration_s"]["mean"] == 2.0
    assert report["total_duration_s"]["p50"] == 2.0
    assert "latency_ms" not in report


def test_request_benchmark_marks_incomplete_runs():
    report = aggregate_request_benchmark([summary(1.0, completed=2, failed=1)])
    assert report["all_requests_completed"] is False
    assert report["runs"][0]["failed"] == 1


def test_request_benchmark_requires_valid_measurements():
    with pytest.raises(ValueError, match="at least one"):
        aggregate_request_benchmark([])
    with pytest.raises(ValueError, match="measurement.duration_s"):
        aggregate_request_benchmark([{"measurement": {}, "counts": {}}])
