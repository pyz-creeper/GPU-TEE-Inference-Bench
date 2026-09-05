from input_bench.backends import parse_server_metrics


def test_vllm_metrics_are_normalized():
    metrics = parse_server_metrics(
        """
vllm:num_requests_running{model_name="m"} 3
vllm:num_requests_waiting{model_name="m"} 4
vllm:kv_cache_usage_perc{model_name="m"} 0.25
vllm:prompt_tokens_total{model_name="m"} 100
vllm:generation_tokens_total{model_name="m"} 20
""", "vllm")
    assert metrics == {
        "running": 3.0, "waiting": 4.0, "kv_cache_usage": 0.25,
        "prompt_tokens_total": 100.0, "generation_tokens_total": 20.0,
    }


def test_sglang_metrics_are_normalized():
    metrics = parse_server_metrics(
        """
sglang:num_running_reqs{model_name="m"} 3
sglang:num_queue_reqs{model_name="m"} 4
sglang:token_usage{model_name="m"} 0.25
sglang:prompt_tokens_total{model_name="m"} 100
sglang:generation_tokens_total{model_name="m"} 20
sglang:cache_hit_rate{model_name="m"} 0.75
""", "sglang")
    assert metrics == {
        "running": 3.0, "waiting": 4.0, "kv_cache_usage": 0.25,
        "prompt_tokens_total": 100.0, "generation_tokens_total": 20.0,
        "prefix_hit_fraction": 0.75,
    }
