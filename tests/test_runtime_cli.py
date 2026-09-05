import json

from input_bench.cli import main
from input_bench.compiler import load_workload


def test_compile_uses_configured_data_tokenizer_and_workload_paths(tmp_path):
    data_root = tmp_path / "data"
    source = data_root / "chat/sharegpt-v3/ShareGPT_V3_unfiltered_cleaned_split.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps([{
        "id": "sample",
        "conversations": [
            {"from": "human", "value": "hello there"},
            {"from": "gpt", "value": "general kenobi"},
        ],
    }]))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "data": {"root": "data"},
        "tokenizer": {"id": "whitespace"},
        "workload": {"path": "out/workload.jsonl"},
        "endpoint": {"backend": "vllm", "model": "test"},
    }))
    status = main([
        "compile", "--config", str(config), "--adapter", "sharegpt",
        "--arrival", "fixed-concurrency",
    ])
    assert status == 0
    workload = tmp_path / "out/workload.jsonl"
    assert len(load_workload(workload)) == 1
    manifest = json.loads((workload.parent / "manifest.json").read_text())
    assert manifest["tokenizer"]["id"] == "whitespace"


def test_configured_dataset_listing(tmp_path, capsys):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"data": {"root": "datasets"}}))
    assert main(["datasets", "--config", str(config)]) == 0
    assert str(tmp_path / "datasets/chat/sharegpt-v3") in capsys.readouterr().out
