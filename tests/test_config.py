import json
from pathlib import Path

import pytest

from input_bench.config import (load_runtime_config,
                                portable_tokenizer_reference,
                                resolve_tokenizer_reference)


def test_config_resolves_paths_relative_to_file_and_environment_wins(tmp_path, monkeypatch):
    config = tmp_path / "conf" / "bench.json"
    config.parent.mkdir()
    config.write_text(json.dumps({
        "data": {"root": "../datasets"},
        "tokenizer": {"path": "../tokenizer", "revision": "rev-1"},
        "workload": {"path": "../workloads/test.jsonl"},
        "results": {"root": "../results"},
        "endpoint": {
            "backend": "sglang",
            "base_url": "http://guest:30000",
            "model": "model-a",
            "headers": {"X-Test": "yes"},
            "tls_verify": False,
        },
    }))
    monkeypatch.setenv("INPUT_BENCH_MODEL", "model-from-env")
    loaded = load_runtime_config(config)
    assert loaded.data_root == (tmp_path / "datasets").resolve()
    assert loaded.tokenizer == str((tmp_path / "tokenizer").resolve())
    assert loaded.workload == (tmp_path / "workloads/test.jsonl").resolve()
    assert loaded.results_root == (tmp_path / "results").resolve()
    assert loaded.backend == "sglang" and loaded.base_url == "http://guest:30000"
    assert loaded.model == "model-from-env" and loaded.headers == {"X-Test": "yes"}
    assert loaded.tls_verify is False and loaded.tokenizer_revision == "rev-1"


def test_backend_defaults_and_validation(tmp_path):
    path = tmp_path / "sglang.json"
    path.write_text('{"endpoint":{"backend":"sglang"}}')
    assert load_runtime_config(path).base_url == "http://127.0.0.1:30000"
    path.write_text('{"endpoint":{"backend":"unknown"}}')
    with pytest.raises(ValueError, match="endpoint.backend"):
        load_runtime_config(path)


def test_local_tokenizer_reference_is_relocatable(tmp_path):
    tokenizer = tmp_path / "bundle" / "tokenizer"
    output = tmp_path / "bundle" / "workloads" / "workload.jsonl"
    tokenizer.mkdir(parents=True)
    output.parent.mkdir()
    reference = portable_tokenizer_reference(str(tokenizer), output)
    assert reference == "../tokenizer"
    assert resolve_tokenizer_reference(reference, output.parent) == str(tokenizer.resolve())


def test_huggingface_tokenizer_id_is_not_treated_as_path(tmp_path):
    identifier = "org/model-name"
    assert portable_tokenizer_reference(identifier, tmp_path / "workload.jsonl") == identifier
    assert resolve_tokenizer_reference(identifier, tmp_path) == identifier
