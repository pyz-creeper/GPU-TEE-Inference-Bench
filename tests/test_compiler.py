import json

from input_bench.adapters.sharegpt import ShareGPTAdapter
from input_bench.arrivals import PoissonPolicy
from input_bench.compiler import CompileOptions, compile_workload, load_workload
from input_bench.tokenizer import WhitespaceTokenizer

def test_compile_is_byte_reproducible_and_filters(tmp_path):
    source = tmp_path / "data.json"
    source.write_text(json.dumps([{"id": str(i), "conversations": [
        {"from": "human", "value": "one two"}, {"from": "gpt", "value": "answer words"}]}
        for i in range(8)]))
    tok = WhitespaceTokenizer(); opts = CompileOptions(seed=9, max_samples=4, min_input_tokens=1,
                                                       max_output_tokens=1, bucket_boundaries=(4, 8))
    outputs = [tmp_path / "a" / "workload.jsonl", tmp_path / "b" / "workload.jsonl"]
    for output in outputs:
        compile_workload(ShareGPTAdapter(source), tok, "whitespace", PoissonPolicy(2, 9), output,
                         opts, source_paths=[source])
    assert outputs[0].read_bytes() == outputs[1].read_bytes()
    rows = load_workload(outputs[0]); assert len(rows) == 4
    assert all(x.max_output_tokens == 1 and x.metadata["input_token_bucket_upper"] == 4 for x in rows)
    manifest = json.loads((outputs[0].parent / "manifest.json").read_text())
    assert manifest["dataset_files"][0]["sha256"] and manifest["counts"]["after_filter"] == 4
