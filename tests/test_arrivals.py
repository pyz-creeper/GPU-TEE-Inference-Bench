from input_bench.arrivals import FixedConcurrencyPolicy, PoissonPolicy, TimestampTracePolicy
from input_bench.schema import SemanticSample

def samples(n=10):
    return [SemanticSample(str(i), "x", "chat", "single", prompt="x",
                           metadata={"timestamp_s": [2, 2, 4][i] if n == 3 else i}) for i in range(n)]

def test_fixed_and_poisson_are_deterministic():
    rows = samples()
    assert FixedConcurrencyPolicy(2).offsets(rows) == [None] * 10
    a = PoissonPolicy(5, seed=7).offsets(rows)
    assert a == PoissonPolicy(5, seed=7).offsets(rows)
    assert a[0] == 0 and a == sorted(a) and len(a) == 10
    assert PoissonPolicy(1, seed=1, duration=0).offsets(rows) == [0]

def test_trace_normalizes_scales_and_preserves_ties():
    rows = samples(3); policy = TimestampTracePolicy(time_scale=2, start_offset=.5)
    assert policy.select(rows) == rows
    assert policy.offsets(rows) == [.5, .5, 1.5]
