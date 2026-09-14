# GLM-5.2 FP8 two-node PP=2 workload report

## Deployment under test

- Hosts: `192.168.0.63` and `192.168.0.65`
- GPUs: 8 x NVIDIA H20 96 GB per host
- Model: `/data/model/GLM-5.2-FP8`
- Parallelism: PP=2, TP=8, DP=1
- SGLang endpoint: `http://192.168.0.63:30000`
- Context length: 32768
- KV cache: BF16
- Networking: TCP sockets on `eth0`; `NCCL_IB_DISABLE=1`
- MTP/speculative decoding: disabled

The server reported `chunked_prefill_size=8192`, `max_prefill_tokens=16384`,
`max_running_requests=16`, DSA prefill `flashmla_sparse`, DSA decode `fa3`, and
DSA top-k `sgl-kernel`.

## Workload

The source is the Mooncake FAST'25 conversation/tool-agent trace at upstream
commit `2b3cefbac4d571bcb001db3aa702772b41b8c647`. It is a shape-and-arrival
trace, not recoverable semantic text. Input Bench materializes deterministic
text from the GLM-5.2 tokenizer vocabulary and verifies exact round-trip token
counts. Records outside the server's 32K context are filtered. Every request
uses `ignore_eos=true`, and every point starts after `/flush_cache`; prefix
reuse within a point is retained through the trace `hash_ids`.

| Workload | Requests | Input tokens | Output tokens |
|---|---:|---:|---:|
| conversation | 12 | mean 8,926; p50 6,946; max 22,280 | exactly 128 each |
| tool-agent fixed | 24 | mean 5,389; p50 4,927; max 14,463 | mean 70; range 13-128 |
| tool-agent trace | 48 | mean 7,703; p50 6,171; max 24,240 | mean 73.6; range 9-128 |

The C1/C4 conversation points replay the same immutable workload. The C4/C8
tool-agent points likewise replay the same immutable workload.

## Results

Run directory: `/data/benchmarks/glm52-pp2-campaign/results/20260905T100100Z`

| Point | Duration | Success | Input tok/s | Output tok/s | TTFT p50 / p95 | TPOT p50 / p95 |
|---|---:|---:|---:|---:|---:|---:|
| conversation C1 | 96.21 s | 12/12 | 1,113.30 | 15.97 | 377 ms / 18.01 s | 34.15 / 37.26 ms |
| conversation C4 | 48.63 s | 12/12 | 2,202.31 | 31.58 | 1.40 s / 17.02 s | 60.65 / 185.09 ms |
| tool-agent C4 | 48.95 s | 24/24 | 2,642.27 | 34.34 | 379 ms / 11.83 s | 60.67 / 180.42 ms |
| tool-agent C8 | 39.01 s | 24/24 | 3,315.90 | 43.09 | 406 ms / 11.43 s | 90.29 / 217.43 ms |
| tool-agent timestamp trace | 66.49 s | 48/48 | 5,560.73 | **53.12** | 1.12 s / 16.69 s | 73.67 / 399.57 ms |

All expected output tokens were returned and there were no HTTP, timeout, SSE,
or token-count errors. Timestamp replay scheduler lag was 3.03 ms mean and
8.78 ms p99, so the client was not the throughput bottleneck. Fixed-concurrency
`client_queue_delay` includes the deliberate closed-loop semaphore backlog and
must not be interpreted as scheduler lag.

## Interpretation

The PP=2 deployment reaches the 50 output-token/s target only as aggregate
throughput under the bursty timestamp trace (53.12 output tok/s). It does not
meet a 50 tok/s single-stream target: conversation C1 TPOT p50 34.15 ms implies
about 29.3 decode tok/s while actively generating, and full-run output
throughput is 15.97 tok/s after real long-prefill time is included.

The earlier random 512-input/512-output C4 result of 64.25 output tok/s therefore
overstates the serving surface for these long-context traces. Increasing fixed
concurrency from 4 to 8 improves tool-agent aggregate throughput from 34.34 to
43.09 tok/s, but worsens TPOT p50 from 60.67 to 90.29 ms. This configuration is
throughput-usable around the target only when latency requirements are loose.

This is one deterministic run per point, not a confidence interval. Production
acceptance should repeat each point at least three times.

## Reproduction

Attach to the completed interactive tmux pane:

```bash
tmux attach -t glm52-realbench
```

Prepare and run the full campaign again:

```bash
cd /root/GPU-TEE-Inference-Bench
scripts/run_glm52_pp2_campaign_tmux.sh
```

To create a new tmux session after the current one is removed or renamed:

```bash
tmux new-session -d -s glm52-realbench -n campaign \
  /root/GPU-TEE-Inference-Bench/scripts/run_glm52_pp2_campaign_tmux.sh
```

Run one point only:

```bash
scripts/run_glm52_pp2_campaign_tmux.sh --only fixed-conversation-c1
```

The wrapper explicitly unsets `http_proxy`, `https_proxy`, `HTTP_PROXY`,
`HTTPS_PROXY`, `ALL_PROXY`, and `all_proxy`, and sets `NO_PROXY=*`.
