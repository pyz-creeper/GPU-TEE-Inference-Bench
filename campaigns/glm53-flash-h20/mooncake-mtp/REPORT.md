# GLM-5.3-Flash / SGLang / H20 Mooncake 同口径报告

## 结论

GLM-5.3-Flash 已在 node1 当前 SGLang 部署上完成与 GLM5.2、DeepSeek-V4-Flash-0731 相同的五点 Mooncake campaign。正式 run 为 `results/20260906T063000Z`，120/120 请求成功、0 失败。

当前部署开启 Adaptive EAGLE MTP，因此这是 **SGLang + GLM5.3-Flash 的当前最佳配置结果**，不是与另外两个模型相同的 target-only/MTP-off 消融结果。

| 测试点 | 时长 | 成功 | Input tok/s | Output tok/s | TTFT p50/p95 | TPOT p50/p95 | E2E p50/p95 | Goodput |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| conversation C1 | 12.08 s | 12/12 | 8,864.17 | **127.12** | 423/1,682 ms | 2.87/3.37 ms | 786/2,101 ms | 100.0% |
| conversation C4 | 7.96 s | 12/12 | 13,449.30 | **192.87** | 1,189/4,019 ms | 3.58/16.87 ms | 1,635/5,150 ms | 91.7% |
| tool-agent C4 | 9.67 s | 24/24 | 13,369.90 | **173.76** | 367/2,942 ms | 9.46/18.87 ms | 1,369/4,136 ms | 95.8% |
| tool-agent C8 | 9.94 s | 24/24 | 13,019.54 | **169.20** | 776/4,166 ms | 21.39/35.32 ms | 2,966/7,715 ms | 41.7% |
| timestamp trace C16 | 61.09 s | 48/48 | 6,051.86 | **57.81** | 1,778/4,961 ms | 6.26/40.12 ms | 2,361/5,446 ms | 85.4% |

Goodput SLO 与原 campaign 相同：TTFT ≤5,000 ms、TPOT ≤20 ms、E2E ≤120,000 ms，单请求必须同时满足三项。

## 测试口径

- 硬件：node1 单机 8×NVIDIA H20 96 GB
- 框架：SGLang，当前已部署服务
- 模型：`/data/model/GLM-5.3-Flash`
- 并行：TP=8、EP=8、PP=1
- 权重 FP8、KV BF16、32K context
- DSA TileLang prefill/decode、Triton linear attention、DeepGEMM MoE
- Adaptive EAGLE MTP：初始 5 steps、top-k 1、6 draft tokens
- `chunked_prefill_size=8192`、`max_prefill_tokens=8192`
- stream、`temperature=0`、`top_p=1`、`ignore_eos=true`
- 每个测试点开始前调用 `/flush_cache`；测试点内部保留 trace 自带的 prefix reuse
- 不计模型启动时间，不做请求级 warmup，与此前五点 campaign 相同
- 未使用代理下载；Mooncake trace 从本机已有校验副本直接 rsync 到 node1
- 单节点测试，不涉及 RDMA

## Workload 一致性

原始 Mooncake trace 与前两组相同：

- conversation trace：`b8cbb061a85206d729d91cdc2981f43c9e0d99209dce588d3af5f7934408b9df`
- tool-agent trace：`48a2db1a13d3bc05e6330140c64f604ba366df20d3c9e128b5c35a01c1fa5f71`

GLM5.3 与 GLM5.2 tokenizer 生成出的最终三个 workload 也逐字节一致：

- conversation：`a0ecf02064b13fbd5a745519b381d578722fe08b605b38fdc0140a6b18793112`
- tool-agent fixed：`c800a70d403c148bb57f732fd8b6e96098145a7719c3682757eb6685233bac85`
- tool-agent trace：`5cdd4665f51a71ba1eced9a7d8fb514c08c6c071e63e93e6e4f92f7fe6c901db`

DSV4 使用相同源记录、token 形状、输出上限和到达模式，但因 tokenizer 不同，最终 prompt 文本和 workload SHA 不同。

## 三模型与框架横向表

下表均为相同五点口径；H800 数据未纳入。

| 测试点 | GLM5.2 SGLang PP2/16×H20 | DSV4 vLLM/8×H20 | DSV4 SGLang/8×H20 | GLM5.3 SGLang MTP/8×H20 |
|---|---:|---:|---:|---:|
| conversation C1 | 15.97 | 54.37 | 74.03 | **127.12** |
| conversation C4 | 31.58 | **236.82** | 192.92 | 192.87 |
| tool-agent C4 | 34.34 | 118.38 | 148.14 | **173.76** |
| tool-agent C8 | 43.09 | 126.60 | **183.83** | 169.20 |
| timestamp trace C16 | 53.12 | 46.60 | 57.28 | **57.81** |

只看同为单机 SGLang 的 DSV4 与 GLM5.3：

- Conversation C1：GLM5.3 高 71.7%，MTP 对单流 decode 帮助非常明显。
- Conversation C4：两者几乎完全相同，192.87 对 192.92 tok/s。GLM5.3 的 TPOT 更好，但 TTFT p95 更差，两个阶段抵消。
- Tool-agent C4：GLM5.3 高 17.3%，goodput 95.8% 对 66.7%。
- Tool-agent C8：GLM5.3 反而低 8.0%，169.20 对 183.83 tok/s；说明该 GLM5.3 配置从 C4 到 C8 已进入吞吐平台区。
- Timestamp trace：57.81 对 57.28 tok/s，差异不到 1%。两者都接近 60 秒到达窗口决定的理论上限 58.87 tok/s，不能用这个点判断纯计算速度。

## 性能解释

GLM5.3 的强项是生成阶段。Conversation C1 的 TPOT p50=2.87 ms，对应生成阶段约 348 tok/s；DSV4 SGLang 为 8.72 ms，约 115 tok/s。但这是 MTP-on 对 target-only，不能把 3.0× 全部解释成模型本体差距。

长上下文并发仍受 prefill 约束。Conversation C4 中 GLM5.3 的 TTFT p95=4.02 s，明显高于 DSV4 SGLang 的 2.28 s；尽管 GLM5.3 TPOT 更低，最终聚合吞吐只是持平。

Tool-agent C8 比 C4 多一倍并发，输出吞吐却从 173.76 降到 169.20 tok/s，同时 TPOT p50 从 9.46 恶化到 21.39 ms。当前 TP=8/EP=8 部署在 C4 左右已接近较优吞吐—延迟工作点，继续增加并发主要是在牺牲单请求速度。

Trace 点 61.09 秒完成，尾部只有约 1.09 秒；输出吞吐 57.81 tok/s 已非常接近 trace 的 58.87 tok/s 到达上限。它说明服务能跟上这组线上到达节奏，但不能证明上限只有 57.81 tok/s。

## 公平性限制

- GLM5.3 开启 MTP；GLM5.2 和 DSV4 对照关闭 speculative decoding。
- GLM5.2 是两节点 16×H20、PP=2 over TCP；另外两个是单节点 8×H20。
- 不同模型的计算量和架构不同，不能只按 GPU 数做线性归一化。
- 当前数据没有 MTP-off 的 GLM5.3 消融，因此尚不能拆出 MTP 增益与模型本体速度。
- 没有 Nsight/NCCL/kernel profile，本报告只能从端到端 TTFT/TPOT/吞吐推断瓶颈。

## 文件和运行入口

- 正式结果：`results/20260906T063000Z/`
- 完整成功日志：`logs/campaign-20260906-142922.log`
- `logs/campaign-20260906-142830.log` 是驱动兼容性检查失败，发生在 workload 编译阶段，没有向模型发出正式请求。
- 配置快照：`config.snapshot.json`
- 冻结计划：`plan.json`
- 原始 events、Parquet 和每点 summary 位于正式结果目录。
- node1 tmux：`tmux attach -t glm53-mooncake-bench`

tmux pane 已在 campaign 完成后回到 `/root/GPU-TEE-Inference-Bench` 的交互式 shell。
