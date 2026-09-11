# GLM-5.3 旗舰版双节点可行性（仅讨论，2026-09-08）

后续用户已授权实施 PP2、不加 MTP 的路线，现已完成376次固定轨迹请求回放，
总耗时121.24分钟、零请求错误，见 [部署与回放说明](GLM53_PP2_TRAJECTORY_RUNBOOK.md)。
以下保留当时只读可行性核查的结论；TP16+MTP仍未实施。

这里的“全量模型”指非 Flash 的 GLM-5.3 旗舰版。两机现有 `/data/model/GLM-5.3`
是原生 FP8 权重，不是 GLM-5.3-BF16。本轮没有启动旗舰版、安装其运行环境或执行通信压测。

## 已核实的条件

- 主机 `192.168.0.63`、`192.168.0.65`，每台 8 张 H20；单卡 NVML 报告 97871 MiB，
  两台合计约 1529.23 GiB 显存。
- 两机均有 141 个 GLM-5.3 safetensors 分片，与索引清单一致，文件大小合计
  755632050320 字节，即 703.74 GiB。config、tokenizer、template、generation config、
  权重索引 SHA-256 在两机相同；本轮未重新计算全部 704 GiB 权重的逐文件哈希。
- 模型为 `GlmMoeDsaForCausalLM`，78 层、64 个 attention heads、256 个 routed experts、
  每 token 激活8个专家、1个 NextN/MTP 层；权重索引确实含 `model.layers.78.*` 的 MTP 参数。
- 两机 `/data/src/sglang-glm53` 均为 `c767511ea832829129ba5255ced67b9ce1a9bc2b`，
  均保留原有 `load_model_utils.py` 改动，文件 SHA-256 为
  `d3555e6bd535ceb4b3eb92cb2fa7d29f5bc36bc35227541a4b0d090d2d715b74`。
  源码相同不等于 Conda/CUDA 依赖相同；`.65` 的 `glm53-sglang` 已实际部署 Flash，
  `.63` 仍需在今后部署前对齐 Python、Torch、FlashInfer、sgl-kernel、NCCL 等依赖。
- 两机 eth0 报告 200000 Mb/s；各有 `mlx5_0`，端口 ACTIVE、200 Gb/s、链路类型 Ethernet，
  可见 `/dev/infiniband/uverbs0`。这表明存在 RDMA/RoCE 设备，不证明端到端 RoCE、GPUDirect
  和 NCCL 实际带宽已经可用。本轮未执行带宽/collective 测试，也未更改网络配置。
- 此前同两台机器的 GLM-5.2 FP8 已跑通过 TP=8、PP=2、DP=1，使用 TCP/eth0、
  `NCCL_IB_DISABLE=1`，MTP 关闭。见 [既有报告](../campaigns/glm52-pp2/REPORT.md)。
  它是双机链路与框架基线，不是 GLM-5.3 的成功证据。

## 内存判断

FP8 旗舰版在16卡上具有合理容量余量。仅按权重均分估算，16卡约 43.98 GiB/卡，
8卡约 87.97 GiB/卡；实际还要考虑层/专家分配、复制参数、KV cache、MTP、CUDA graphs、
NCCL buffer 与内核工作区。所以优先考虑双节点，不把单机8卡当作稳妥的完整服务方案。

“总权重能放下”不等于可以直接开1M上下文。初次验证适合64K、请求并发1、小规模 CUDA graph，
再按显存实测扩大。FP8是量化精度，依然包含旗舰模型的完整参数；不是只加载激活专家。
如果“全量”特指 BF16，则需另讨论约1.5TB权重与运行余量；当前两台H20不宜以BF16作为起点。
官方明确区分 FP8 与 BF16，并提供 MTP 支持说明。[SGLang GLM-5.3 文档](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/GLM/GLM-5.3.mdx)

## 两条部署路线

| 路线 | 并行布局 | 能否带 MTP | 判断 |
|---|---|---|---|
| 先验证完整模型可用 | 每节点 TP=8，总 PP=2，DP=1 | 当前版本不能 | 与既有 GLM-5.2 双节点基线接近，跨机主要传流水线边界；C1 有流水线空泡 |
| 验证旗舰版 MTP | 跨两节点 TP=16，PP=1，DP=1 | 源码与模型层面可尝试，未实测 | 不触发 PP/MTP 禁止条件，但跨机 collective 更频繁，通信可能抵消投机收益 |

第一条路线需要 `disable_overlap_schedule`，且不设置任何 speculative 参数。
第二条路线使用 EAGLE/NextN，不使用 DeepSeek 0731 的 DSPARK。当前源码将旗舰模型的 draft
映射为 `GlmMoeDsaForCausalLMNextN`；官方给出的 MTP 形状包括 5/1/6 和较保守的1/1/2。
实际未来验证应从短草稿和 C1 开始，再增加步数，测量接受长度与端到端耗时；高接受率本身不是加速证据。

**当前 PP=2 + MTP 不可用是代码明确限制，不只是经验猜测。**
部署源码 `arg_groups/validation_hook.py` 在 `pp_size > 1` 时要求
`disable_overlap_schedule and speculative_algorithm is None`，否则拒绝启动。
本机现用0.5.18的 `server_args.py` 也有相同断言。
[固定版本源码](https://github.com/sgl-project/sglang/blob/c767511ea832829129ba5255ced67b9ce1a9bc2b/python/sglang/srt/arg_groups/validation_hook.py#L51)

TP16+MTP仍需验证：模型/量化内核在TP16下的分片尺寸、NextN权重加载、跨节点NCCL、
上下文与草稿工作区，以及实际接受率。不能把 Flash 单机 MTP 成功外推为旗舰版双机成功。

## 将来获准实施时的顺序

1. 对齐两机的已验证 SGLang 源码及依赖，保留本地补丁并记录哈希。
2. 确认空闲GPU；仅在实施阶段运行NCCL通信检查。优先验证RDMA/RoCE，不直接假设200Gb/s标称带宽可达。
3. 使用相同模型路径、两节点rank0/1、同一`dist-init-addr`建立PP2/TP8无MTP基线。
4. 另开独立实验尝试PP1/TP16，先无MTP，再启用EAGLE；不在已有测量中途切换并行或草稿配置。
5. 复用本轮20条/376次冻结轨迹，保持high、4096输出上限及C1，分别保存结果。

以上仅是待实施路线。本轮仅部署和测试 `.65` 的 GLM-5.3-Flash。
