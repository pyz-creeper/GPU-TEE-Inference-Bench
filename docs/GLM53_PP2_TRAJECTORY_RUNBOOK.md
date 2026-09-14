# GLM-5.3 FP8：双节点 PP=2、MTP 关闭的 SWE 轨迹回放

## 2026-09-09 后端恢复

按用户要求结束暂停的 vLLM 回放并关闭两节点 vLLM，恢复本 SGLang 环境。
当前启动脚本及配置模板使用 `--mem-fraction-static 0.90`，TP8/PP2、MTP 关闭，
其余参数沿用之前成功部署的配置。此次仅部署与接口检查，不启动轨迹回放。
下方 2026-09-08 的 121.24 分钟结果来自 **0.80** 配置，历史结果不变。

恢复已于北京时间 15:01:43 完成。`/get_server_info` 确认 TP8、PP2、nnodes=2、
mem_fraction_static=0.9、BF16 KV、64K 上下文、speculative_algorithm=null。
实际 high 思考请求 `17×19` 返回 HTTP 200、答案 `323`，耗时 1.266 秒。
验证记录：`runs/agentic-replay/sglang-glm53-pp2-high-20/deployment/restore-20260909/`。

## 本轮实测结果（2026-09-08）

正式运行 `sglang-glm53-pp2-high-20-r1` 于北京时间12:09:11至14:10:25完成。
20条会话、376/376请求完整回放，HTTP错误/超时/取消/漏发均为0。
总耗时 **7274.665秒（121.24分钟，约2小时1分15秒）**；输出195675 tokens，
服务端输入与参考输入均为3992020 tokens。首token延迟p50为781ms，单请求耗时p50为6.538秒。
11次达到输出上限，4次没有最终答案；两者不是请求传输失败，也不代表解题成功。

此前百炼GLM-5.3为3039.519秒（50.66分钟），本次耗时为其2.3934倍。
六端点结果与校验见[完整报告](../runs/agentic-replay/sglang-glm53-pp2-high-20/REPORT.md)。
不同输出长度、模型修订、思考、缓存、启动探针预热状态和网络条件限制因果解释。
本轮没有新增百炼API请求。61项回归测试通过；真实冻结bundle另行校验。

## 部署选择

- 节点：192.168.0.63（PP0/API/客户端）、192.168.0.65（PP1），各8张H20 96GB。
- 两端模型：`/data/model/GLM-5.3`，141个原生FP8分片。
- 两端复用 `/opt/miniforge3/envs/glm52-sglang`。环境名称保留，实际包为
  SGLang `0.0.0.dev1+gc767511ea`、Torch 2.13.0、sglang-kernel 0.4.6.post1、
  FlashInfer 0.6.18、Transformers 5.12.1；源码和原有加载超时补丁一致。
- TP8、PP2、DP1；不传 speculative 参数，并显式关闭 overlap schedule。
- 参考[固定版本官方 recipe](https://github.com/sgl-project/sglang/blob/c767511ea832829129ba5255ced67b9ce1a9bc2b/docs/cookbook/autoregressive/GLM/GLM-5.3.mdx)：
  FP8权重、BF16 KV、DSA prefill `flashmla_sparse`、decode `fa3`、top-k `sgl-kernel`，
  memory fraction 0.80。MoE runner保持auto，由已安装实现选择适用内核。
- 64K上下文用于容纳最长参考输入31695 tokens及4096输出上限；不尝试1M上下文。
- 沿用此前GLM5.2 PP2验证过的prefill chunk8192、max prefill16384、max running16，
  decode CUDA graphs为1/2/4/8/16，关闭prefill CUDA graphs以减少启动捕获。
- 沿用TCP/eth0，`NCCL_IB_DISABLE=1`；本轮不切换或测试RDMA。
- `.65`原有Flash服务已停止以释放GPU；没有修改或删除原Flash启动脚本与结果。

## 启动与查看

确认两机GPU空闲后，在项目根目录执行：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD scp scripts/run_glm53_pp2_node.sh root@192.168.0.65:/root/glm53_pp2_node.sh
tmux new-session -d -s glm53-pp2-node0 'bash /root/GPU-TEE-Inference-Bench/scripts/run_glm53_pp2_node.sh 0'
env -u LD_LIBRARY_PATH -u LD_PRELOAD ssh root@192.168.0.65 "tmux new-session -d -s glm53-pp2-node1 'bash /root/glm53_pp2_node.sh 1'"
```

已有会话时不要重复启动。查看日志：

```bash
tmux attach -t glm53-pp2-node0
ssh -t root@192.168.0.65 'tmux attach -t glm53-pp2-node1'
```

API只绑定PP0的`127.0.0.1:30000`；分布式初始化使用`192.168.0.63:29500`。
两机日志均写入`/data/benchmarks/glm53-pp2-swe/logs/`，按启动时间分文件保存。

首次部署中，PP1冷文件缓存使并发权重加载变慢。停止尚未就绪的两个服务后，使用4个顺序
读取线程预读PP1所需的79个分片（403.6秒），再以原参数重启。
此过程只读取文件，不修改权重；启动与预读耗时不属于轨迹计时。
日志和节点依赖/模型元数据核对结果保存在结果根目录的`deployment/`下。

## 轨迹回放

复用百炼及两组本地Flash实验的冻结bundle，SHA256：
`8a8b4d97202f388b7719c232d69ec5181091d1b8bc61a8f89c4df36a7a3e60b1`。
20条完整会话、376次请求，high思考、流式响应、会话/HTTP并发均1、每次输出上限4096，
不重试、不做轨迹warmup、不主动清缓存。`reasoning_effort=high`经已安装SGLang映射到
模型模板；模板始终开启思考。客户端不覆盖temperature/top_p，与前几组实验保持一致。

离线检查与正式回放（每次使用新的run ID）：

```bash
/opt/miniforge3/envs/test-api/bin/python scripts/run_agentic_replay.py dry-run \
  --config scenarios/sglang_glm53_pp2_swe_trajectory_high.json --model all
tmux new-session -d -s test-run-glm53-pp2 \
  'bash /root/GPU-TEE-Inference-Bench/scripts/run_glm53_pp2_trajectory_tmux.sh sglang-glm53-pp2-high-20-r1'
```

上述r1已完成。保留当前服务、再运行一次时，使用新编号和会话名：

```bash
tmux new-session -d -s test-run-glm53-pp2-r2 \
  'bash /root/GPU-TEE-Inference-Bench/scripts/run_glm53_pp2_trajectory_tmux.sh sglang-glm53-pp2-high-20-r2'
```

不需要API Key。回放脚本检查实际模型名称、TP/PP、上下文及MTP关闭状态后才发送请求。
部署探针与服务内部启动warmup不计入正式计时；正式首请求开销保留。
这是固定请求性能回放，不运行真实agent工具，不计算SWE-bench解题率。

结果根目录：`runs/agentic-replay/sglang-glm53-pp2-high-20/`。
与云端比较时，输出长度、实际思考、缓存和硬件差异仍需单独考虑；
不能把旗舰版PP2无MTP与Flash单机MTP的差异归因于MTP本身。
