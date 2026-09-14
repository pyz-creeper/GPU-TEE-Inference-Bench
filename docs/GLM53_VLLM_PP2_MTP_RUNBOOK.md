# GLM-5.3：vLLM 双节点 PP2 + MTP 实验

2026-09-09 更新：用户取消继续实验，已终止暂停的客户端并关闭两节点 vLLM。
部分结果保留；后端切回 [SGLang TP8/PP2、显存比例 0.90](GLM53_PP2_TRAJECTORY_RUNBOOK.md)。
下面的暂停状态是 2026-09-08 的历史记录，不代表进程仍在运行。

本轮比较同一份 SWE trajectory 在 vLLM PP2/MTP 下的耗时与此前 SGLang
PP2/无 MTP 的 **7274.665 秒（121.24 分钟）**。两节点已完成部署和基础校验。
2026-09-08 17:50（北京时间），按用户要求暂停第二轮：已保存 **342/376 请求**，
0 个请求错误、38 次输出达到上限，已完成请求段耗时 **104.55 分钟**。
尚无完整连续回放结果，不能判断整套是否低于 120 分钟。
状态与原始数据见 `runs/agentic-replay/vllm-glm53-pp2-mtp-high-20/`。

暂停采用 SIGSTOP，仅停止回放客户端 PID 3261897；模型服务仍驻留显存，
暂停前的在途请求可能在服务端完成。`launches/...-r2/pause.json` 保存暂停记录。
直接 SIGCONT 后的在途请求延迟和总墙钟时间不能用于连续实验比较；报告脚本
遇到该标记会拒绝生成连续实验结论。后续若继续剩余请求，需标为分段回放；
要得到完整连续计时，应从头运行新的 run ID，保留当前数据。

## 独立环境与补丁

两节点为 `192.168.0.63`、`192.168.0.65`，各 8 张 H20 96GB。
模型直接使用两端的 `/data/model/GLM-5.3`，不复制或改写权重。
新环境为 `/data/envs/glm53-vllm-pp2-mtp`，使用 Python 3.12 venv，
基础解释器来自现有 Conda 环境；新环境的包独立安装，不修改旧环境。

- vLLM 0.28.0，Torch 2.13.0，Transformers 5.15.0。
- nvcc/crt/nvvm 固定 13.0.88，CCCL 固定 13.0.85，与 Torch 的
  CUDA runtime 13.0.96 头文件保持同一 13.0 系列。默认解析得到的 nvcc 13.3
  曾导致 DeepGEMM 编译失败，已记录并修复。动态库路径包含基础 Conda
  解释器的 `lib`，以提供 SQLite/ICU 所需的新版 libstdc++。
- 显式设置系统 `gcc/g++`，清除继承的编译 flags；FlashInfer 缓存放在本轮
  独立目录。pip CUDA 包补齐 `lib64 -> lib`、`libcudart.so` 和 `libnvrtc.so`
  链接，仅修改新环境。两节点均通过 SM90 FP8 FlashInfer 内核的单独编译/加载验证。
- 官方基线提交 `2cf0a6915ce544dc493a0990f2ea38d81601128a`。
- 移植 [PR #46994](https://github.com/vllm-project/vllm/pull/46994)，
  上游提交固定为 `bee20119f1bf9f7dd7dccc5311572b3c0a4ef14c`。
- PP/MTP 移植只修改 Python；保持官方 0.28.0 对应的预编译计算内核。
  三处上下文差异手工适配，分别是模型暴露 top-k buffer，FlashAttention/XPU
  稀疏 MLA 在执行时读取 indexer 当前 buffer。
- 修改包括草稿模型的 PP 支持与 embedding 加载、PP 草稿 token 同步、
  广播尺寸匹配、稀疏 MLA buffer 更新。不是只绕过启动检查。
- 补丁：[vllm-0.28.0-pp-mtp.patch](../scripts/patches/vllm-0.28.0-pp-mtp.patch)。
  安装器校验每个原文件及补丁文件哈希，并备份原文件。
- 最新上游 PR 的公开验证对象为 GLM-5.2；本轮 GLM-5.3 属于实验验证。

### 本机 UVA 兼容问题

在真实预热中，CPU top-k=50/top-p=0.9，GPU 却读到 0/0，导致采样索引越界。
独立于模型和 PP 的小测试同样复现：`torch.zeros(pin_memory=True)` 的
`is_pinned()` 返回 False，但 CUDA `cudaHostGetDevicePointer` 成功并返回同一地址。
因此 vLLM 原绑定走了复制分支，得到初始快照，破坏了后续 CPU 更新对 GPU 的可见性。

本轮另加小型 [ATen 映射绑定](../scripts/patches/glm53_uva_view.cpp)，由相同 Torch
版本即时编译；在 `VLLM_GLM53_UVA_FIX=1` 时直接根据 CUDA 的映射结果建立真实别名。
它保留 CPU Tensor 生命周期，不钳制 top-k、不修改请求参数、不跳过预热。
该扩展与 PP/MTP 移植补丁分开记录，最终安装包见 deployment/uva-patch。
临时采样打印已移除，启动脚本清除对应诊断开关。

## 配置与测量

TP8、PP2、DP1，V2 model runner，先验证 MTP K=1。
64K 上下文、max sequences=16、prefill token budget=8192、显存利用率 0.8。
按 [vLLM recipe](https://recipes.vllm.ai/zai-org/GLM-5.3) 使用 FP8 KV；
它与此前 SGLang 的 BF16 KV 不同，因此最终结果是部署方案之间的比较，
不能直接归结为 MTP 单因素加速。

模型 API 监听 `.63` 的 `127.0.0.1:30002`；两节点初始化端口为 29501。
沿用 TCP/eth0、NCCL_IB_DISABLE=1。为读取实际配置，开发诊断路由只随该
loopback API 启用。客户端不需要 API Key，不调用百炼。

复用 20 条会话、376 次请求的冻结 bundle，SHA256：
`8a8b4d97202f388b7719c232d69ec5181091d1b8bc61a8f89c4df36a7a3e60b1`。
固定 high 思考强度、stream、单会话/HTTP 并发、每请求最多 4096 输出 tokens，
无重试、无 trajectory warmup；沿用模型默认 temperature/top_p。
请求没有发送采样 seed；冻结 bundle 的 seed=42 用于样本选择，不等于模型采样种子。
本轮 vLLM 服务 seed=42，已归档的 SGLang `random_seed` 为 742952915，
因此两次生成文本也不能视为相同随机条件下的输出。即使设置相同 seed，跨框架
也不保证逐 token 一致；本轮衡量的是部署方案的单次请求体验。
启动、编译和独立 smoke 不计入轨迹计时。记录正式前后 MTP counters 差值，
同时比较生成 token 数、截断数、无最终答案数和失败数。

### 首轮的文本协议修正

首轮 `r1` 在第 12 个请求返回 `tool_calls` 后被回放校验器中止：11 个成功、
1 个协议失败、364 个未发送。原始记录完整保留，不能用这一轮总耗时比较。
vLLM 0.28 的流式工具解析可解析自发生成的工具标记；这些固定轨迹请求没有
OpenAI `tools`，它们要求 shell 命令作为文本返回。因此移除服务的
`--tool-call-parser glm47 --enable-auto-tool-choice`，保留 `--reasoning-parser glm45`。
这不改变 prompt、采样参数或输出 token 上限。重启后从头运行 `r2`。
首轮另有两次达到输出上限，其中一次为明显重复的 `ls -la`；第二轮继续统计
输出长度和截断，不能把短 smoke 通过视为普遍输出正确性的证明。

## 操作入口

环境安装脚本需在两节点分别执行（第二节点下载依赖时使用已有代理）：

```bash
bash scripts/setup_glm53_vllm_mtp_env.sh
```

在刚安装、尚未打补丁的 0.28.0 环境中，可直接从仓库生成和安装完整补丁包：

```bash
/data/envs/glm53-vllm-pp2-mtp/bin/python scripts/build_glm53_vllm_patch.py /tmp/glm53-vllm-patch
/data/envs/glm53-vllm-pp2-mtp/bin/python scripts/apply_glm53_vllm_mtp_patch.py \
  /tmp/glm53-vllm-patch
/data/envs/glm53-vllm-pp2-mtp/bin/python -m pytest -c /dev/null -q \
  scripts/patches/test_vllm_pp_mtp.py
bash scripts/run_glm53_vllm_mtp_node.sh 0 1 --check-uva
```

本轮两端已安装；已有环境不需要重新生成补丁。实际归档位于
`runs/agentic-replay/vllm-glm53-pp2-mtp-high-20/deployment/uva-patch/`。

GPU 释放并完成两端补丁一致性检查后，在各节点 tmux 内执行：

```bash
# .63
bash scripts/run_glm53_vllm_mtp_node.sh 0 1
# .65（先同步该启动脚本）
bash run_glm53_vllm_mtp_node.sh 1 1
```

通过输出与 MTP 检查后运行；若改变 K，必须同步更新 scenario 的 deployment 字段：

```bash
/opt/miniforge3/envs/test-api/bin/python scripts/run_agentic_replay.py dry-run \
  --config scenarios/vllm_glm53_pp2_mtp_swe_trajectory_high.json --model all
bash scripts/run_glm53_vllm_mtp_trajectory.sh vllm-glm53-pp2-mtp-high-20-r3
/opt/miniforge3/envs/test-api/bin/python scripts/report_glm53_vllm_mtp.py \
  --run-id vllm-glm53-pp2-mtp-high-20-r3
```

正式脚本在发送前检查服务实际模型、TP/PP、V2、上下文和 MTP K。
既有 SGLang 启动脚本和结果保留，可按原 runbook 恢复。
