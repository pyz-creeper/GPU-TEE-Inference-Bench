# DeepSeek-V4-Flash vLLM 裸金属复跑记录

记录时间：2026-09-03 UTC
记录来源：停服前从 CVM 运行中进程 `/proc/436032/{cmdline,environ}`、tmux 和软件环境直接采集。

## 停服状态

2026-09-03 15:36 UTC 前后，向 `vllm-server` tmux pane 发送 `Ctrl-C`，vLLM 完成优雅退出。核验结果：

- tmux `vllm-server` 会话不存在；
- APIServer、EngineCore、TP0--TP7 worker 均已退出；
- guest `0.0.0.0:8000` 无监听；host 转发端口 `127.0.0.1:18000` 不可达；
- 8 张 H800 均无 compute process，显存占用 `0 MiB`、GPU utilization `0%`；
- 虚拟机本身没有关闭；日志、profile 和模型文件均保留。

## 实际启动环境

服务使用的可执行文件来自独立 venv，而不是 conda 环境中的 vLLM：

| 项目 | 值 |
|---|---|
| venv | `/data/venvs/vllm025` |
| Python | 3.12.14（conda-forge build） |
| vLLM | 0.25.0 |
| PyTorch | 2.11.0+cu130 |
| PyTorch CUDA | 13.0 |
| Transformers | 5.16.1 |
| CUDA nvcc | 13.0.88 |
| NVIDIA driver | 580.159.03 |
| NCCL Python package | `nvidia-nccl-cu13==2.28.9` |
| FlashInfer | 0.6.13 |
| Triton | 3.6.0 |
| TileLang | 0.1.9 |

进程继承了 `CONDA_DEFAULT_ENV=vllm`，但这不决定实际 runtime；启动命令显式指定了如下环境和绝对路径：

```bash
VLLM_DEEP_GEMM_WARMUP=skip
PATH=/data/venvs/vllm025/lib/python3.12/site-packages/nvidia/cu13/bin:/data/venvs/vllm025/bin:/usr/local/cuda/bin:/usr/bin:/bin
LD_LIBRARY_PATH=/data/venvs/vllm025/lib/python3.12/site-packages/torch/lib:/data/venvs/vllm025/lib/python3.12/site-packages/nvidia/cu13/lib:/usr/lib/x86_64-linux-gnu
```

主要 CUDA 包版本：

```text
nvidia-cuda-cccl==13.0.85
nvidia-cuda-crt==13.0.88
nvidia-cuda-cupti==13.0.85
nvidia-cuda-nvcc==13.0.88
nvidia-cuda-nvrtc==13.0.88
nvidia-cuda-runtime==13.0.96
nvidia-nccl-cu13==2.28.9
```

## CVM 实际服务命令

```bash
env \
  VLLM_DEEP_GEMM_WARMUP=skip \
  PATH=/data/venvs/vllm025/lib/python3.12/site-packages/nvidia/cu13/bin:/data/venvs/vllm025/bin:/usr/local/cuda/bin:/usr/bin:/bin \
  LD_LIBRARY_PATH=/data/venvs/vllm025/lib/python3.12/site-packages/torch/lib:/data/venvs/vllm025/lib/python3.12/site-packages/nvidia/cu13/lib:/usr/lib/x86_64-linux-gnu \
  /data/venvs/vllm025/bin/vllm serve /models/DeepSeek-V4-Flash-0731 \
    --served-model-name DeepSeek-V4-Flash-0731 \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 32768 \
    --trust-remote-code \
    --tokenizer-mode deepseek_v4 \
    --reasoning-parser deepseek_v4 \
    --kv-cache-dtype fp8 \
    --block-size 256 \
    --enable-expert-parallel \
    --moe-backend marlin \
    --linear-backend deep_gemm \
    --profiler-config '{"profiler":"torch","torch_profiler_dir":"/data/vllm-profiles/v025-off-shapes","torch_profiler_with_stack":false,"torch_profiler_record_shapes":true,"torch_profiler_with_memory":false,"torch_profiler_with_flops":false,"torch_profiler_dump_cuda_time_total":false,"torch_profiler_use_gzip":true}'
```

该命令通过 tmux 启动，并把 stdout/stderr 追加到 `/data/vllm-logs/v025-tmux.log`。仓库中的 `scripts/start_dsv4_vllm.sh` 封装了同一组参数。它不依赖当前是否 activate conda，且只使用 vLLM，不使用 TensorRT-LLM。

## 模型与硬件身份

- 模型目录：`/models/DeepSeek-V4-Flash-0731`
- 8 × NVIDIA H800，每卡 81,559 MiB；GPU 间拓扑在 guest 中均显示 `NV8`
- guest kernel：Ubuntu `6.8.0-134-generic`
- guest CPU：128 vCPU，1 socket，64 cores/2 threads，1 NUMA node
- guest 内存：1.0 TiB，无 swap

模型关键文件校验值：

```text
6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023  config.json
5fccff80f55a4d455bbe516bdd552edf3e9623df95e99fbf2a3c3389fdf91af0  generation_config.json
6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547  tokenizer_config.json
```

推理配置：TP=8、expert parallel enabled、FP8 KV cache、block size 256、max model length 32768、DeepGEMM dense linear、Marlin MoE。运行日志确认 CUDA Graph 为 `FULL_AND_PIECEWISE`、prefix caching 开启、speculative decoding 关闭。

## 裸金属启动与复跑

先确认裸金属中的 venv 和模型目录与上面一致；若路径不同，通过环境变量覆盖，不要修改冻结 workload：

```bash
cd /home/user007/input-bench

# 默认路径与 CVM 完全一致。
scripts/start_dsv4_vllm.sh

# 示例：裸金属 venv 路径不同时。
VLLM_VENV=/path/to/vllm025 scripts/start_dsv4_vllm.sh

tmux attach -t vllm-server
curl --noproxy '*' http://127.0.0.1:8000/health
```

健康检查返回 200 后，先跑不启 profiler 的 performance suite，再跑由 `/start_profile`、`/stop_profile` 控制的 profile suite：

```bash
cd /home/user007/input-bench
scripts/run_dsv4_campaign.sh baremetal performance http://127.0.0.1:8000
scripts/run_dsv4_campaign.sh baremetal profile http://127.0.0.1:8000
```

不要再次执行 `prepare`。CVM/裸金属必须复用同一个 `campaigns/dsv4-large-scale/plan.json` 和 `workloads/`；冻结 plan SHA-256 为 `f61d1d6b47114f5de17d3939b9505e5843d364988d820fa051d2cc80e6a8a0fa`。

裸金属实验结束后可优雅停服：

```bash
cd /home/user007/input-bench
scripts/stop_dsv4_vllm.sh
```

## 对照注意事项

- 保持上述服务参数、软件版本、GPU power/clock policy、workload 顺序和客户端位置一致；否则不能把差异只归因于 CVM。
- `VLLM_DEEP_GEMM_WARMUP=skip` 会留下明显的首次 shape/JIT 效应；必须同时保留并比较 r1/r2。
- profiler 仅在 profile suite 内开启。CUPTI 在本次 CVM 中不可用，因此 CVM trace 只有 CPU/user annotation；裸金属端若 CUPTI 可用，应保留 CUDA kernel、memcpy 和 NCCL activity。
- CVM 完整结果和 profile trace 已复制到 `campaigns/dsv4-large-scale/results/`，不依赖虚拟机继续运行。

## 2026-09-03 裸金属实际执行记录

裸金属模型位于 `/data/models/DeepSeek-V4-Flash-0731`。实际启动命令为：

```bash
cd /home/user007/input-bench
VLLM_VENV=/data/venvs/vllm025 \
MODEL_PATH=/data/models/DeepSeek-V4-Flash-0731 \
WORK_DIR=/home/user007/input-bench \
PROFILE_DIR=/data/vllm-profiles/v025-baremetal-shapes \
LOG_FILE=/data/vllm-logs/v025-baremetal-tmux-attempt2.log \
scripts/start_dsv4_vllm.sh
```

`scripts/start_dsv4_vllm.sh` 已显式把 `CUDA_HOME` 和 `CUDA_PATH` 指向 venv 内的 CUDA 13.0，避免 DeepGEMM/TileLang JIT 误用裸金属 `/usr/local/cuda` 的 CUDA 13.1。

当前裸金属 glibc 2.43 与 CUDA 13.0 的 `rsqrt/rsqrtf` 声明存在 exception-specification 冲突。隔离 venv 中的以下 header 已给两个声明增加 `noexcept(true)`：

```text
/data/venvs/vllm025/lib/python3.12/site-packages/nvidia/cu13/include/crt/math_functions.h
```

性能和 profile suite 均已成功完成：

```text
performance: results/20260903T164113Z-baremetal-performance
profile:     results/20260903T172455Z-baremetal-profile
```

完整结果解读见 `BAREMETAL_REPORT.md`。8 个包含 CUDA activity 的 TP trace 已复制进 profile 结果目录的 `server-traces/`，关闭服务不会丢失这些产物。
