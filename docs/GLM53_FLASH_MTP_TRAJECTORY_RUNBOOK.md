# 192.168.0.65：GLM-5.3-Flash MTP 轨迹回放

复用 `/opt/miniforge3/envs/glm53-sglang` 和 `/root/glm53-flash/scripts/start_sglang.sh`，
原有源码补丁保留。8×H20，TP=8、EP=8、FP8权重、BF16 KV，DSA TileLang、KDA Triton、
DeepGEMM、自适应 EAGLE MTP 5/1/6。原32K上下文改为64K，以容纳完整冻结输入加4096输出额度。

服务只绑定远端loopback；客户端仍位于 `.63`，通过SSH隧道访问，不需要百炼Key。
SSH加密与节点间网络耗时包含在客户端指标内。

## 启动入口

已启动的服务不要重复启动。远端已有wrapper副本：

```bash
ssh -t 192.168.0.65 'tmux attach -t glm53-flash-mtp'
```

重新部署时，可将仓库脚本复制到远端新路径后运行（先确认旧服务已停、GPU空闲）：

```bash
scp scripts/run_glm53_flash_mtp_server_remote.sh \
  192.168.0.65:/root/glm53-flash/scripts/start_swe_mtp_64k.sh
ssh 192.168.0.65 'tmux new-session -d -s glm53-flash-mtp "bash /root/glm53-flash/scripts/start_swe_mtp_64k.sh"'
```

本机隧道tmux为`glm53-flash-tunnel`；创建命令：

```bash
tmux new-session -d -s glm53-flash-tunnel \
  'env -u LD_LIBRARY_PATH -u LD_PRELOAD /usr/bin/ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -L 127.0.0.1:30001:127.0.0.1:30000 192.168.0.65'
```

tmux可能继承旧Conda环境的动态库路径；隧道进程显式清除这两个变量，避免系统SSH加载不匹配的OpenSSL。

运行同批轨迹（第一次默认r1，重跑必须使用新run-id）：

```bash
bash scripts/run_glm53_flash_mtp_trajectory_tmux.sh sglang-glm53-flash-mtp-high-20-r3
```

当前正式回放tmux：`test-run-glm53-flash-mtp`。脚本核验实际64K、模型名、TP/EP、MTP后才发送请求。
`/health`可能执行内部单token探测，且服务内部有启动warmup；均在正式计时之前。轨迹warmup为0。
不在正在执行的shell脚本上编辑内容；末尾exec Python，完整退出码由runner提供。

## 实验契约与结果

配置：`scenarios/sglang_glm53_flash_mtp_swe_trajectory_high.json`。
仍为`aliyun-high-20/bundle`，SHA-256
`8a8b4d97202f388b7719c232d69ec5181091d1b8bc61a8f89c4df36a7a3e60b1`，20条完整轨迹、376次请求。
固定历史、不开工具；high、stream、max_tokens4096、并发1、重复1、warmup0、retries0。

GLM检查点模板始终开启think，并读取reasoning_effort。当前SGLang将顶层reasoning_effort传入模板，
因此profile只发送`reasoning_effort=high`，不透传无效的enable_thinking开关。
模板会注入`Reasoning Effort: High`。该映射由源码、模板和mock测试验证；实际输出另记录思考文本。
temperature/top_p与云端一样在请求中省略；本地模型默认1.0/0.95，不能保证云端默认值相同。

产物根目录：`runs/agentic-replay/sglang-glm53-flash-mtp-high-20/`。
`launches/<run-id>`含日志与部署快照，`results/<run-id>/GLM-5.3-Flash`含逐请求事件、汇总和Parquet。

与昨晚百炼Flash离线比较：

```bash
python3 scripts/run_agentic_replay.py compare \
  runs/agentic-replay/aliyun-high-20/results/aliyun-high-20-r1/ZHIPU_GLM-5.3-Flash \
  runs/agentic-replay/sglang-glm53-flash-mtp-high-20/results/sglang-glm53-flash-mtp-high-20-r2/GLM-5.3-Flash \
  --output runs/agentic-replay/sglang-glm53-flash-mtp-high-20/comparisons/cloud-vs-remote-r1
```

比值为远端自部署总耗时/百炼总耗时。模型服务、网络、缓存、思考与实际输出量不同，
不归因于TEE，也不是MTP开关对照。缺失/失败请求不计算有效耗时比。

## 部署修正记录

正式运行编号为 `sglang-glm53-flash-mtp-high-20-r2`。r1首请求因server PATH中缺少Conda的ninja而失败，
保存为incomplete并附EXCLUDED.md；375个请求未发送。wrapper已显式加入Conda bin并检查ninja，
保留原有环境和补丁，r2从头回放同批376次请求。没有更改采样或输出参数来绕过错误。
