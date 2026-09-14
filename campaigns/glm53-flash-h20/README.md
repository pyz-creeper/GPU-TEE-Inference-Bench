# GLM-5.3-Flash 本机实验包

完整结论见 [`REPORT.md`](REPORT.md)，聚合数据见 [`performance-summary.csv`](performance-summary.csv) 和 [`performance-summary.json`](performance-summary.json)。

## 启动已验证的 SGLang

```bash
/root/glm53-flash/scripts/start_sglang.sh
```

默认使用全部 8 张 H20，并只监听 `127.0.0.1:30000`。需要远程访问时显式设置 `GLM_HOST=0.0.0.0`，并在外层配置鉴权和防火墙。

烟测：

```bash
curl --noproxy '*' http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"GLM-5.3-Flash","messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":64,"temperature":0}'
```

复跑冻结矩阵：

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
  /opt/miniforge3/envs/glm53-sglang/bin/python \
  /root/glm53-flash/scripts/run_input_bench.py \
  --backend sglang --base-url http://127.0.0.1:30000
```

## vLLM 注意事项

专用镜像和诊断配置已固定在 `configs/vllm-runtime.json`，但该 day-0 镜像在本机生成重复 `lock` token，MTP 启动也会触发 CUDA 越界断言，所以没有正式 vLLM 吞吐结果。

`scripts/start_vllm_docker.sh` 默认 fail-closed。只有复现故障时才执行：

```bash
GLM53_ALLOW_KNOWN_BAD_VLLM=1 /root/glm53-flash/scripts/start_vllm_docker.sh
/root/glm53-flash/scripts/stop_vllm_docker.sh
```

新版镜像可用后，应先通过 `logs/vllm-chat-smoke.json` 同等的语义烟测，再运行：

```bash
/opt/miniforge3/envs/glm53-sglang/bin/python \
  /root/glm53-flash/scripts/run_input_bench.py \
  --backend vllm --base-url http://127.0.0.1:18000/v1
```

## 更新汇总

```bash
/opt/miniforge3/envs/glm53-sglang/bin/python /root/glm53-flash/scripts/build_report.py
```
