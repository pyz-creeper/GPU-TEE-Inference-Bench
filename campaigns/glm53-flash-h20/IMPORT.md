# Node1 导入记录

- 来源主机：`root@192.168.0.65`（hostname `iv-yer00g5vr44c5qwvh1lz`）
- 来源路径：`/root/glm53-flash/`
- 本机归档路径：`/root/GPU-TEE-Inference-Bench/campaigns/glm53-flash-h20/`
- 导入日期：2026-09-06
- 来源文件：278 个普通文件，合计 17,848,178 bytes
- 传输方式：保留目录、时间和权限的 rsync archive copy
- 校验：导入后对来源和本机的 278 个文件逐文件计算 SHA-256，清单无差异

`IMPORT.md` 和 `ANALYSIS.md` 是首次导入校验后在本机新增的说明文件。随后按相同 Mooncake 五点口径完成的新测试归档在 `mooncake-mtp/`；其中原始 campaign 文件来自 node1 的 `/data/benchmarks/glm53-flash-mooncake-h20-mtp`，`mooncake-mtp/REPORT.md` 是拉回后新增的分析报告。首次导入的其余内容保持来源文件字节不变。

原始脚本中的工作目录仍固定为 node1 的 `/root/glm53-flash`。本目录首先是可审计结果归档，不应直接假定其中脚本已完成路径重定位。
