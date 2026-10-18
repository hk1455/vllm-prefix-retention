# vLLM Prefix KV Cache Retention on Jetson

**在 vLLM V1 BlockPool 中实现基于复用记录的 SLRU 保留策略，减少有限 KV 容量下的重复 Prefill。**

Block-level SLRU retention for vLLM on an NVIDIA Jetson AGX Orin: a runtime patch, deterministic Zipf benchmark, and measured results. Tested against the supplied **vLLM 0.9.2 V1** source snapshot; this is an experimental integration, not an upstream vLLM feature.

## 已测得的收益

Jetson AGX Orin 64GB / Qwen3-4B-Instruct / BF16 / 2048 KV blocks / 每块 16 tokens。固定请求轨迹，三个 seed，每种策略共 **3000 个测量请求**。

| 指标 | 默认 LRU-like | SLRU | 改善 |
|---|---:|---:|---:|
| 重复 prefix 重算 tokens | 5,121,840 | 4,502,848 | **减少 12.09%** |
| Saved prefill tokens | 3,759,312 | 4,378,304 | 增加 618,992 |
| Token 命中率 | 40.33% | 46.97% | +6.64 个百分点 |
| 重复请求完整命中率 | 41.20% | 47.28% | +6.09 个百分点 |
| 平均 TTFT | 2.826 s | 2.583 s | **降低 8.59%** |
| 单并发请求完成速率 | 0.3539 req/s | 0.3871 req/s | **提高 9.39%** |

这是 **合成 Zipf、并发 1、每请求输出 1 token** 的结果；请求完成速率不代表最大服务吞吐或 decode kernel 加速。p95 未出现一致改善。热门 P0–P9 的重复重算减少 38.94%，长尾 P10–P99 增加 2.64%。

另一次 LRU → SLRU → LRU 短测中，两次 LRU 平均 TTFT 接近，SLRU 平均 TTFT 分别降低 14.81% / 14.96%。完整配置、每个 seed 的数字和实验历史见 [实验报告](docs/RESULTS_ZH.md)，可公开复算的数据见 [benchmarks](benchmarks/README.md)。

## 做了什么

KV cache 容量不足时，新的长 prompt 会挤出之后仍会复用的 prefix；再次访问时需要重新 prefill。本项目在原有 block 分配、引用计数和 hash 管理路径内改变可淘汰块的保留顺序。

- **策略实现**：probation / protected 双队列；实际命中后记录复用，最后一个使用者释放后进入保护区。
- **生命周期集成**：保护活跃块；物理块重新分配时清空旧复用标记；缓存 reset 同步清空策略历史。
- **可恢复安装**：SHA256 校验精确源码、备份、幂等安装和拒绝覆盖未知修改的回滚工具。
- **实验工具**：固定 prefix pool 与请求序列、逐请求命中与 TTFT、输入哈希校验、模拟 HTTP 测试和 Jetson 遥测。

实现采用经典 SLRU 思路。没有新增调度算法、修改 attention kernel，也没有使用未来请求或运行时不可获得的 prefix popularity 标签。它不是按 Prefill 毫秒成本评分的预测器。设计与边界见 [DESIGN](docs/DESIGN.md)。

## 快速开始

在已能运行目标模型的 **vLLM 0.9.2 环境或容器**中执行。现有 vLLM 源码必须匹配 [manifest.json](manifest.json)，版本号相同不保证源码一致。

```bash
git clone https://github.com/hk1455/vllm-prefix-retention.git
cd vllm-prefix-retention
python3 -m pip install -r requirements-bench.txt

# 安装前先停止该环境的 vLLM 服务。
python3 manage_patch.py check
python3 scripts/run_tests.py
python3 manage_patch.py install
python3 tests/test_retention.py --installed
```

终端 A 启动基线：

```bash
MODEL=/models/Qwen3-4B-Instruct bash start_server.sh lru 2048
```

终端 B 等待服务 ready 后执行：

```bash
python3 smoke_server.py --label lru --strict-output
RUN_ID=pilot_r1 bash run_bench.sh lru 2048 42
```

停止终端 A 的 LRU 服务，再启动 SLRU：

```bash
MODEL=/models/Qwen3-4B-Instruct bash start_server.sh slru 2048
```

终端 B：

```bash
python3 smoke_server.py --label slru --strict-output
RUN_ID=pilot_r1 bash run_bench.sh slru 2048 42
python3 compare_results.py \
  results/lru_b2048_s42_pilot_r1 \
  results/slru_b2048_s42_pilot_r1
```

**客户端的 `lru/slru` 参数只是标签，不会切换服务。** 实际策略以服务器的 `PREFIX_RETENTION policy=...` 日志为准。完整长测、容器、遥测和回滚说明见 [运行指南](docs/RUNNING.md)。

无需 GPU 复算已发布结果：

```bash
python3 scripts/summarize_results.py
```

## 目录

| 路径 | 内容 |
|---|---|
| [payload/prefix_retention.py](payload/prefix_retention.py) | SLRU 队列、复用记录与统计 |
| [block_pool.patch](block_pool.patch) | BlockPool 接入改动的 diff |
| [manage_patch.py](manage_patch.py) | 校验、安装、备份、恢复 |
| [reference/](reference/) | 保留许可证的 vLLM 原始源码快照 |
| [zipf_prefix_bench.py](zipf_prefix_bench.py) | 固定轨迹与逐请求测量 |
| [diagnostics/](diagnostics/) | ABA 自动化与宿主机遥测 |
| [tests/](tests/) | CPU 生命周期、安装、协议与结果比较测试 |
| [benchmarks/](benchmarks/README.md) | 脱敏汇总、来源哈希及历史结果 |
| [docs/](docs/) | 设计、复现说明与实验报告 |

## 验证与限制

本次整理的 **45 项 CPU / 模拟服务测试通过**，见 [验证记录](docs/VALIDATION.json)。它们和已完成的 Jetson 推理实验分别记录；GitHub Actions 仅运行不依赖 GPU 的检查。当前补丁针对特定源码快照，不承诺适配其他 vLLM 版本。保护比例固定为 0.5；不声称所有 workload 最优。队列操作平均 O(1)，元数据 O(blocks)，其 CPU / RSS 开销未单独定量。

## License

[Apache-2.0](LICENSE)。vLLM 源码保留原始 SPDX 与版权信息，来源和修改范围见 [NOTICE](NOTICE)。仓库不包含模型权重。
