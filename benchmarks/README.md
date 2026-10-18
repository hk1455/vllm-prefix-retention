# Published benchmark summaries

这些文件来自已完成的 Jetson 实验；不是模拟生成的性能数据。

| 目录 | 内容 |
|---|---|
| `published/lru_s*.json` | 三个 seed 的正式 LRU 基线 |
| `published/slru_s*.json` | 2026-10-15 三个 seed 的 SLRU 长测 |
| `history/slru_s*.json` | 较早一次 SLRU 长测，独立保留 |
| `aba/` | LRU → SLRU → LRU 的 Zipf 和零命中冷请求汇总 |
| `provenance.json` | 源压缩包 SHA256、成员路径到公开文件的映射 |

数值测量、workload 指纹、客户端配置和 Prometheus 增量保持原值；删除 endpoint/output 路径及自由文本日志字段。原始模型响应、机器进程列表、完整服务日志和压缩包不随仓库发布。本目录足够复算汇总指标，不包含逐请求记录，不能独立重做全部逐请求和硬件时间对齐分析。

```bash
python3 scripts/summarize_results.py
python3 compare_results.py benchmarks/published/lru_s42.json benchmarks/published/slru_s42.json
```

主对照每种策略 3 × 1000 个测量请求，每 seed 另有 200 次预热。平均 TTFT 按请求数加权；速率用成功请求总数除以测量总时长，不能直接平均各 seed 的速率；完整命中率按请求数还原分子后合并，不能把三个 p95 求平均当作整体 p95。

重复 prefix 重算只统计已有完成访问历史的请求；首次访问的必需计算不计入该指标。Token 命中率分母包含模板与后缀的全部 prompt tokens。

早期 SLRU 长测平均 TTFT 为 2.839 s，2026-10-15 为 2.583 s；两次逐请求 cached_tokens 在私有原始记录核查中一致。主表采用后一次测量，保留历史以便区分策略命中行为和不同运行的执行时间。全部数字的适用范围见 [实验报告](../docs/RESULTS_ZH.md)。
