# 面向 Jetson 的 vLLM Prefix KV Cache 保留策略优化

整理日期：2026-10-15。项目英文名：Workload-aware Prefix KV Cache Retention for vLLM on Jetson。

**项目结论**

针对边缘 GPU 上 KV cache 容量受限、重复 prefix 被淘汰后需要重新 prefill 的问题，在 vLLM 0.9.2 V1 的 BlockPool 中实现 block 级 SLRU 保留策略。利用实际缓存复用记录，将可淘汰 block 分为 probation（观察区）和 protected（保护区），优先保留已经被复用的内容。

在相同模型、缓存容量和请求轨迹下，三个 Zipf seed 的长测中，SLRU 相比 LRU 将重复 prefix 重算减少 **12.09%**；本次长测平均 TTFT 从 **2.826 秒降至 2.583 秒，降低 8.59%**。另一次 LRU → SLRU → LRU 短测中，平均 TTFT 降低 **14.81%–14.96%**。

**1. 问题与优化目标**

Prefix cache 可以复用相同 token 前缀的 KV，但容量不足时必须回收缓存块。如果后续会被反复访问的前缀被淘汰，请求就需要重新执行 prefill，增加首 token 延迟。

前期 Hot–Cold 压力实验观察到：固定长 HOT prefix 在插入随机 COLD 请求后，回访 TTFT 从约 0.246 秒增加到 21.393 秒。随后用 Zipf 合成轨迹验证容量影响：在早期 100 请求 pilot 中，2048、4096、10414 blocks 下，重复请求完整 prefix 命中率分别约为 53.73%、80.60%、100%。这些 pilot 用于确认问题与容量的关系，不作为新策略的主要收益数字。

本次优化目标是：在相同 KV 容量下，更好地保留有复用记录的 prefix block，减少重复计算，使缓存收益体现为更低的平均 TTFT。

**2. 实现方式**

基线保留参考快照版本的 `FreeKVCacheBlockQueue`，即 LRU-like block 淘汰顺序。通过策略配置，在同一份补丁中切换 LRU 和 SLRU，避免用完全不同的实现环境作为对照。

| 路径 | SLRU 行为 | 目的 |
|---|---|---|
| block 首次缓存、释放 | 进入 probation 队尾 | 为尚无复用证据的内容设置观察期 |
| 实际缓存命中 | 在 BlockPool.touch 路径记录复用标记 | 以真正用于请求的缓存命中作为信号 |
| 被复用 block 最后一个使用者释放 | 引用计数归零后进入 protected 队尾 | 保护已有复用记录的内容 |
| protected 超出上限 | 将 protected 队首 block 降级到 probation 队尾 | 限制保护区规模，保留更新能力 |
| 申请新 block | 优先从 probation 队首取块；为空时才从 protected 取块 | 减少未复用内容对受保护内容的淘汰 |
| 物理 block 被重新分配 | 清空旧内容的复用标记 | 防止新内容继承旧内容的保留优先级 |
| reset_prefix_cache | 在原有空闲检查和 hash 清理后，清空保留历史与计数器 | 保证缓存重置语义和实验可重复性 |

实现由两个 OrderedDict、一个按 block ID 索引的复用标记数组及统计计数器组成。复用信号是“是否发生过实际命中”，不是累积访问频率，也不是对未来请求的直接预测。

每个 block 的常规入队、出队、移除、晋升和降级依靠常数时间的容器操作，不需要扫描整个缓存或计算全局排序；清空历史属于线性操作。元数据规模为 O(blocks)。这些是实现结构的复杂度说明，不等同于已经测得 CPU 开销或内存开销为零。

保护比例设为 0.5，2048 个物理块扣除 null block 后，protected 上限为 `floor(2047 × 0.5) = 1023`。两个队列只保存当前可淘汰的块：正在被请求使用的 block 不进入淘汰队列。保护区是逻辑保留规则，不是额外分配一份 KV 显存。

本次实现采用经典 SLRU 思路，重点在与 vLLM block 生命周期集成。没有实现按 prefill 毫秒成本打分的 cost-aware 算法；项目名称和介绍使用“复用感知／workload-aware retention”更准确。

**3. 主要代码与工程工作**

| 组件 | 本次工作 |
|---|---|
| prefix_retention.py | 实现 SegmentedFreeBlockQueue、复用标记、保护区上限、晋升/降级及策略计数器 |
| block_pool.py | 接入可选策略、实际命中记录和重置历史，复用原有分配、释放和 hash 失效路径 |
| manage_patch.py | 源文件 SHA256 校验、补丁安装、备份与恢复 |
| start_server.sh | 固定模型、block 数与关键引擎配置，记录实际策略和启动参数 |
| zipf_prefix_bench.py | 构造确定性 prefix pool/请求轨迹，记录逐请求 cached_tokens、重算 tokens、TTFT 和输入指纹 |
| compare_results.py | 检查两组配置及 workload 哈希后比较结果 |
| run_aba.py / collect_jetson.py | 自动运行 ABA 短测，并在宿主机采集频率、温度、功率轨和进程状态 |

主要改动集中在 BlockPool 与新增保留队列，没有新增调度算法或改写 attention kernel。本次收益来自复用已有 KV、减少 prefill 工作量。

策略配置：

```bash
VLLM_PREFIX_RETENTION_POLICY=lru   # 基线
VLLM_PREFIX_RETENTION_POLICY=slru  # 本次策略
VLLM_PREFIX_RETENTION_PROTECTED_RATIO=0.5
```

本地验证记录共 38 项通过，涵盖运行时生命周期、安装/恢复、模拟 HTTP benchmark、结果比较和 smoke 协议。它们是源码级及模拟服务验证。设备端另外完成了缓存预检与实际推理 benchmark；本次三组长测测量请求全部成功，未发生 preemption。

**4. 实验配置**

| 项目 | 配置 |
|---|---|
| 硬件 | Jetson AGX Orin 64GB |
| 功耗配置 | 本次遥测开始时查询为 MODE_30W |
| 推理引擎 | vLLM 0.9.2 V1 |
| 模型 | Qwen3-4B-Instruct，服务名 qwen3-4b |
| 精度与执行方式 | BF16，enforce-eager |
| Prefix caching | 开启 |
| Block size | 16 tokens |
| GPU blocks | 2048，服务日志报告 32768-token KV 容量 |
| 最大序列长度 | 16384 |
| 引擎 max-num-seqs | 8 |
| 测试实际并发 | 1 |
| 每请求输出 | 1 token |
| Prefix pool | 100 个固定 prefix，正文各 3072 tokens |
| Pool seed | 20261013 |
| 请求分布 | Zipf，alpha=1.0 |
| 长测 seed | 42、43、44 |
| 每组长测 | 预热 200 次，正式测量 1000 次 |
| SLRU protected ratio | 0.5 |

这是具有热门与长尾特征的合成 workload，不宣称来自真实生产请求采样。该 Zipf 设置下 P0 理论概率约 19.28%，不是早期讨论中的示例概率 40%。不同 prefix 的可复用 block 前缀经过校验，不依靠共同模板制造虚假的大规模共享。

对照组使用相同 prefix pool、同 seed 的请求顺序与输入 token 指纹。预热请求不计入收益指标。三个 seed 的长测主对照中，LRU 和 SLRU 各有 3000 个正式测量请求。

**5. 长测主要结果**

主对照采用已完成的 LRU 基线与 2026-10-15 的 SLRU 长测复测，两组配置及请求轨迹一致。

| 指标 | LRU | SLRU | 变化 |
|---|---:|---:|---:|
| 重复 prefix 重算 tokens | 5,121,840 | 4,502,848 | **减少 12.09%** |
| Saved prefill tokens | 3,759,312 | 4,378,304 | 增加 618,992 tokens |
| Token 命中率 | 40.33% | 46.97% | 提高 6.64 个百分点 |
| 重复请求完整命中率 | 41.20%（1191/2891） | 47.28%（1367/2891） | 提高 6.09 个百分点 |
| 平均 TTFT | 2.826 s | 2.583 s | **降低 8.59%** |
| 单并发请求完成速率 | 0.3539 req/s | 0.3871 req/s | **提高 9.39%** |

平均 TTFT 按请求数加权。百分比按未四舍五入的原始数值计算。

请求完成速率按三个 seed 的总成功请求数除以总测量时长计算：LRU 为 3000 / 8477.777 秒，SLRU 为 3000 / 7750.018 秒。这里是单并发闭环 workload 的实际完成速率，不代表服务的最大吞吐能力。每请求输出 1 token，因此本次输出 tokens/s 数值相同，也不能据此声称 decode kernel 速度提升。

| seed | LRU 重算 tokens | SLRU 重算 tokens | 重算减少 | LRU 平均 TTFT | SLRU 平均 TTFT | TTFT 降低 |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 1,762,352 | 1,578,160 | 10.45% | 2.903 s | 2.777 s | 4.33% |
| 43 | 1,695,648 | 1,461,600 | 13.80% | 2.832 s | 2.466 s | 12.92% |
| 44 | 1,663,840 | 1,463,088 | 12.07% | 2.742 s | 2.505 s | 8.62% |

两次 SLRU 长测中，包含预热的全部 3600 条请求的 cached_tokens 逐条一致。这说明本 workload 下保留策略的缓存行为稳定、可以复现。

**6. ABA 短测结果**

另一次短程诊断按 LRU → SLRU → LRU 顺序运行，每阶段重新启动服务。固定 seed=42，每阶段 Zipf 预热 50 次、测量 100 次；此外进行 30 次单独的零命中冷请求校准。

| 指标 | 第一次 LRU | SLRU | 第二次 LRU |
|---|---:|---:|---:|
| 冷请求 TTFT p50 | 4.593 s | 4.590 s | 4.580 s |
| Zipf 平均 TTFT | 2.881 s | 2.454 s | 2.886 s |
| 重复 prefix 重算 tokens | 101,728 | 71,712 | 101,728 |
| 重复请求完整命中率 | 52.78% | 66.67% | 52.78% |

两次 LRU 平均 TTFT 相差 0.17%；SLRU 相对前后两轮 LRU 分别降低 14.81% 和 14.96%，重算减少 29.51%。短测与长测使用不同的预热长度和轨迹窗口，结果分别报告，不混合计算收益。

对第一轮 LRU 与 SLRU 的 100 条测量请求逐条配对：12 条请求在 SLRU 下命中更多，2 条命中更少，86 条命中量相同。命中增加的 12 条合计节省约 46.57 秒，命中减少的 2 条增加约 3.71 秒，命中相同的 86 条合计只增加约 0.18 秒。这说明平均 TTFT 改善与缓存复用增加相符。

**7. 热门保护与策略取舍**

| 前缀组的重复重算 tokens | LRU | SLRU | 变化 |
|---|---:|---:|---:|
| P0–P9 | 1,814,288 | 1,107,888 | **减少 38.94%** |
| P10–P99 | 3,307,552 | 3,394,960 | 增加 2.64% |

P0–P9 是生成器预设的热门前缀组。SLRU 更好地保护了它们，同时让长尾承担少量额外重算；净结果是总重算减少。这是策略取舍，不应包装成所有请求都会变快。

本次三个 seed 的 p95 相对 LRU 有升有降，没有一致改善。单并发、单输出 token 的测试主要验证 prefix 保留与 prefill 对 TTFT 的影响，未据此宣称提升了最大并发吞吐或长输出 decode 性能。Python 队列元数据和维护开销尚未单独定量，不将“常数时间操作”写成“零开销”。

**8. 硬件观察与实验记录**

本次长测的硬件采样完整覆盖三个测量窗口。零命中请求内部的 GPU 频率中位数为 604、610、610 MHz，EMC 均为 3199 MHz，GPU 温度最高约 53.53°C，未发现随测试持续进行而出现持续 GPU 低频的趋势。

设备存在短时后台 CPU 负载与请求延迟尖峰，服务日志中也出现 TCPStore 检查告警；相关证据已保存在独立诊断报告中，原因未全部定位。它们作为运行时观测记录保留，不将“已经证明降频”作为项目结论。

为保留全部实验历史，首次 SLRU 长测的合并平均 TTFT 为 2.839 秒，本次复测为 2.583 秒，两次命中结果逐条一致。本文使用本次复测作为主要延迟结果，并明确收益属于该轮测量；之前的结果和原始日志继续保留。

**9. 指标口径**

- Saved prefill tokens：正式测量请求的 reported cached_tokens 之和，并与 Prometheus hit 增量核对。
- Token 命中率：saved prefill tokens / 总 prompt tokens。分母包含请求模板和后缀，不能与完整 prefix 命中率混用。
- 重复请求：该 prefix 在本次运行中已有请求完成后再次被请求，包含预热建立的访问历史。
- 重复 prefix 重算 tokens：仅对重复请求统计，可复用 prefix tokens 减去命中的 prefix tokens；不将所有首次访问必需计算算作策略造成的重算。
- 重复请求完整命中率：重复请求中完整命中可复用 prefix 的请求数 / 重复请求总数。
- TTFT：流式客户端从发出请求到收到第一个实际生成 token 的时间，包含请求路径开销，不等于单个 GPU kernel 的执行时间。


**公开材料**

- [代码与运行入口](../README.md)
- [已完成实验摘要](../benchmarks/README.md)
- [无需 GPU 的结果复算](../scripts/summarize_results.py)

硬件原始进程日志未上传；本文包含其汇总观察。正式实验汇总保留在 benchmarks/，历史 SLRU 长测也单独保留。
