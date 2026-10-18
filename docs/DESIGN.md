# Block 级 SLRU 设计

## 接入位置

补丁只替换 `vllm/v1/core/block_pool.py` 并新增 `prefix_retention.py`。原始基线使用参考源码中的 `FreeKVCacheBlockQueue`；SLRU 使用 `SegmentedFreeBlockQueue`，复用同一套 BlockPool 分配、引用计数和 hash 失效流程。其余参考文件用于安装校验及 CPU 测试。

| 事件 | 状态变化 |
|---|---|
| 新缓存块释放 | ref_cnt 归零后进入 probation 队尾 |
| 实际使用缓存命中 | `BlockPool.touch` 记录复用标记；活跃共享块也记录 |
| 命中过的块释放 | 最后一个引用释放后进入 protected 队尾 |
| protected 超出上限 | 队首降级到 probation 队尾，清除复用标记 |
| 分配新块 | 优先 probation 队首；为空时回收 protected 队首 |
| 物理块复用 | 清除旧内容的复用标记，原 BlockPool 清理旧 hash |
| reset | 先检查原有空闲条件、清 hash，再合并队列并清空历史 |

## 不变量

- 可淘汰队列仅包含 `ref_cnt == 0` 的非 null block。
- 活跃块不进入淘汰队列；实际命中信号可以跨活跃区间保留。
- 同一个 block 不能同时进入两个队列。
- 物理 block ID 重用不继承旧内容优先级。
- 忙碌时拒绝 reset，不提前清空策略历史。
- 原 manager 的尾部优先释放顺序在每个队列内保留。

`protected_limit = floor((num_blocks - 1) * protected_ratio)`。默认 2048 blocks、ratio=0.5，对应上限 1023。它限制的是**空闲保护队列**，活跃块暂不占队列名额；保护区不是另一份 GPU KV 存储，也不是不可回收的 pin。

## 复杂度与取舍

两个 `OrderedDict` 和一个按物理 block ID 索引的 bytearray 保存状态。常规队列操作平均 O(1)，reset 为 O(blocks)，元数据为 O(blocks)。Python 容器开销不等于零；`container_bytes_shallow` 也不等于 RSS。

这是以释放顺序近似 recency 的 SLRU 变体。复用信号是“实际命中过”，不是 LFU 的累计频次；没有 ghost queue、TTL、衰减或 Prefill 成本预测。仅有一次性新请求、没有新热点晋升时，旧热点可能长期驻留。

策略以 block 为单位，不维护额外 prefix 树；复杂共享访问下不保证所有保留块都构成完整前缀。因此性能判断使用实际 `cached_tokens`，不能用 protected 队列长度代替有效命中量。

## 测试

`test_retention.py` 从参考源码载入实际 block/queue 类，覆盖引用计数、hash、reset、晋升降级、并发共享、6000 次随机操作和 2400 次 LRU 差分操作。外部 GPU 依赖在 CPU 测试中替换；`--installed` 则载入已安装 vLLM 的对应类。

安装测试覆盖校验拒绝、幂等安装和恢复。协议测试覆盖缓存预检、流式 TTFT、缺失命中明细和错误输入。它们不代替 CUDA 数值正确性验证；设备端 smoke 与性能实验独立进行。
