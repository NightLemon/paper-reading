---
concept: "激活检查点"
aliases: ["activation checkpointing", "activation recomputation", "rematerialization", "gradient checkpointing"]
tags: ["llm-training", "memory", "recomputation", "parallelism"]
papers: ["topics/llm-training/2020-zero", "topics/llm-training/2019-gpipe-pipedream"]
---

# 激活检查点

> **一句话定义**：前向时只保存选定的边界激活，反向需要内部中间值时重新执行对应前向计算，用额外计算换取较少的长期激活存储。

## 为什么需要它

反向计算梯度时通常需要前向输入和中间结果。层数、batch 或序列长度增大后，保存所有中间激活可能比权重本身更占显存。模型状态分片之后，激活还可能成为新的容量限制。

## 它是怎么工作的

```text
前向：保存边界输入 x → 计算区间内各层 → 保存下一边界，丢弃可重算中间值
反向：取回 x → 重跑该区间前向 → 用重建的中间值计算梯度 → 释放临时值
```

区间越长，长期保存的边界可能越少，但重算阶段的工作集和计算成本可能增加。实际策略可以按层、算子或局部区域选择保存点。

### 一个简化的容量模型

对 $L$ 层、每层典型激活大小 $A$ 的顺序网络，若每 $k$ 层保存边界，并在反向逐区间重建，可以粗略估计保存与局部工作集为：

$$M_{act}\approx\left(\frac{L}{k}+k\right)A.$$

在层大小相近等理想条件下，$k\approx\sqrt L$ 给出 $O(\sqrt L A)$ 的量级。这个教学模型不包括跳连、大矩阵临时值、不同层形状、流水中多个在飞 microbatches，不能直接作为所有 Transformer 的显存计算器。

### 与分片、offload 的区别

| 手段 | 改变什么 | 主要额外代价 |
| --- | --- | --- |
| checkpointing | 哪些中间值需要长期保存 | 重算 |
| checkpoint 分片 | 同一个保存值由哪些 rank 分担 | 使用前 gather |
| checkpoint offload | 保存值驻留在哪个存储层 | CPU / 设备等数据传输 |

ZeRO-R 可以先只保留 checkpoints，再在 MP 组内分片，需要时 all-gather；还可以将分片搬到 CPU。它们是可组合的机制，不能把三者的节省和成本混成一个固定百分比。

### 重算必须保持前向语义

dropout 等随机操作需要恢复对应随机状态；有副作用的 buffer 更新不能在重算时无意重复；前向与重算使用的权重版本也需匹配。流水训练中可能同时存在多个在飞批次，这些要求尤其重要。

计算图的数学等价不保证浮点结果逐位相同。是否保持目标训练行为，还要结合随机性、数值精度和模型中的有状态层验证。

## 关键性质与代价

| 收益 | 代价与条件 |
| --- | --- |
| 减少长期保存的中间激活 | 额外前向计算与调度 |
| 可支持更大 batch 或更长序列 | 增大的全局 batch 可能改变收敛行为 |
| 与模型状态分片互补 | 仍需容纳重算区间的瞬时工作集 |
| 可与 checkpoint 分片 / offload 叠加 | 增加通信或存储层传输，不一定提高速度 |

## 常见误解

- **误解**：checkpointing 把梯度存到磁盘 → **实际**：这里保存的是反向所需的激活边界，与训练断点文件不同。
- **误解**：只要保存很少，整个训练就几乎不占激活显存 → **实际**：重算与反向仍需要临时激活工作集。
- **误解**：固定增加 33% 时间 → **实际**：这取决于重算区域、算子效率与通信；FLOPs 比例也不等于墙钟比例。
- **误解**：checkpointing 与状态分片二选一 → **实际**：二者分别减少激活和参数相关状态。
- **误解**：重跑前向天然与第一次一样 → **实际**：随机数、权重版本和副作用必须保持一致。

## 出现在哪些论文里

- [GPipe / PipeDream](../topics/llm-training/2019-gpipe-pipedream/README.md) —— GPipe 保存 stage 边界激活、反向重算 cell 内部前向，以降低多微批的保存量；PipeDream 的容错 checkpoint 与此含义不同，重算还需匹配相应权重版本。

- [ZeRO](../topics/llm-training/2020-zero/README.md) —— ZeRO-R 在 MP 组内分片保存 checkpoints，重算前 gather，并选择性 CPU offload。§6.1 对 100B、batch 32、序列长 1024、MP16、每 Transformer 层一个输入 checkpoint 的估算约为 33 GB → 2 GB/设备；这个量不包含全部瞬时工作集。

## 延伸阅读

- [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174)：重计算与激活存储取舍的经典工作。
- [ZeRO 状态分片](zero-state-sharding.md)：先分清显存消耗属于模型状态还是激活。
- [IO-aware 算法设计](io-aware-kernel-design.md)：某些重计算还能减少昂贵的数据搬运，额外 FLOPs 不必对应同比时间增长。
