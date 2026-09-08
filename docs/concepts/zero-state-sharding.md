---
concept: "ZeRO 状态分片"
aliases: ["Zero Redundancy Optimizer", "optimizer state sharding", "ZeRO-1/2/3"]
tags: ["llm-training", "data-parallelism", "memory", "optimizer"]
papers: ["topics/llm-training/2020-zero", "topics/llm-training/2019-gpipe-pipedream"]
---

# ZeRO 状态分片

> **一句话定义**：让数据并行组里的不同 rank 分别持有和更新一部分训练状态，计算时按需通信，使每个 rank 不必长期保存整套优化器、梯度和参数副本。

## 为什么需要它

普通复制式数据并行只切样本，每个 rank 仍存完整模型状态，增加设备数不能降低这部分单卡显存。Adam 主参数与两矩可能比用于前后向的低精度权重还大。

这些状态有明确的 owner 和生命周期：某个参数的更新只需相应梯度和历史状态，其余 rank 无需重复执行同一份更新。去掉这种复制，可以保留按完整样本计算的粒度。

## 它是怎么工作的

设参数量 $\Psi$、DP 度 $N_d$。在 FP16 参数 / 梯度、FP32 主参数与 Adam 两矩的特定配置下，每参数共 16 bytes：

| 阶段 | 分片保存 | 每 rank 长期模型状态 |
| --- | --- | --- |
| 标准 DP | 无 | $16\Psi$ |
| ZeRO-1 | 优化器状态（含 FP32 主参数） | $4\Psi+12\Psi/N_d$ |
| ZeRO-2 | 再分梯度 | $2\Psi+14\Psi/N_d$ |
| ZeRO-3 | 再分参数 | $16\Psi/N_d$ |

三个阶段累计启用；若优化器或精度改变，需要重新计算系数。以上未包含激活、通信 buffer 和当前层物化的瞬时副本。

### 一次更新的数据流

```text
各 DP rank 在自己的样本上做前后向
  → 把某个参数片段的梯度归约到它的 owner
  → owner 用该片的优化器状态更新参数
  → 其他 rank 在需要时获取更新后的参数
```

stage 1 / 2 可以保持完整参数供下一步使用；stage 3 平时只保留 owner 分片，逐计算单元获取参数，用完释放额外副本。即使长期分片很小，当前工作集仍要能放下。

### 通信账本

大消息 DP all-reduce 可以理解成 reduce-scatter 梯度 + all-gather 梯度。ZeRO 将后半段替换为 all-gather 更新后的参数，基本流程的总量仍约 $2\Psi$ 个元素。stage 3 的前向、反向各收集参数一次，再 reduce-scatter 梯度，总量约 $3\Psi$。

这里沿用论文近似计数，省略 $(N_d-1)/N_d$，并统一发送量口径；按发送加接收统计会同时放大所有项。通信量不是时间保证，消息次数、拓扑、预取和重算会改变真实成本。

### 同步语义与组合

保持相同 batch、损失归一化、全局梯度和优化器规则，可以保持同步更新语义；全局 gradient clipping 或 overflow 判断仍需跨 owner 聚合信息。浮点归约顺序可能导致数值差异。

ZeRO 切状态归属，TP 切算子，PP 切层和执行流水。组合时，ZeRO 在相应的 DP 组内分片，不是自动在全世界所有 GPU 之间分片。

## 关键性质与代价

| 收益 | 代价与条件 |
| --- | --- |
| 去掉训练状态的冗余副本 | 需要 owner 映射、归约、参数获取和一致性管理 |
| stage 1 / 2 可大幅降低状态显存 | 未分片的完整参数 / 梯度构成容量下限 |
| stage 3 持久状态随 DP 度缩小 | 当前层物化、通信桶与激活仍消耗显存 |
| 更多显存可支持更高效 batch | 改变全局 batch 后需重新验证优化效果 |
| 与其他并行维度组合 | 通信组、rank 拓扑和 checkpoint 恢复更复杂 |

## 常见误解

- **误解**：ZeRO 是把 Adam 换成更省内存的优化器 → **实际**：基本机制保持优化器状态与更新规则，改变存储和执行归属。
- **误解**：ZeRO-3 不需要完整层参数 → **实际**：通常仍要临时物化当前计算单元所需数据。
- **误解**：显存缩小 $N_d$ 倍就等于总训练峰值缩小 $N_d$ 倍 → **实际**：该比例针对理想长期模型状态。
- **误解**：通信量相同就一定一样快 → **实际**：通信依赖位置、粒度和重叠仍不同。
- **误解**：16 bytes/参数适用于所有训练 → **实际**：它依赖指定混合精度 Adam 账本。
- **误解**：原始 ZeRO 论文已训练出万亿模型 → **实际**：所读版本的万亿结果是容量分析，实测实现为 ZeRO-100B。

## 出现在哪些论文里

- [GPipe / PipeDream](../topics/llm-training/2019-gpipe-pipedream/README.md) —— 本仓库的组合分析：持久状态分片与流水在飞激活、权重版本共同决定峰值；两篇流水原文没有进行 ZeRO 组合实验。

- [ZeRO](../topics/llm-training/2020-zero/README.md) —— 提出三阶段模型状态分片与 ZeRO-R。在 7.5B、64 路 DP、上述精度账本的 Figure 1 中，模型状态从 120 GB/设备理想降到 31.4、16.6、1.875 GB；这是容量分析，不是完整训练显存实测。

## 延伸阅读

- [张量并行](tensor-parallelism.md) · [激活检查点](activation-checkpointing.md)：分别处理算子规模与激活保存。
- [ZeRO-Offload](https://arxiv.org/abs/2101.06840) · [ZeRO-Infinity](https://arxiv.org/abs/2104.07857)：将存储与执行进一步扩展到 CPU / NVMe 等层次的后续工作。
