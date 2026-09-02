---
concept: "IO-aware 算法设计"
aliases: ["IO-Awareness", "访存感知", "tiling", "kernel fusion"]
tags: ["gpu", "performance", "algorithm-design"]
papers: ["topics/foundations/2022-flashattention"]
---

# IO-aware 算法设计

!!! note "本页为占位骨架"
    待 GitHub Issue **「精读 FlashAttention（NeurIPS'22）」** 完成后由接单 agent 补全。

> **一句话定义**：<待补全>

## 为什么需要它

<待补全：渐进复杂度只计浮点运算次数，忽略了存储层次之间的数据搬运；在现代 GPU 上后者常常才是瓶颈。>

## 它是怎么工作的

<待补全：分块（tiling）、算子融合、重计算换访存。>

## 关键性质与代价

| 收益 | 代价 |
| --- | --- |
| | |

## 常见误解

- <待补全>

## 出现在哪些论文里

- [FlashAttention](../topics/foundations/2022-flashattention/README.md) —— <待补全>

## 延伸阅读

- [Self-Attention](self-attention.md) · [分页式 KV 管理](paged-kv-memory.md)
