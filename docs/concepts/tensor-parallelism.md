---
concept: "张量并行"
aliases: ["Tensor Parallelism", "TP", "模型并行", "Megatron-style parallelism"]
tags: ["distributed-training", "parallelism", "llm-training"]
papers: ["topics/llm-training/2019-megatron-lm"]
---

# 张量并行

!!! note "本页为占位骨架"
    待 GitHub Issue **「精读 Megatron-LM（张量并行）」** 完成后由接单 agent 补全。

> **一句话定义**：<待补全>

## 为什么需要它

<待补全：单卡装不下模型时，把单个算子的权重矩阵按行/列切到多张卡上。>

## 它是怎么工作的

<待补全：MLP 的列切分 + 行切分组合、注意力按 head 维度切分、每层两次 all-reduce。>

## 关键性质与代价

| 收益 | 代价 |
| --- | --- |
| | |

## 与其他并行维度的关系

<待补全：数据并行 / 流水并行 / 序列并行的分工。>

## 常见误解

- <待补全>

## 出现在哪些论文里

- [Megatron-LM](../topics/llm-training/2019-megatron-lm/README.md) —— <待补全>

## 延伸阅读

- [Multi-Head Attention](multi-head-attention.md) · [分页式 KV 管理](paged-kv-memory.md)
