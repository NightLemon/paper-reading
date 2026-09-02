---
title: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
authors: ["Tri Dao", "Daniel Y. Fu", "Stefano Ermon", "Atri Rudra", "Christopher Ré"]
affiliation: "Stanford University / University at Buffalo"
venue: "NeurIPS 2022"
year: 2022
arxiv: "2205.14135"
url: "https://arxiv.org/abs/2205.14135"
topic: "foundations"
tags: ["attention", "io-aware", "gpu-kernel", "memory", "tiling"]
concepts: ["io-aware-kernel-design", "self-attention"]
status: "to-read"
rating:
read_date:
---

# FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

!!! note "本页为占位骨架"
    这篇尚未精读。工作已拆分为 GitHub Issue **「精读 FlashAttention（NeurIPS'22）」**，由协作 agent 接单完成。
    接单者请阅读该 issue 中的文件归属约定后再动手：本文件与 `docs/concepts/io-aware-kernel-design.md` 由该工单独占，其余共享索引文件一律不要修改。

> **一句话结论**：<待精读>

---

## 1. 元信息

| 项目 | 内容 |
| --- | --- |
| 作者 / 机构 | Tri Dao 等 · Stanford / University at Buffalo |
| 发表 | NeurIPS 2022（arXiv:2205.14135） |
| 原文 | [arXiv](https://arxiv.org/abs/2205.14135) |
| 关键词 | IO-awareness · tiling · recomputation · SRAM/HBM |
| 前置知识 | [Self-Attention](../../../concepts/self-attention.md)、GPU 存储层次（SRAM / HBM）、softmax 的在线归约 |

---

## 2. 摘要速览（5 分钟版）

### 2.1 要解决的问题
<待精读>

### 2.2 核心方法
<待精读>

### 2.3 主要结果
<待精读>

### 2.4 我的评价
<待精读>

---

## 3. 细读

<待精读：章节标题需与原文实际章节一一对应>

---

## 4. 关键问题解析

<待精读>

---

## 5. 可迁移的知识点

- [IO-aware 算法设计](../../../concepts/io-aware-kernel-design.md) —— 本文是这一范式最有影响力的实例。
- [Self-Attention](../../../concepts/self-attention.md) —— 本文不改变其数学结果，只改变访存方式。

---

## 6. 批判与开放问题

<待精读>
