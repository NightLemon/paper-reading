---
concept: "Multi-Head Attention"
aliases: ["多头注意力", "MHA"]
tags: ["transformer", "attention"]
papers: ["topics/foundations/2017-attention-is-all-you-need"]
---

# Multi-Head Attention

> **一句话定义**：把表示空间切成 $h$ 份，在每一份上独立跑一次注意力，再把 $h$ 个结果拼接后线性混合——用同样的计算预算换来 $h$ 组彼此独立的注意力分布。

## 为什么需要它

单个注意力头对每个 query 位置只产生**一个** softmax 分布，因此输出只能是 values 的**一个凸组合**。当一个位置同时需要来自多个不同位置的、不同性质的信息时（例如既要指向句法上的支配词，又要指向语义上的论元），单个分布必须折中，两路信号都被削弱。

这是「加权平均降低有效分辨率」的具体含义：平均本身是一种信息损失。

## 它是怎么工作的

$$\mathrm{MultiHead}(Q,K,V) = \mathrm{Concat}(\mathrm{head}_1, \dots, \mathrm{head}_h)\,W^O$$

$$\mathrm{head}_i = \mathrm{Attention}(QW_i^Q,\; KW_i^K,\; VW_i^V)$$

- 每个头有自己的三组投影矩阵，把 $d_{\text{model}}$ 维投影到 $d_k$（或 $d_v$）维。
- $h$ 个头**并行**计算，各自产生一个注意力分布与一个输出。
- 所有头的输出沿特征维拼接（长度 $h \cdot d_v$），再经 $W^O$ 投影回 $d_{\text{model}}$。

**关键约定**：取 $d_k = d_v = d_{\text{model}} / h$。这使得 $h$ 个头的**总计算量与全维单头相当**——multi-head 不是加算力，是在同等预算下重新划分表示空间。

## 关键性质与代价

| 收益 | 代价 |
| --- | --- |
| 同一层内可同时保留多组指向不同位置的信息 | 每个头的 $d_k$ 随 $h$ 增大而减小 |
| 总计算量与单头持平 | $d_k$ 过小时兼容性判定退化，质量反降 |
| 推理时各头可并行，对 GPU 友好 | KV Cache 大小正比于 $h \cdot d_k$（这是 MQA / GQA 的优化对象） |

存在一个**最优头数**：头太少无法覆盖多种关系，头太多则每头维度不足。这两端在实验上都能观测到质量下降。

## 常见误解

- **误解**：多头 = 多花几倍算力 → **实际**：$d_k = d_{\text{model}}/h$ 的约定使总成本与单头基本持平。
- **误解**：每个头都学到了人类可解释的独立语言学功能 → **实际**：这是定性观察。定量研究发现相当比例的头可以被剪掉而质量几乎不降，说明头之间存在冗余。
- **误解**：缩放因子应该随 $h$ 变化 → **实际**：缩放因子只依赖 $d_k$。在 $d_k = d_{\text{model}}/h$ 的约定下，改变 $h$ 会**间接**改变缩放，不需要额外引入 $h$。

## 与推理优化的关系

KV Cache 的体积正比于 $\text{layers} \times h \times d_k \times \text{seq\_len} \times 2$。头数直接决定了缓存占用，因此后续出现了 **MQA**（所有 query 头共享一组 K/V）与 **GQA**（分组共享）来压缩这一项。这类变体改变的是 K/V 的头数而非 query 的头数，属于对本概念的直接改造。

## 出现在哪些论文里

- [Attention Is All You Need](../topics/foundations/2017-attention-is-all-you-need/README.md) —— 提出 multi-head 结构；给出两处动机（抵消平均导致的分辨率损失、关注不同表示子空间）；消融显示单头比最佳设置差 0.9 BLEU，头数过多同样掉点。

## 延伸阅读

- MQA / GQA：压缩 K/V 头数以缩小 KV Cache。
- 注意力头剪枝相关研究：检验「不同头功能互补」这一论断的定量证据。
