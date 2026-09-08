---
concept: "Grouped-Query Attention"
aliases: ["GQA", "分组查询注意力", "MQA", "Multi-Query Attention"]
tags: ["transformer", "attention", "kv-cache", "inference"]
papers: ["topics/foundations/2023-gqa"]
---

# Grouped-Query Attention

> **一句话定义**：保留多个 query 头，让每组 query 头共享一个 key 头和一个 value 头，以减少缓存和 K/V 投影的成本。

## 为什么需要它

自回归解码每步都要读取历史 K/V。标准 MHA 为每个 query 头分别存 K/V，导致 KV 随头数与上下文长度增长。MQA 让所有 query 共享一组 KV，压缩最激进；GQA 允许选择中间组数，在缓存、表示能力和训练成本之间取舍。

这里 MQA 是 GQA 的单组特例，并非所有组数下都与 GQA 同义。

## 它是怎么工作的

设 query 头数为 $H$，KV 组数为 $G$，每头维度为 $d_h$。常见等分组实现要求 $H$ 能被 $G$ 整除，第 $i$ 个 query 头所属组为 $g(i)$：

$$O_i=\operatorname{softmax}\!\left(\frac{Q_iK_{g(i)}^T}{\sqrt{d_h}}+M\right)V_{g(i)}.$$

$M$ 表示适用的 attention mask。各 query 头仍产生自己的输出，拼接后经输出投影。共享 K/V 不要求共享 Q，也不要求同组的注意力权重相同。

| 结构 | Query 头数 | K/V 头数 | 关系 |
| --- | --- | --- | --- |
| MHA | $H$ | $H$ | 每头独立 K/V |
| GQA | $H$ | $G$ | 每组共享 K/V |
| MQA | $H$ | 1 | 全部 query 共享 K/V |

### 缓存量与计算量

对 $L$ 层、缓存长度 $n$、每元素 $s$ bytes 的单请求 decoder self-attention：

$$\mathrm{KVBytes}=2Lnd_hGs.$$

相对 MHA 缩小到 $G/H$。query 头数仍为 $H$，因此 QK / AV 的注意力工作量没有随 $G/H$ 等比例缩小；K/V 投影本身则减少。实际缓存还要计入 TP 副本、布局与分配器开销。

encoder-decoder 模型的 cross-attention KV 长度由 encoder 输入决定，应单独计算。不能用 decoder 输出长度代替所有 KV 的长度。

### 从已有 MHA checkpoint 转换

一种已验证的路线是在每组内分别平均 K/V 投影权重，然后继续预训练以适应结构变化：

$$\bar W_g^K=\frac1{|\mathcal G_g|}\sum_{i\in\mathcal G_g}W_i^K,\quad
\bar W_g^V=\frac1{|\mathcal G_g|}\sum_{i\in\mathcal G_g}W_i^V.$$

平均保留部分既有权重信息，但 attention 包含非线性 softmax，转换前后输出通常不同。继续预训练与下游 fine-tuning 是后续步骤；GQA 也可以从头训练，但那是不同的成本和实验对照。

### 与张量并行的关系

如果一个逻辑 KV 头为了按 query 头分片而在多张设备复制，物理缓存量会高于逻辑公式。多个 KV 组可以与分片对齐，但当组数小于分片数或分组不对齐时仍可能需要复制或通信。最佳 $G$ 因而还取决于部署的 TP 度与 kernel 布局。

## 关键性质与代价

| 收益 | 代价与条件 |
| --- | --- |
| 减少逻辑 KV 容量与每步读取需求 | 独立 K/V 表示减少，质量需要重新评估 |
| 可在 MHA / MQA 之间调节共享程度 | 组数需结合模型、任务、TP 和硬件选择 |
| 已有 MHA 可通过转换继续训练 | 需要数据、训练计算和模型验证 |
| 可以叠加分页、量化与高效 attention kernel | 叠加后瓶颈会改变，各自加速比不能相乘 |

## 常见误解

- **误解**：GQA 把 query 头也减少了 → **实际**：标准 GQA 保留 query 头，减少 K/V 头。
- **误解**：同组 query 的 attention 权重相同 → **实际**：Q 不同，因此对同一 K 的分数仍不同。
- **误解**：KV 缩小 4 倍，推理必然快 4 倍 → **实际**：总时间还包含权重、算术、通信与其他开销。
- **误解**：mean-pooling 是无损 checkpoint 转换 → **实际**：它改变模型函数，需要适应训练和质量验证。
- **误解**：GQA-8 表示每组 8 个 query 头 → **实际**：本文命名里的 8 是 KV 组数，每组 query 数为 $H/8$。
- **误解**：组数 8 是通用最优值 → **实际**：这是 GQA 论文 T5 实验选择的折中，迁移后需要重新评估。

## 出现在哪些论文里

- [GQA](../topics/foundations/2023-gqa/README.md) —— 提出组内 K/V mean-pooling 加 uptraining 的迁移路线。在 T5.1.1、8 个 TPUv4、greedy decoding、每模型优化 batch 与并行、5% uptraining 的 Table 1 中，GQA-8-XXL 的每样本每 chip 时间为 0.28 s，MHA-XXL 为 1.51 s；七项任务平均分分别为 47.1 / 47.2。该结果包含 decoder self-attention 与 cross-attention，不能直接当作 decoder-only GPU 延迟。

## 延伸阅读

- [Multi-Head Attention](multi-head-attention.md) · [KV Cache](kv-cache.md) · [张量并行](tensor-parallelism.md)。
- [MQA 原论文](https://arxiv.org/abs/1911.02150)：理解共享单组 KV 的出发点。
- [Sarathi-Serve](../topics/llm-serving/2024-sarathi-serve/README.md)：缓存允许更多请求之后，仍需处理大批次与尾延迟的取舍。
