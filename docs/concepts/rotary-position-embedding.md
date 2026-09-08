---
concept: "Rotary Position Embedding"
aliases: ["RoPE", "旋转位置编码"]
tags: ["transformer", "position", "attention", "long-context"]
papers: ["topics/foundations/2021-roformer"]
---

# Rotary Position Embedding

> **一句话定义**：按 token 的位置旋转 query 与 key 的二维分量对，使内积中的位置项通过两端旋转角差表达相对距离。

## 为什么需要它

attention 需要位置信息来区分顺序。输入 embedding 加位置向量的办法可以提供绝对位置，但若希望 QK 内积中的显式位置只依赖相对距离，就需要额外约束其组合形式。

RoPE 对每个位置分别变换 Q/K，再利用旋转的合成性质得到相对关系。这样每个 token 的位置变换可独立计算，无需为每一对 token 建立独立的可学习相对位置向量。

## 它是怎么工作的

### 二维旋转与相对距离

定义：

$$R(\phi)=\begin{pmatrix}\cos\phi&-\sin\phi\\\sin\phi&\cos\phi\end{pmatrix}.$$

位置 $m$ 的 query 变成 $R(m\theta)q_m$，位置 $n$ 的 key 变成 $R(n\theta)k_n$，于是：

$$\tilde q_m^T\tilde k_n=q_m^TR((n-m)\theta)k_n.$$

范数保持，夹角按相对位置变化。内积仍然依赖内容向量；相对位置是其中的显式位置项。

### 多维实现

把偶数单头维度 $d_h$ 拆成 $d_h/2$ 对，典型频率为 $\theta_j=10000^{-2j/d_h}$，$j$ 从 0 开始。每对 $(a,b)$ 独立计算：

$$a'=a\cos(m\theta_j)-b\sin(m\theta_j),\quad
b'=a\sin(m\theta_j)+b\cos(m\theta_j).$$

整体为块对角旋转，$R_m^TR_n=R_{n-m}$。无需构造稠密矩阵，额外运算随向量维度线性增长。旋转通常施加在每层投影后的 Q/K，标准形式不需要旋转 V。

维度配对方式、频率、位置编号必须与模型权重和实现约定一致。改变其中任何一项都不能默认对模型无影响。

### 与 KV Cache 的一致性

常见做法缓存已旋转的 key，后续 query 按自己的绝对位置旋转后读取。chunked prefill 应延续整条请求的位置编号；把每块重置为 0 会改变跨块距离。

上述是由公式得到的执行要求。文本移动到不同上下文时，深层表示也可能变化，所以“旋转可补相位差”不足以证明全部 KV 可任意搬到新位置复用。

## 关键性质与代价

| 性质 / 收益 | 条件 / 代价 |
| --- | --- |
| 显式位置项以相对距离出现 | 内容和此前上下文仍影响 Q/K |
| 正交旋转保持单个向量范数 | 不保证 attention 权重或训练过程不变 |
| 原始固定频率形式无需可学习位置表 | 仍需 sin/cos 计算或缓存，并保持配置一致 |
| 任意整数位置都可计算 | 不保证超训练长度的质量 |
| 可构造保留结合律的 linear attention 变体 | 原文该变体分母不旋转，权重不再严格概率归一化 |

几何频率叠加带来多尺度位置变化，RoFormer 用 Abel 变换和数值图讨论长期衰减。这个分析不意味着任意 token 对的内积单调下降：二维 $q=k=(1,0)$ 的内积就是周期性的 $\cos((n-m)\theta)$。softmax 还会受其他 key 影响。

## 常见误解

- **误解**：RoPE 只使用相对位置，所以不需要绝对编号 → **实际**：每端按绝对编号旋转，内积时角差成为相对距离。
- **误解**：先在输入 embedding 上旋转一次即可 → **实际**：任意投影通常不与旋转交换；原构造作用于每层投影后的 Q/K。
- **误解**：距离越远，attention 必然越小 → **实际**：理论分析有条件，实际分数包含内容和振荡项。
- **误解**：位置函数能算到 100K 就代表模型能用好 100K → **实际**：可计算性与训练分布外泛化要分别验证。
- **误解**：RoPE 让精确 attention 变成线性复杂度 → **实际**：标准 QK / AV 的二次复杂度仍在；linear attention 是另一种注意力形式。
- **误解**：位置插值和后续缩放方法的效果都是原始 RoFormer 的实验证据 → **实际**：它们改变位置映射或训练过程，须单独注明来源。

## 出现在哪些论文里

- [RoFormer](../topics/foundations/2021-roformer/README.md) —— 推导二维旋转、块对角推广、高效实现与长期衰减分析；在翻译、MLM、GLUE、Performer 和中文案件匹配中评测。中文模型预训练包含长度 1536，Table 5 的 1024 长度结果不能当作超训练长度外推证明。

## 延伸阅读

- [Positional Encoding](positional-encoding.md) · [Self-Attention](self-attention.md) · [KV Cache](kv-cache.md)。
- [Chunked Prefill](chunked-prefill.md)：执行被切块时保持位置和上下文连续。
- [Positional Interpolation](https://arxiv.org/abs/2306.15595)：区分位置变换、适应训练与长上下文实验。
