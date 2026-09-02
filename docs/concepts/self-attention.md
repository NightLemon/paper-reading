---
concept: "Self-Attention"
aliases: ["自注意力", "intra-attention", "Scaled Dot-Product Attention"]
tags: ["transformer", "attention", "sequence-modeling"]
papers: ["topics/foundations/2017-attention-is-all-you-need"]
---

# Self-Attention

> **一句话定义**：序列中的每个位置都用自己生成的 query 去和**同一序列所有位置**的 key 做匹配，按匹配度加权求和这些位置的 value，得到该位置的新表示。

## 为什么需要它

序列建模要解决的核心问题是：位置 $i$ 的表示应该受哪些其他位置影响，影响多大。

- **循环层**沿位置逐步传递隐状态。位置 $i$ 与位置 $j$ 之间的信息要穿过 $|i-j|$ 步，路径长且必须顺序执行。
- **卷积层**用固定窗口聚合。单层只能连接距离 $< k$ 的位置对，覆盖全序列需要堆叠 $O(n/k)$ 层（连续核）或 $O(\log_k n)$ 层（膨胀卷积）。

self-attention 让任意两个位置**一步直连**：路径长度 $O(1)$，且所有位置的计算彼此独立，可以在一次矩阵乘中并行完成。

## 它是怎么工作的

对输入序列的表示矩阵 $X \in \mathbb{R}^{n \times d}$，用三个投影得到 query、key、value：

$$Q = XW^Q, \quad K = XW^K, \quad V = XW^V$$

然后

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

三步：

1. $QK^T$ 得到 $n \times n$ 的**相似度矩阵**，第 $(i,j)$ 项是位置 $i$ 的 query 与位置 $j$ 的 key 的点积。
2. 除以 $\sqrt{d_k}$ 后按行 softmax，得到每个位置对全部位置的**注意力权重分布**（每行和为 1）。
3. 用权重对 $V$ 加权求和。

「self」体现在 $Q$、$K$、$V$ 都来自同一个序列。当 $Q$ 来自序列 A、$K$/$V$ 来自序列 B 时，这就变成 **cross-attention**（如 encoder-decoder attention）。

### 缩放因子 $1/\sqrt{d_k}$

设 $q$、$k$ 各分量独立、均值 0、方差 1，则 $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ 的方差为 $d_k$，标准差 $\sqrt{d_k}$。$d_k$ 越大，softmax 的输入差异越大，输出越接近 one-hot，梯度越小。除以 $\sqrt{d_k}$ 使 softmax 输入的分布与 $d_k$ 无关。

### 因果 mask

自回归生成要求位置 $i$ 不能看到 $>i$ 的位置。实现方式是在 **softmax 之前**把非法位置的相似度置为 $-\infty$：$e^{-\infty} = 0$ 使这些位置权重严格为 0，且不进入归一化分母。在 softmax 之后置 0 是错的——剩余权重的和不再为 1。

## 关键性质与代价

| 收益 | 代价 |
| --- | --- |
| 任意两位置路径长度 $O(1)$，长程依赖易学 | 每层复杂度 $O(n^2 \cdot d)$，序列越长越贵 |
| 最小顺序操作数 $O(1)$，样本内完全并行 | 需要显式注入位置信息（本身置换等变） |
| 可直接复用高度优化的矩阵乘 kernel | 训练时需保存 $n \times n$ 注意力矩阵用于反向 |
| 注意力权重可视化，提供一定可解释性 | 加权平均降低有效分辨率，需 multi-head 补偿 |

**成立条件**：self-attention 每层比循环层便宜的前提是 $n < d$。$n$ 增长到与 $d$ 同量级以上时，$O(n^2 d)$ 反超。这个前提是后续所有稀疏 / 滑窗 / 线性注意力工作的出发点。

## 常见误解

- **误解**：self-attention 的复杂度是 $O(n^2)$ → **实际**：是 $O(n^2 \cdot d)$。忽略 $d$ 会让人误判它与 $O(n \cdot d^2)$ 的循环层的交叉点位置。
- **误解**：路径长度 $O(1)$ 意味着长程依赖一定学得好 → **实际**：路径长度只衡量信息**可达性**，不衡量优化难度。深层堆叠仍依赖残差与归一化才能训练。
- **误解**：注意力权重高 = 该位置更重要 → **实际**：注意力权重是模型内部的中间量，与「特征重要性」没有严格对应关系。
- **误解**：FlashAttention 把复杂度降到了线性 → **实际**：它把**显存**降到线性，计算量仍是 $O(n^2 d)$；它是精确算法，不改变数学结果。

## 与推理优化的关系

自回归解码时，位置 $j$ 的 $K_j$、$V_j$ 只依赖位置 $\le j$ 的输入。新生成的 token 不会改变已有位置的 $K$、$V$，因此可以缓存复用——这是 **KV Cache** 的全部依据，也是 prefix caching、prefix 亲和路由等一系列 serving 优化的前提。

## 出现在哪些论文里

- [Attention Is All You Need](../topics/foundations/2017-attention-is-all-you-need/README.md) —— 提出 scaled dot-product 形式，并首次把 self-attention 作为唯一的序列建模原语（不再依附于 RNN）。

## 延伸阅读

- [KV Cache](kv-cache.md)：上述前缀不变性在推理侧的直接产物。
- FlashAttention：IO-aware 的精确注意力，解决 $n \times n$ 中间矩阵的显存问题。
- 稀疏 / 滑窗 / 线性注意力：放宽「全连接」以降低 $O(n^2)$ 项。
