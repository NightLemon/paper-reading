---
title: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
authors: ["Tri Dao", "Daniel Y. Fu", "Stefano Ermon", "Atri Rudra", "Christopher Ré"]
affiliation: "Stanford University / University at Buffalo, SUNY"
venue: "NeurIPS 2022"
year: 2022
url: "https://arxiv.org/abs/2205.14135"
topic: "foundations"
tags: ["attention", "io-awareness", "tiling", "kernel-fusion", "recomputation", "neurips"]
concepts: ["io-aware-kernel-design", "self-attention"]
status: "done"
rating: 5
read_date: "2026-09-02"
---

# FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

> **一句话结论**：注意力慢不是因为 FLOPs 多，而是因为把 $N \times N$ 的中间矩阵在 HBM 上来回搬。用分块 online softmax 让它**根本不被物化**、用重计算替代保存，HBM 访问量从 $\Theta(Nd + N^2)$ 降到 $\Theta(N^2 d^2 M^{-1})$，**结果逐位精确**——GPT-2 medium 上前向+反向从 41.7 ms 降到 7.3 ms，显存从 $O(N^2)$ 降到 $O(N)$。

---

## 1. 元信息

| 项目 | 内容 |
| --- | --- |
| 作者 / 机构 | Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré · Stanford / SUNY Buffalo |
| 发表 | NeurIPS 2022 |
| 原文 | [arXiv:2205.14135](https://arxiv.org/abs/2205.14135) · 本地 [paper.pdf](paper.pdf) / [paper.html](paper.html)（ar5iv 版） |
| 代码 | <https://github.com/HazyResearch/flash-attention>（开源，后成为事实标准） |
| 关键词 | IO-awareness · tiling · online softmax · recomputation · kernel fusion · block-sparse |
| 前置知识 | [Self-Attention](../../../concepts/self-attention.md)、[IO-aware 算法设计](../../../concepts/io-aware-kernel-design.md)、GPU 存储层次 |

---

## 2. 摘要速览（5 分钟版）

### 2.1 要解决的问题

Transformer 想要长上下文，卡在 self-attention 对序列长度 $N$ 的**二次方时间与显存**上。此前大量近似注意力方法（稀疏、低秩及其组合）把计算复杂度降到线性或近线性，但论文指出一个尴尬的事实：

> 它们中的许多**在墙钟时间上并不比标准注意力快**，因此没有被广泛采用。主要原因是它们盯着减少 FLOPs（这与墙钟速度未必相关），而忽略了访存（IO）开销。

这句话是全文的靶子。论文主张缺失的原则是让注意力算法 **IO-aware**——认真核算在快慢不同的存储层级之间的读写。而现代 GPU 上算力增长早已超过访存带宽，Transformer 里**大多数算子都是访存受限的**。

本仓库的 [Transformer 笔记](../2017-attention-is-all-you-need/README.md) §3.4 与 Q5 已经指出这个缺口：原论文 Table 1 用「每层复杂度 / 顺序操作数 / 最长路径」三项比较架构，**唯独没有显存这一维**，而 $N \times N$ 注意力矩阵在训练时必须保留用于反向传播，这才是长序列的第一约束。FlashAttention 补的正是这一维——并且注意，**它不改变 Table 1 里的任何一个渐进复杂度**，这恰恰说明那张表衡量的维度不完整。

### 2.2 核心方法

目标很明确：**不把注意力矩阵读写到 HBM**。这需要解决两个技术难点：

1. 不接触完整输入的情况下完成 **softmax 归约**；
2. 反向传播不保存那个巨大的中间注意力矩阵。

对应两个手段（论文强调二者都是**已有技术**，贡献在于组合与分析）：

- **Tiling（分块）**：把 $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ 切块，从 HBM 载入 SRAM，逐块计算并**增量地完成 softmax 归约**。关键在于每块结果按正确的归一化因子缩放后累加，最终得到精确解。
- **Recomputation（重计算）**：前向只保存输出 $\mathbf{O}$ 与 softmax 的归一化统计量 $(m, \ell)$；反向时从 SRAM 里的 $\mathbf{Q},\mathbf{K},\mathbf{V}$ 块**重新算出** $\mathbf{S}$ 与 $\mathbf{P}$。这是一种选择性的 gradient checkpointing，但与常规 checkpointing **用速度换显存**不同——这里即使 FLOPs 增加了，反向反而**更快**，因为省下的 HBM 访问远超多出的计算。

实现上用 CUDA 手写，把所有注意力操作**融进单个 kernel**。

### 2.3 主要结果

**IO 复杂度**（Theorem 2，$d \le M \le Nd$）：标准注意力 $\Theta(Nd + N^2)$，FlashAttention $\Theta(N^2 d^2 M^{-1})$。典型 $d$ 为 64–128、$M$ 约 100KB 时 $d^2 \ll M$，因而少很多倍。论文还给了下界（Proposition 3）：**不存在**对所有 $M \in [d, Nd]$ 都能做到 $o(N^2 d^2 M^{-1})$ 的精确注意力算法。

**最有说服力的一组数字**（Figure 2 左，GPT-2 medium：序列长 1024、head dim 64、16 头、batch 64，A100，前向+反向）：

| | 标准注意力 | FlashAttention |
| --- | --- | --- |
| GFLOPs | 66.6 | **75.2**（更多！） |
| HBM 读写 (GB) | 40.3 | **4.4** |
| 运行时间 (ms) | 41.7 | **7.3** |

**FLOPs 多了 13%，时间少了 82%**。这一行数据本身就是全文论点的证明。

**端到端**（均为 8×A100）：

| 任务 | 结果 |
| --- | --- |
| BERT-large（seq 512，Wikipedia，到 72.0% MLM 精度，10 次平均） | 17.4±1.4 min vs Nvidia MLPerf 1.1 的 20.0±1.5 min，**快 15%** |
| GPT-2 small（OpenWebText） | 2.7 天 vs HuggingFace 9.5 天（**3.5×**）、Megatron-LM 4.7 天（**1.7×**），ppl 均为 18.2 |
| GPT-2 medium | 6.9 天 vs HF 21.0 天（**3.0×**）、Megatron 11.5 天（**1.8×**），ppl 14.3 |
| LRA（seq 1K–4K） | **2.4×**，平均精度 59.8 vs 标准 59.3；block-sparse 2.8×、59.6 |
| GPT-2 small 长上下文 | 4K 上下文仍比 Megatron 的 1K **快 30%**，且 ppl 好 0.7（17.5 vs 18.2） |
| 长文档分类 | MIMIC-III：seq 16K 比 512 高 **4.3** 点（57.1 vs 52.8）；ECtHR：8K 比 512 高 **8.5** 点（80.7 vs 72.2） |
| Path-X（seq 16K） | **61.4**——首个超过随机水平的 Transformer（此前所有模型 OOM 或随机） |
| Path-256（seq 64K） | block-sparse 版 **63.1**，首个做到的序列模型 |

**注意力基准**（单张 A100 40GB，带 dropout 与 padding mask）：常见序列长度（≤2K）下比 PyTorch 实现快至多 **3×**；显存比精确注意力基线省至多 **20×** 且随 $N$ **线性**增长。但有一处交叉点值得记住：**序列长 512–1024 之间，近似注意力方法开始反超 FlashAttention 的速度**。

### 2.4 我的评价

这篇论文的分量不在于某个技巧，而在于它**换掉了衡量算法好坏的尺子**。在它之前，「注意力太贵」的默认解法是降低 FLOPs——于是有了几十种近似注意力。FlashAttention 指出这些方法大多没能真正变快，因为瓶颈根本不在 FLOPs，然后用一张三行的表（FLOPs 更多、时间更少）把这件事钉死。

它还有一个容易被低估的性质：**结果是精确的**。近似注意力要求使用者承担精度风险、重新调参、重新验证下游效果；FlashAttention 是一个可以无条件替换进去的 drop-in，模型定义一个字都不用改（论文明确说 ppl 与基线一致）。**「无需权衡」是它能在一年内成为行业默认的根本原因**——这一点比 3× 加速本身更重要。

评分 ⭐⭐⭐⭐⭐。今天所有主流推理与训练框架都内置了它或它的后继版本，online softmax + tiling 已经成为写注意力 kernel 的默认范式。

---

## 3. 细读

> 章节编号与原文一致（正文 5 节 + 附录 A–E）。

### 3.1 Introduction

论证链条极其干净，值得逐步拆开：

1. Transformer 变大变深，但**加长上下文仍然困难**，因为 self-attention 的时间与显存都是序列长度的二次方。
2. 大量近似注意力方法（稀疏、低秩及组合）把计算降到线性/近线性，**但很多没有墙钟加速，也没有被广泛采用**。
3. 原因诊断：它们**盯着 FLOPs（与墙钟速度未必相关），忽略访存开销**。
4. 提出缺失的原则：**IO-awareness**——认真核算快慢存储层级之间的读写。依据是现代 GPU 上算力增速已超过访存，Transformer 里多数算子受访存约束。
5. 佐证：IO-aware 算法在数据库 join、图像处理、数值线性代数等访存受限场景里早已是关键。
6. 障碍：PyTorch / TensorFlow 这类 Python 接口**不允许细粒度控制访存**——所以只能手写 CUDA。

论文顺带给了主结果的预告：GPT-2 上注意力计算加速 **7.6×**（Figure 1 右），因为不再向 HBM 读写那个 $N \times N$ 矩阵。

> **批注**：第 3 步是全文的枢纽，也是最值得学的一句话——**「FLOP reduction may not correlate with wall-clock speed」**。这句话适用范围远远超出注意力：任何时候你看到「我们把复杂度从 $O(n^2)$ 降到了 $O(n\log n)$」而没有墙钟数据，都应该问一句常数因子和访存模式怎么样。论文没有停在口头断言，而是给了一张 FLOPs 更多但更快的表，**用自己的方法作为反例证明了旧尺子是错的**。这是很高明的论证结构。

### 3.2 Background

#### 2.1 Hardware Performance

**GPU 存储层次**。A100 的具体数字（这些数字后面每一处推导都要用）：

| 层级 | 容量 | 带宽 |
| --- | --- | --- |
| HBM | 40–80 GB | 1.5–2.0 TB/s |
| 片上 SRAM | 192 KB × 108 个 SM | 约 19 TB/s（估计） |

SRAM 比 HBM **快一个数量级，小很多个数量级**。而算力相对访存越来越快，所以算子越来越受 HBM 访问约束，用好 SRAM 也就越来越重要。

**执行模型**：每个 kernel 从 HBM 载入到寄存器与 SRAM、计算、再写回 HBM。

**性能特征**：按算术强度（每字节访存对应多少次算术操作）分两类：

1. **Compute-bound**：时间由算术操作数决定。典型例子是内维度大的矩阵乘、通道数多的卷积。
2. **Memory-bound**：时间由访存次数决定。**其余大多数算子**都属于这类——逐元素操作（激活、dropout）与归约（sum、softmax、batchnorm、layernorm）。

**Kernel fusion**：加速访存受限算子的最常见手段——同一输入被多个操作使用时，只从 HBM 载入一次。编译器能自动融合许多逐元素操作。**但论文立刻点出它的局限**：在训练场景下，中间值仍然要写回 HBM 以备反向传播，朴素的 kernel fusion 因此效果打折。

> **批注**：最后这句是全文的一处关键伏笔，务必读到。**朴素融合在训练时失效，因为反向传播要用中间值**。FlashAttention 的重计算正是为拆掉这个约束而来——不保存中间值，反向时重算。所以「tiling + 重计算」不是两个并列的优化，而是**重计算解除了 tiling 在训练场景下的枷锁**，缺一不可。

#### 2.2 Standard Attention Implementation

给定 $\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d}$：

$$\mathbf{S}=\mathbf{Q}\mathbf{K}^{\top}\in\mathbb{R}^{N\times N},\quad \mathbf{P}=\mathrm{softmax}(\mathbf{S})\in\mathbb{R}^{N\times N},\quad \mathbf{O}=\mathbf{P}\mathbf{V}\in\mathbb{R}^{N\times d}$$

softmax 按行施加。标准实现**把 $\mathbf{S}$ 与 $\mathbf{P}$ 物化到 HBM**，占 $O(N^2)$ 显存。论文提醒通常 $N \gg d$（GPT-2 里 $N=1024$、$d=64$）。

**Algorithm 0（标准注意力）**：

```text
输入：HBM 中的 Q, K, V ∈ R^{N×d}
1: 按块从 HBM 载入 Q, K，计算 S = QKᵀ，把 S 写回 HBM
2: 从 HBM 读 S，计算 P = softmax(S)，把 P 写回 HBM
3: 按块载入 P 和 V，计算 O = PV，把 O 写回 HBM
4: 返回 O
```

三步里 $N\times N$ 的矩阵被**写了两次、读了两次**。由于其中一些（尤其 softmax）是访存受限的，大量访存直接变成慢墙钟。掩码作用于 $\mathbf{S}$、dropout 作用于 $\mathbf{P}$ 会让问题更严重，所以此前已有不少工作在融合这些逐元素操作（例如把掩码与 softmax 融合）。

> **批注**：把 Algorithm 0 一步步写出来是这篇论文写作上的聪明之处。**问题的全部在这三行里肉眼可见**——两次写、两次读一个 $N\times N$ 矩阵。读者在看到解法之前就已经自己算出账了，后面的方案因此几乎不需要说服。这与 [vLLM 笔记](../../llm-serving/2023-vllm-pagedattention/README.md) §3.1 批注里说的「先把浪费量化清楚，方案自然成立」是同一种手法。

### 3.3 FlashAttention: Algorithm, Analysis, and Extensions

#### 3.1 An Efficient Attention Algorithm With Tiling and Recomputation

**Tiling 的数学基础：softmax 的可分解性**。这是全文最需要讲透的一点。

为数值稳定，向量 $x\in\mathbb{R}^{B}$ 的 softmax 按下式计算：

$$m(x):=\max_i x_i,\quad f(x):=\left[e^{x_1-m(x)}\ \cdots\ e^{x_B-m(x)}\right],\quad \ell(x):=\sum_i f(x)_i,\quad \mathrm{softmax}(x):=\frac{f(x)}{\ell(x)}$$

关键在于：把两段 $x^{(1)},x^{(2)}$ 拼接起来的 softmax，**可以从两段各自的统计量组合出来**：

$$m(x)=\max\big(m(x^{(1)}),m(x^{(2)})\big)$$

$$f(x)=\left[e^{m(x^{(1)})-m(x)}f(x^{(1)})\ \ \ e^{m(x^{(2)})-m(x)}f(x^{(2)})\right]$$

$$\ell(x)=e^{m(x^{(1)})-m(x)}\ell(x^{(1)})+e^{m(x^{(2)})-m(x)}\ell(x^{(2)})$$

于是只要**额外维护两个统计量 $(m, \ell)$**，就能一次一块地算 softmax。论文脚注指出这类聚合叫 **algebraic aggregation**。

**Recomputation**。反向传播通常需要 $\mathbf{S},\mathbf{P}\in\mathbb{R}^{N\times N}$ 来算 $\mathbf{Q},\mathbf{K},\mathbf{V}$ 的梯度。但只要保存了输出 $\mathbf{O}$ 与统计量 $(m,\ell)$，就能在反向时从 SRAM 里的 $\mathbf{Q},\mathbf{K},\mathbf{V}$ 块**重算出** $\mathbf{S}$ 与 $\mathbf{P}$。论文把它定位为一种**选择性 gradient checkpointing**，并强调差别：常规 checkpointing 的所有实现都是**拿速度换显存**，而这里即使 FLOPs 更多，反向**反而更快**，因为 HBM 访问减少了。

**Kernel fusion**：tiling 让整个算法能塞进**一个** CUDA kernel——载入、矩阵乘、softmax、（可选）掩码与 dropout、矩阵乘、写回。

**Algorithm 1（FlashAttention 前向）**转写：

```text
输入：HBM 中的 Q, K, V ∈ R^{N×d}；片上 SRAM 大小 M

 1: 设块大小 B_c = ⌈M/(4d)⌉,  B_r = min(⌈M/(4d)⌉, d)
 2: 在 HBM 中初始化 O = (0)_{N×d},  ℓ = (0)_N,  m = (−∞)_N
 3: 把 Q 切成 T_r = ⌈N/B_r⌉ 块（每块 B_r×d）；把 K, V 各切成 T_c = ⌈N/B_c⌉ 块（每块 B_c×d）
 4: 把 O 切成 T_r 块，ℓ 与 m 各切成 T_r 块
 5: for 1 ≤ j ≤ T_c do                          ← 外层循环：K, V
 6:     把 K_j, V_j 从 HBM 载入 SRAM
 7:     for 1 ≤ i ≤ T_r do                      ← 内层循环：Q
 8:         把 Q_i, O_i, ℓ_i, m_i 从 HBM 载入 SRAM
 9:         片上计算 S_ij = Q_i K_jᵀ ∈ R^{B_r×B_c}
10:         片上计算 m̃_ij = rowmax(S_ij),  P̃_ij = exp(S_ij − m̃_ij),  ℓ̃_ij = rowsum(P̃_ij)
11:         片上计算 m_i^new = max(m_i, m̃_ij)
                    ℓ_i^new = e^{m_i − m_i^new} ℓ_i + e^{m̃_ij − m_i^new} ℓ̃_ij
12:         写回 O_i ← diag(ℓ_i^new)^{-1} ( diag(ℓ_i) e^{m_i − m_i^new} O_i
                                          + e^{m̃_ij − m_i^new} P̃_ij V_j )
13:         写回 ℓ_i ← ℓ_i^new,  m_i ← m_i^new
14:     end for
15: end for
16: 返回 O
```

第 12 行是整个算法的心脏：**用新旧两个 $m$ 的差做指数缩放，把已累积的 $\mathbf{O}_i$ 重新归一化后，再加上当前块的贡献**。

**Theorem 1**：Algorithm 1 返回 $\mathbf{O}=\mathrm{softmax}(\mathbf{Q}\mathbf{K}^{\top})\mathbf{V}$，用 $O(N^2 d)$ FLOPs，且**除输入输出外只需 $O(N)$ 额外显存**。

> **批注**：注意 Theorem 1 里的 FLOPs 仍是 $O(N^2 d)$——**渐进计算复杂度一点没变**。变的只有访存量和显存。这正好呼应 [Transformer 笔记](../2017-attention-is-all-you-need/README.md) Q5 的判断：Table 1 那三列衡量不到 FlashAttention 做的任何事情，因为它优化的维度**根本不在那张表上**。一个方法能带来 7.6× 加速却不改变任何一个渐进复杂度，这件事本身就说明该换尺子了。
>
> 另外值得注意 $B_r = \min(\lceil M/4d \rceil, d)$ 里那个容易被忽略的 $d$：行块大小额外被 head dim 卡住。原因是第 12 行累积 $\mathbf{O}_i$ 时需要 $B_r \times d$ 的片上空间，若 $B_r$ 随 $M/d$ 增长而 $d$ 又大，这块会挤爆 SRAM。**这是「块大小受 SRAM 容量约束」的具体形式，而不是一句空话**。

#### 3.2 Analysis: IO Complexity of FlashAttention

**Theorem 2**（$d \le M \le Nd$）：标准注意力需 $\Theta(Nd+N^2)$ 次 HBM 访问，FlashAttention 需 $\Theta(N^2 d^2 M^{-1})$ 次。

**证明思路**（论文正文给的直觉）：SRAM 容量为 $M$，故每次可载入 $\Theta(M)$ 大小的 $\mathbf{K},\mathbf{V}$ 块（第 6 行）。对每一对 $\mathbf{K},\mathbf{V}$ 块，都要遍历一遍所有 $\mathbf{Q}$ 块（第 8 行），于是对 $\mathbf{Q}$ 一共扫了 $\Theta(NdM^{-1})$ 遍；每遍载入 $\Theta(Nd)$ 个元素，合计 $\Theta(N^2d^2M^{-1})$。反向同理：标准 $\Theta(Nd+N^2)$，FlashAttention $\Theta(N^2d^2M^{-1})$。

代入典型值：$d$ 为 64–128、$M$ 约 100KB，则 $d^2 \ll M$，因此访存少很多倍（Figure 2 里至多 9×）。

**Proposition 3（下界）**：不存在算法能对**所有** $M\in[d,Nd]$ 都用 $o(N^2d^2M^{-1})$ 次 HBM 访问算出精确注意力。证明依赖 $M=\Theta(Nd)$ 时任何算法都要做 $\Omega(N^2d^2M^{-1})=\Omega(Nd)$ 次访问。论文坦承这类「在 $M$ 的子区间上成立」的下界是流式算法文献里的常见形式，并把「以 $M$ 为参数的参数化复杂度下界」列为未来工作。

**实证验证**（Figure 2）：

- **左**：即便 FlashAttention 因反向重计算而 FLOPs 更高，访存少得多，因此快得多（66.6 vs 75.2 GFLOPs、40.3 vs 4.4 GB、41.7 vs 7.3 ms）。
- **中**：扫块大小 $B_c$，块越大访存越少、运行越快；但**超过 256 之后**运行时间转由其他因素（算术操作）主导，且更大的块装不进 SRAM。

> **批注**：Figure 2 中间那条曲线是全文最有工程价值的一张图，它给出了**收益的终点**。块大小的增大不是无限有益的：先是访存收益递减，然后被算力顶住，最后被 SRAM 容量物理卡死。这条「先访存受限、后算力受限」的曲线形状，在任何 tiling 优化里都会出现——**调块大小时应该去找那个拐点，而不是一味调大**。
>
> Proposition 3 的下界也值得用正确的方式理解：它说的是**不能在所有 $M$ 上渐进地更好**，不是说 FlashAttention 在每个具体 $M$ 上都最优。论文自己把更强的参数化下界留给了未来工作，这是诚实的表述。

#### 3.3 Extension: Block-Sparse FlashAttention

把 FlashAttention 推广到近似注意力：给定块状的稀疏掩码 $\mathbf{M}\in\{0,1\}^{N/B_r\times N/B_c}$，算法与 Algorithm 1 **完全相同，只是跳过零块**。

**Proposition 4**：block-sparse FlashAttention 需 $\Theta(Nd+N^2d^2M^{-1}s)$ 次 HBM 访问，其中 $s$ 是非零块比例。

即稀疏性直接按比例改善 IO 复杂度里的大项。大 $N$ 时 $s$ 常取 $N^{-1/2}$ 或 $N^{-1}\log N$，对应 $\Theta(N\sqrt N)$ 或 $\Theta(N\log N)$ 的 IO 复杂度。下游实验用固定的 **butterfly 稀疏模式**（已被证明可逼近任意稀疏结构）。LRA 上 block-sparse 版加速 2.8×，精度与标准注意力持平。

> **批注**：这一节的定位是「证明 FlashAttention 是个**原语**而不只是一个 kernel」。更有意思的是它对近似注意力这条路线的回应：论文在 §1 批评近似方法没有真加速，这里给出的解释是它们**卡在访存开销上**——把它们架在 IO-aware 的实现之上，稀疏性的理论收益才真正兑现成墙钟收益。这是一个相当有风度的处理：不是否定整条路线，而是指出它缺了一层地基。

### 3.4 Experiments

#### 4.1 Faster Models with FlashAttention

**BERT**（BERT-large、Wikipedia、从 MLPerf 提供的同一初始化出发、跑到 72.0% MLM 目标精度、8×A100、10 次平均）：17.4±1.4 分钟 vs Nvidia MLPerf 1.1 记录的 20.0±1.5 分钟，**快 15%**。

**GPT-2**（OpenWebText、8×A100）：

| 实现 | ppl | 训练时间（加速比） |
| --- | --- | --- |
| GPT-2 small - HuggingFace | 18.2 | 9.5 天（1.0×） |
| GPT-2 small - Megatron-LM | 18.2 | 4.7 天（2.0×） |
| GPT-2 small - FlashAttention | 18.2 | **2.7 天（3.5×）** |
| GPT-2 medium - HuggingFace | 14.2 | 21.0 天（1.0×） |
| GPT-2 medium - Megatron-LM | 14.3 | 11.5 天（1.8×） |
| GPT-2 medium - FlashAttention | 14.3 | **6.9 天（3.0×）** |

论文强调 **ppl 与基线相同，因为模型定义没有改**，附录 E 给出了训练全程的验证 ppl 曲线，确认数值稳定性与基线一致。

**Long-range Arena**（各任务序列长 1024–4096）：FlashAttention 相比标准注意力加速 **2.4×**，平均精度 59.8 vs 59.3；block-sparse 版 2.8×、59.6。论文加了一条诚实的脚注：**LRA 精度高度依赖调参过程，作者复现的基线比原始对比中报告的更好**。

#### 4.2 Better Models with Longer Sequences

**长上下文语言建模**：GPT-2 small 用 4K 上下文时，仍比 Megatron-LM 的 1K 上下文**快 30%**，且 ppl 好 0.7（17.5 vs 18.2）。完整阶梯：

| 实现 | 上下文长度 | ppl | 训练时间（加速比） |
| --- | --- | --- | --- |
| Megatron-LM | 1K | 18.2 | 4.7 天（1.0×） |
| FlashAttention | 1K | 18.2 | 2.7 天（1.7×） |
| FlashAttention | 2K | 17.6 | 3.0 天（1.6×） |
| FlashAttention | 4K | 17.5 | 3.6 天（1.3×） |

**长文档分类**（micro-$F_1$，在预训练 RoBERTa 上加长序列，位置嵌入按 Beltagy 等人的做法重复）：

| 序列长度 | 512 | 1024 | 2048 | 4096 | 8192 | 16384 |
| --- | --- | --- | --- | --- | --- | --- |
| MIMIC-III | 52.8 | 50.7 | 51.7 | 54.6 | 56.4 | **57.1** |
| ECtHR | 72.2 | 74.3 | 77.1 | 78.6 | **80.7** | 79.2 |

MIMIC-III 平均 2,395 token（最长 14,562），ECtHR 平均 2,197（最长 49,392）。论文注意到两者最优长度不同（16K vs 8K），推测是分布漂移所致：MIMIC-III 是专业医学文本，可能对文档长度的分布漂移更敏感。

**Path-X 与 Path-256**：任务是判断 $128\times128$（或 $256\times256$）黑白图中两点是否连通，图像**逐像素**喂进 Transformer。此前所有 Transformer 要么 OOM 要么只有随机水平。做法是先在 Path-64 上预训练，再通过空间插值位置嵌入迁移到 Path-X。FlashAttention 在 Path-X（seq 16K）达到 **61.4**；block-sparse 版把序列推到 64K，在 Path-256 达到 **63.1**。论文脚注说明 Path-256 序列更长但路径相对更短，所以更容易拿高分。

#### 4.3 Benchmarking Attention

单张 A100 40GB，带 dropout 与 padding mask，扫序列长度。

**运行时**：运行时间随序列长度**二次增长**，但比精确注意力基线快得多——比 PyTorch 实现至多 **3×**。许多近似/稀疏方法的运行时随序列长度线性增长，但短序列下 FlashAttention 因访存少仍然更快；**交叉点出现在序列长 512 到 1024 之间**。block-sparse FlashAttention 则在所有序列长度上快过作者已知的一切精确、稀疏、近似实现。

**显存**：FlashAttention 与 block-sparse 版显存相同，随序列长度**线性**增长，比精确注意力基线省至多 **20×**，也比近似方法更省。除 Linformer 外的所有算法在 64K 之前就在 A100 上 OOM，而 FlashAttention 仍比 Linformer 省 **2×**。

> **批注**：512–1024 那个交叉点是论文自己给出的、最诚实的一处边界。它说明 FlashAttention **没有推翻近似注意力的价值，只是抬高了它们必须超越的门槛**——短序列下不必近似，长序列下近似仍有意义（而且最好架在 IO-aware 实现上，即 block-sparse 版）。一篇论文主动标出「我的方法从哪里开始不再占优」，可信度比只报最好数字高得多。

### 3.5 Limitations and Future Directions

论文自陈三条：

1. **编译到 CUDA**。当前做法是**为每种新的注意力变体手写一个 CUDA kernel**，需要用比 PyTorch 低级得多的语言，工程量大，而且**实现未必能跨 GPU 架构迁移**。作者呼吁一种「用高级语言写、编译成 IO-aware CUDA 实现」的方法，类比图像处理领域的 Halide。
2. **IO-aware 深度学习**。注意力是 Transformer 里最访存密集的计算，但**网络的每一层都要碰 HBM**，希望这套思路能推广到其他模块。
3. **多 GPU 的 IO-aware 方法**。当前实现「在常数因子意义上对单卡最优」，但注意力可以跨多卡并行，多卡会引入**卡间数据传输**这一新的 IO 层次，留给未来工作。

> **批注**：第 1 条在三年后回看几乎是一份路线图，而且被验证得相当准。「每个变体都要重写 kernel、且不跨架构可移植」正是后来 Triton、CUTLASS 抽象层、以及 FlashAttention-2/3 分别为 Ampere/Hopper 重写的原因——**这个痛点没有被消除，只是被工具链摊薄了**。第 3 条则直接指向了后来的 Ring Attention 等工作。

---

## 4. 关键问题解析

### Q1: 标准注意力的 HBM 访问量是多少？FlashAttention 降到多少？推导过程是什么？

**标准注意力：$\Theta(Nd + N^2)$**（Theorem 2）。

逐项数（对照 Algorithm 0）：

- 载入 $\mathbf{Q},\mathbf{K}$：$\Theta(Nd)$；
- 把 $\mathbf{S}\in\mathbb{R}^{N\times N}$ 写回 HBM：$\Theta(N^2)$；
- 读回 $\mathbf{S}$ 算 softmax，再把 $\mathbf{P}$ 写回：$\Theta(N^2)$；
- 读 $\mathbf{P}$ 与 $\mathbf{V}$、写 $\mathbf{O}$：$\Theta(N^2 + Nd)$。

合计 $\Theta(Nd + N^2)$。因为通常 $N \gg d$（GPT-2：$N=1024$、$d=64$），$N^2$ 项完全主导。**中间矩阵被完整地写两次读两次，这就是全部问题**。

**FlashAttention：$\Theta(N^2 d^2 M^{-1})$**，$M$ 为 SRAM 大小。

推导（论文正文的直觉，完整证明在附录 C）：

1. SRAM 容量 $M$，故每次载入的 $\mathbf{K}_j,\mathbf{V}_j$ 块规模为 $\Theta(M)$（Algorithm 1 第 6 行）。因为 $\mathbf{K},\mathbf{V}$ 共 $\Theta(Nd)$ 个元素，外层循环次数为 $T_c = \Theta(Nd / M)$。
2. 对**每一个** $\mathbf{K},\mathbf{V}$ 块，内层要遍历所有 $\mathbf{Q}$ 块（第 8 行），即完整扫一遍 $\mathbf{Q}$（连带 $\mathbf{O},\ell,m$），载入 $\Theta(Nd)$ 个元素。
3. 于是总访存 $= T_c \times \Theta(Nd) = \Theta(Nd/M) \times \Theta(Nd) = \Theta(N^2d^2M^{-1})$。

**为什么这是巨大的改进**：代入 $d = 64\text{–}128$、$M \approx 100\text{KB}$，$d^2$ 比 $M$ 小很多倍。把两式相比（忽略 $Nd$ 项）：

$$\frac{\Theta(N^2)}{\Theta(N^2d^2M^{-1})} = \Theta\!\left(\frac{M}{d^2}\right)$$

**加速比正比于 $M/d^2$**——SRAM 越大、head dim 越小，收益越大。Figure 2 实测至多 9× 的访存降低（40.3 GB → 4.4 GB）与此吻合。

反向传播的结论相同：标准 $\Theta(Nd+N^2)$，FlashAttention $\Theta(N^2d^2M^{-1})$。

**下界**（Proposition 3）：不存在算法能对所有 $M\in[d,Nd]$ 都做到 $o(N^2d^2M^{-1})$。所以在这个意义上 FlashAttention 已经到顶了。

### Q2: online softmax / 分块归约如何做到不物化 $N\times N$ 矩阵而仍得到精确结果？

这是全文最需要讲透的一点，分三步。

**第一步：为什么 softmax 是障碍**。$\mathbf{S}=\mathbf{Q}\mathbf{K}^\top$ 与 $\mathbf{O}=\mathbf{P}\mathbf{V}$ 都是矩阵乘，天然可分块。卡住的是中间的 softmax：它按行做归一化，**分母是整行所有元素的和**。看起来必须先有完整的一行 $\mathbf{S}$ 才能算，而完整的一行就意味着物化。

**第二步：softmax 的代数可分解性**。数值稳定版的 softmax 需要三个量：行最大值 $m(x)$、指数化向量 $f(x)$、以及归一化因子 $\ell(x)$：

$$m(x)=\max_i x_i,\quad f(x)=\left[e^{x_1-m(x)},\dots,e^{x_B-m(x)}\right],\quad \ell(x)=\sum_i f(x)_i,\quad \mathrm{softmax}(x)=\frac{f(x)}{\ell(x)}$$

关键恒等式是：拼接向量的统计量可以由两段的统计量**精确组合**出来。设 $x = [x^{(1)}\ x^{(2)}]$：

$$m(x)=\max\big(m(x^{(1)}),\,m(x^{(2)})\big)$$

$$\ell(x)=e^{m(x^{(1)})-m(x)}\,\ell(x^{(1)})+e^{m(x^{(2)})-m(x)}\,\ell(x^{(2)})$$

**为什么这是精确的**：$\ell(x^{(1)})=\sum_i e^{x^{(1)}_i-m(x^{(1)})}$，乘上 $e^{m(x^{(1)})-m(x)}$ 后变成 $\sum_i e^{x^{(1)}_i-m(x)}$——**基准从局部最大值换到了全局最大值**。两段都换成同一基准后就能直接相加。这里没有任何近似，只是指数函数 $e^{a-b}=e^{a-c}\cdot e^{c-b}$ 的恒等变形。

**第三步：把输出也增量地修正**。光有统计量不够——已经写出去的输出块 $\mathbf{O}_i$ 是按**旧的**归一化因子算的，来了新块要修正它。Algorithm 1 第 12 行做的正是这件事：

$$\mathbf{O}_i \leftarrow \mathrm{diag}(\ell_i^{\mathrm{new}})^{-1}\Big(\mathrm{diag}(\ell_i)\,e^{m_i-m_i^{\mathrm{new}}}\,\mathbf{O}_i + e^{\tilde m_{ij}-m_i^{\mathrm{new}}}\,\tilde{\mathbf{P}}_{ij}\mathbf{V}_j\Big)$$

拆开读：

1. $\mathrm{diag}(\ell_i)\mathbf{O}_i$ —— **撤销**旧的归一化，还原成未归一化的加权和；
2. $e^{m_i-m_i^{\mathrm{new}}}$ —— 把基准从旧的行最大值换到新的；
3. $+\, e^{\tilde m_{ij}-m_i^{\mathrm{new}}}\tilde{\mathbf{P}}_{ij}\mathbf{V}_j$ —— 加上当前块的贡献（同样换算到新基准）；
4. $\mathrm{diag}(\ell_i^{\mathrm{new}})^{-1}$ —— 按新的归一化因子重新归一化。

**精确性的来源**：全部四步都是**恒等变换**，没有截断、没有采样、没有低秩近似。Theorem 1 因此能断言算法返回的恰好是 $\mathrm{softmax}(\mathbf{Q}\mathbf{K}^\top)\mathbf{V}$。

**代价是什么**：每处理一个新块都要重新缩放一次已有的 $\mathbf{O}_i$，这是额外的算术——正是它让 FLOPs 从 66.6 涨到 75.2 GFLOPs。**用多算 13% 的 FLOPs 换掉 90% 的访存**，这笔账在访存受限的算子上非常划算。

需要说明的是，论文明确表示 tiling 与 recomputation 都是**已有技术**（分块 softmax 引用了 Milakov & Gimelshein 的 online softmax 等工作），本文的贡献在于把它们组合起来、给出 IO 复杂度分析、并真正写出高效的融合 kernel。

### Q3: 反向传播为什么可以靠重计算而不是保存注意力矩阵？代价与收益如何权衡？

**为什么可以**：反向要算 $\mathbf{Q},\mathbf{K},\mathbf{V}$ 的梯度，通常需要 $\mathbf{S},\mathbf{P}\in\mathbb{R}^{N\times N}$。但 $\mathbf{S}$ 与 $\mathbf{P}$ 完全由 $\mathbf{Q},\mathbf{K}$ 与统计量 $(m,\ell)$ 决定——$\mathbf{S}_{ij}=\mathbf{Q}_i\mathbf{K}_j^\top$ 是一个矩阵乘，$\mathbf{P}$ 再由 $(m,\ell)$ 归一化得到。所以**只要保存 $\mathbf{O}$ 和 $(m,\ell)$（共 $O(N)$ 而非 $O(N^2)$），就能在反向时按块重算出来，全程在 SRAM 内完成**。

论文把它定位为一种**选择性 gradient checkpointing**。

**权衡为什么是划算的（这是最反直觉的一点）**：常规 gradient checkpointing 是**拿速度换显存**——论文明确说「所有我们知道的实现都必须用速度换显存」。FlashAttention 的重计算打破了这个取舍：**即使 FLOPs 更多，反向反而更快**。

原因在于两种资源的价格差。重算 $\mathbf{S},\mathbf{P}$ 的代价是**片上的算术操作**；保存并读回它们的代价是 $\Theta(N^2)$ 次 **HBM 访问**。而 A100 上 SRAM 带宽约 19 TB/s、HBM 只有 1.5–2.0 TB/s，差一个数量级；同时注意力是访存受限算子，算力本就有富余。**在算力过剩、带宽紧缺的机器上，重算比搬运便宜。**

Figure 2 左给出了整笔账（GPT-2 medium、seq 1024、head dim 64、16 头、batch 64、A100、前向+反向）：FLOPs 66.6 → 75.2（+13%），HBM 读写 40.3 → 4.4 GB（−89%），运行时间 41.7 → 7.3 ms（−82%）。

**这个权衡什么时候会失效**：如果算子本身是 compute-bound（算术强度高），或者硬件的算力/带宽比反过来，重算就不再划算。论文没有讨论这个边界，但它是 IO-aware 这套方法论的一般前提——见 Q5。

### Q4: block size 受什么约束？

Algorithm 1 第 1 行给出了两个块大小：

$$B_c=\left\lceil\frac{M}{4d}\right\rceil,\qquad B_r=\min\left(\left\lceil\frac{M}{4d}\right\rceil,\ d\right)$$

**约束一：SRAM 容量 $M$**。这是 $M/4d$ 的来源。片上同时要驻留 $\mathbf{K}_j,\mathbf{V}_j$（各 $B_c\times d$）以及 $\mathbf{Q}_i,\mathbf{O}_i$（各 $B_r\times d$）——四个 $\cdot\times d$ 的块，所以分母是 $4d$。A100 每个 SM 的 SRAM 只有 192 KB，这是硬上限。

**约束二：head dim $d$，且它以两种方式起作用**。

- 出现在**分母**里：$d$ 越大，同样的 SRAM 只能装下越少的行/列，块越小，外层循环次数越多，访存越多。这与 Q1 里「加速比正比于 $M/d^2$」是一致的——**$d$ 是 IO 效率的敌人**。
- 出现在 $B_r$ 的 $\min$ 里：行块大小额外被 $d$ 卡住上限。这是因为第 12 行累积输出块 $\mathbf{O}_i$ 需要 $B_r\times d$ 的片上空间，若 $B_r$ 随 $M/d$ 增长而 $d$ 又大，这块会挤爆 SRAM。

**约束三：收益递减与算力天花板（实测的，不在公式里）**。Figure 2 中的曲线扫了 $B_c$：块越大访存越少、越快，**但超过 256 之后运行时间转由算术操作主导**，不再随块增大而下降；而且更大的块根本装不进 SRAM。

所以完整的图景是：**块大小从小到大先受访存约束（越大越好），到某个点后受算力约束（不再变好），最后被 SRAM 容量物理卡死。** 实践中要找的是那个拐点，不是最大值。

这也解释了为什么 FlashAttention 对大 head dim 支持得更晚更吃力——原始实现的 head dim 上限是 64，后来才扩到 128。$d$ 同时挤压 $B_c$ 与 $B_r$，是这套方案最本质的一处受限。

### Q5: 「IO 复杂度」这个分析框架在什么条件下会失效？

论文自己给出了一部分边界，我再补几条。

**论文明说的：**

1. **多 GPU 场景不适用**。§5 承认当前实现「在常数因子意义上对**单卡**计算注意力是最优的」，多卡会引入卡间数据传输这一**新的 IO 层次**，现有分析没有覆盖。
2. **下界只在 $M$ 的子区间上成立**。Proposition 3 说的是不能对**所有** $M\in[d,Nd]$ 渐进地更好，不是每个具体 $M$ 上都最优。论文把参数化复杂度下界列为未来工作。
3. **实现不跨架构可移植**（§5 第 1 条）。$M$ 是个随硬件变的量，为 A100 调好的块大小换到别的架构上就不对了——**IO 复杂度是硬件相关的分析，不是硬件无关的算法性质**。

**我的补充〔以下为推断，论文未讨论〕：**

4. **算子本身 compute-bound 时框架失去意义**。整套分析成立的前提是「时间由访存决定」。§2.1 自己给了判据——算术强度。对内维度大的矩阵乘这类 compute-bound 算子，减少访存不会改善墙钟时间，重计算换访存的账也会反过来算亏。**IO-aware 是访存受限算子的方法论，不是普适的**。
5. **两级存储的模型过于简化**。分析只区分 HBM 与 SRAM 两层，实际还有寄存器、L2 cache、以及 A100 上的异步拷贝等机制。$\Theta(N^2d^2M^{-1})$ 里的常数因子被这些细节左右，这也是为什么论文能证明「常数因子内最优」却仍需要 FlashAttention-2 重写一版（后者优化的是并行划分与非 matmul 操作的占比，完全不体现在这个 IO 复杂度里）。
6. **它衡量不了并行度与占用率**。$\Theta(\cdot)$ 只数搬了多少字节，不管这些搬运能不能被足够多的线程块掩盖。FlashAttention-2 的主要改进正是在这个维度上——**同样的 IO 复杂度，两倍的实际速度**，这本身就说明该框架不完备。

**一句话总结这个框架的适用条件**：算子访存受限、存储层次可近似为两级、单设备、且硬件参数 $M$ 已知。满足时它极其有力（能同时给出算法、下界和调参依据）；不满足时它会给出误导性的乐观结论。

### Q6: 它与 vLLM 的 PagedAttention 是什么关系？两者都改了注意力的访存方式，区别在哪？

两者都在动注意力的访存，但**动的层次完全不同，且互不冲突**。

| 维度 | FlashAttention | PagedAttention |
| --- | --- | --- |
| 优化的存储层次 | **HBM ↔ SRAM**（片外到片上） | **HBM 内部的布局**（怎么摆放 KV） |
| 要消除的浪费 | $N\times N$ 中间矩阵的读写 | KV Cache 的**碎片与过量预留** |
| 主要场景 | **训练**（也用于 prefill） | **推理服务**的 decode 阶段 |
| 手段 | tiling + online softmax + 重计算 | 定长块 + block table 间接层 |
| 关键收益 | 时间 3–7.6×、显存 $O(N^2)\to O(N)$ | KV 有效利用率 20.4%–38.2% → 接近 100% |
| 结果精确性 | 精确 | 精确（不改变计算，只改存储位置） |

**核心区别**：FlashAttention 关心的是**一次注意力计算内部**，怎么把数据在 HBM 与 SRAM 之间搬得更少——它假设 KV 就在 HBM 里连续摆着。PagedAttention 关心的是**多个请求之间**，KV 在 HBM 里该怎么摆才不浪费——它不改变每次计算搬多少数据。

**两者叠加是常态而非例外**。实际系统里 PagedAttention 的 kernel 内部依然用 FlashAttention 的分块 online softmax，只是读 K/V 时多走一层 block table 的间接寻址。vLLM 论文自己也说明了这一点。所以正确的理解是：**FlashAttention 定义了「怎么算一次注意力」，PagedAttention 定义了「KV 存在哪里」**，两个问题正交。

**一个有意思的对照**：两篇论文批评的对象结构相同——都是「已有实现按最坏情况分配/搬运了大量根本用不上的数据」。FlashAttention 打的是「物化了根本不需要落地的中间矩阵」，PagedAttention 打的是「预留了根本用不到的 KV 空间」。两者都不改变数学，只改变数据在存储层次里的行踪。这正是 [IO-aware 算法设计](../../../concepts/io-aware-kernel-design.md) 这个概念的一般形式。

详见 [分页式 KV 管理](../../../concepts/paged-kv-memory.md) 与 [vLLM 笔记](../../llm-serving/2023-vllm-pagedattention/README.md)。

### Q7: 这篇与 Transformer 原论文是什么关系？它补上了 Table 1 缺的哪一维？

[Transformer 笔记](../2017-attention-is-all-you-need/README.md) §3.4 的 Table 1 用三项指标比较架构：

| Layer Type | Complexity per Layer | Sequential Operations | Maximum Path Length |
| --- | --- | --- | --- |
| Self-Attention | $O(n^2\cdot d)$ | $O(1)$ | $O(1)$ |
| Recurrent | $O(n\cdot d^2)$ | $O(n)$ | $O(n)$ |

**缺的那一维是显存**。$n\times n$ 的注意力矩阵在训练时必须保留用于反向传播，这项开销在三列里完全不可见，却是长序列训练的第一约束——模型不是跑得慢，是**根本跑不起来**（论文 §4.3 实测：除 Linformer 外所有方法在 64K 之前就在 A100 上 OOM）。

FlashAttention 把它从 $O(N^2)$ 降到 $O(N)$（Theorem 1：除输入输出外只需 $O(N)$ 额外显存），**而不改变 Table 1 里的任何一格**：

- Complexity per Layer 仍是 $O(N^2 d)$（Theorem 1 明确写了 $O(N^2d)$ FLOPs，甚至因重计算实测更多）；
- Sequential Operations 与 Maximum Path Length 完全没动，因为计算图的数学结构一个字没改。

**这就是这篇论文对 Table 1 最尖锐的反驳**：一个能带来 7.6× 墙钟加速、把可行序列长度从 2K 推到 64K 的方法，在那张表上是**完全隐形的**。所以问题不在于表里某一格填错了，而在于**这张表衡量的维度不完整**——它只计浮点运算与依赖结构，不计存储层次之间的数据搬运，也不计峰值显存。

Transformer 笔记 Q5 里还列了 Table 1 没覆盖的另外两件事（常数因子与硬件适配、推理侧的成本结构），三者是同一个问题的三个侧面：**渐进复杂度是一个刻意抹掉硬件的抽象，而 2017 年之后的性能故事几乎全部发生在被抹掉的那部分里。**

一个附带的历史观察：Transformer 论文选点积注意力而非加性注意力，理由正是「可以复用高度优化的矩阵乘实现」——**这是一个纯粹的硬件适配论证，但它没有被写进 Table 1**。作者当年已经知道这个维度重要，只是没有给它一个位置。

### Q8: 为什么说「精确」这件事比 3× 加速更重要？

论文自己没有这样强调，但这是它与几十种近似注意力方法命运不同的根本原因。

**近似方法的隐性成本**。采用一个近似注意力，使用者要承担：模型质量可能下降（且下降多少与任务相关，必须自己测）、可能需要重新调参、下游效果需要重新验证、以及「这个模型的注意力和别人的不一样」带来的长期维护负担。**这些成本都不出现在论文的加速比里，但它们是采用决策的真正阻力。**

**FlashAttention 把这份成本清零**。论文 §4.1 写得很明白：「FlashAttention 达到与另外两个实现相同的 ppl，因为**我们没有改变模型定义**」，附录 E 还给了训练全程的验证 ppl 曲线证明数值稳定性一致。这意味着它是一个 **drop-in 替换**——换上去，模型的数学行为完全不变，只是快了、省显存了。

**证据在论文自己的数据里**：LRA 表上，Linear Attention 的加速比是 2.3×、Performer 1.8×、Linformer 2.5×，与 FlashAttention 的 2.4× **处在同一量级**。但精度那一列上，Linformer 平均 54.9、Local Attention 56.0，而 FlashAttention 是 59.8（标准注意力 59.3）。**加速比相当，代价却完全不同。**

这解释了 §1 里那句诊断的后半句——近似方法「没有获得广泛采用」。论文把原因归给「没有真加速」，但即使某个近似方法确实快了，**「要不要承担精度风险」这道关仍然存在**。FlashAttention 绕过了这道关。

**可迁移的判断**：评估一个优化时，除了收益，还要看它**要求使用者放弃什么**。不要求放弃任何东西的优化，采用曲线会陡峭得多。〔这是我的归纳，论文未作此论述〕

---

## 5. 可迁移的知识点

- [IO-aware 算法设计](../../../concepts/io-aware-kernel-design.md) —— 本文是这个方法论最完整的示范：指出渐进复杂度只计浮点运算、忽略存储层次搬运，并给出 tiling、算子融合、重计算换访存三种通用手段。
- [Self-Attention](../../../concepts/self-attention.md) —— 本文不改变注意力的数学定义，只改变它的计算与存储方式；理解「softmax 按行耦合所有列」是理解本文难点的前提。
- [分页式 KV 管理](../../../concepts/paged-kv-memory.md) —— 与本文正交的另一层访存优化，见 Q6 的对比表。
- [KV Cache](../../../concepts/kv-cache.md) —— 本文主要面向训练与 prefill；推理 decode 阶段的访存瓶颈在 KV Cache 上，是另一个故事。

**跨领域可迁移的四条：**

1. **FLOP 减少不等于墙钟加速**。看到复杂度改进而没有墙钟数据时，先问常数因子与访存模式。本文用「FLOPs 多 13%、时间少 82%」把这条钉死了。
2. **在算力过剩、带宽紧缺的机器上，重算比搬运便宜**。这直接推翻了「gradient checkpointing 必然拿速度换显存」的常识。判据是算术强度。
3. **分块归约的可行性取决于该归约是否代数可分解**。softmax 之所以能分块，是因为拼接的统计量能从各段统计量精确组合（algebraic aggregation）。遇到看似「必须看到全部数据」的归约时，先检查它有没有这个性质。
4. **不要求使用者放弃任何东西的优化，采用速度会快得多**。精确 vs 近似的差别不在论文的加速比表里，却决定了实际影响力。

---

## 6. 批判与开放问题

### 6.1 作者承认的局限

1. **必须为每个注意力变体手写 CUDA kernel**（§5）。需要用比 PyTorch 低级得多的语言，工程量巨大。作者呼吁类似 Halide 的高级语言到 IO-aware 实现的编译路径。
2. **实现不跨 GPU 架构可移植**（§5）。这是上一条的推论，但后果更严重——每换一代硬件就要重做一次。
3. **仅对单卡最优**（§5）。多 GPU 会引入卡间传输这一新的 IO 层次，当前分析未覆盖。
4. **下界只在 $M$ 的子区间上成立**（Proposition 3 后的讨论）。作者坦承这类下界是流式算法文献的常见形式，更强的参数化复杂度下界留作未来工作。
5. **长序列下会被近似方法反超**（§4.3）。序列长 512–1024 之间出现交叉点，此后 Linformer 等线性方法开始更快。
6. **LRA 的基线依赖调参**（§4.1 脚注）。作者主动说明 LRA 精度高度依赖调参过程，且他们复现的基线**比原始论文报告的更好**——这是诚实的，但也说明该表上的对比不宜过度解读。

### 6.2 我的质疑

1. **端到端加速比与注意力加速比之间的落差没有被分析**。摘要里注意力计算本身快 7.6×，GPT-2 端到端只有 3.0–3.5×，相对 Megatron 更只有 1.7–1.8×。这个落差符合 Amdahl 定律的预期（注意力只占总时间的一部分），但论文**没有给出注意力占端到端时间比例的分解**。缺了这个分解，读者无法判断「继续优化注意力还能榨出多少」——而这恰恰是决定要不要投入下一轮工程的关键数据。〔推断：论文未提供该分解〕

2. **最亮眼的两个结果都建立在极不寻常的任务上**。Path-X（61.4）与 Path-256（63.1）是论文最有冲击力的宣传点——「首个超过随机水平的 Transformer」。但这两个任务是**把图像逐像素喂进 Transformer**，序列长度纯粹来自像素展开，与自然语言的长文档在依赖结构上差别很大。而且 Path-X 的做法是先在 Path-64 预训练再插值位置嵌入迁移，**并非直接在 16K 上训练**。这些结果证明了「显存不再是障碍」，但把它读作「长上下文能力的普遍提升」是过度延伸。

3. **长文档分类的收益曲线不单调，论文的解释是猜测**。MIMIC-III 上 512→1024 精度**下降**（52.8 → 50.7），之后才回升；ECtHR 上 8192 是峰值、16384 反而回落（80.7 → 79.2）。论文的解释是「可能源于细微的分布漂移」，用的是 may be（推测语气），没有做任何验证。**一个非单调的曲线配一个未经检验的解释，是这组实验里最弱的一环**——它甚至可能说明「更长上下文更好」这个论断本身有条件。

4. **GPT-2 的加速比在 small 与 medium 之间下降，未被讨论**。small 是 3.5×，medium 降到 3.0×（相对 HuggingFace）。模型变大时注意力占比下降是合理解释，但**这意味着加速比会随模型规模继续衰减**——而 2022 年之后的模型都比 GPT-2 medium 大得多。论文既没给出这个趋势的外推，也没在更大模型上验证。〔推断：论文最大只测到 GPT-2 medium〕

5. **重计算的代价只在一个配置上量化过**。Figure 2 左的 FLOPs 对比（66.6 vs 75.2）只针对 GPT-2 medium 的一组特定参数（seq 1024、head dim 64、16 头、batch 64）。而重计算的相对代价应当随 $N$、$d$、SRAM 大小变化。论文声称的「即使 FLOPs 更多也更快」是否在所有配置下成立，**没有扫参数验证**。考虑到这是全文最核心的反直觉论断，只有一个数据点略显单薄。

6. **「常数因子内最优」的说法被自己的后续工作打脸**。论文 §5 称当前实现「在常数因子意义上对单卡最优」。但一年后的 FlashAttention-2 在**不改变 IO 复杂度**的前提下又快了约 2×，靠的是更好的并行划分与减少非 matmul 操作。这不是论文的错误——它说的是 IO 意义上的最优，而 FA-2 优化的是别的维度——但它清楚地说明：**IO 复杂度不是性能的完整刻画，「最优」这个词在这里的适用范围比字面看上去窄得多**。

### 6.3 开放问题

1. **高级语言能否编译出 IO-aware 实现？** 作者自己提的问题，也是最有价值的一个。三年过去，Triton 部分地回答了它（能写出接近手写 CUDA 的 kernel 且可读性好得多），但「给定一个注意力变体，自动推导出最优的分块与融合策略」仍未解决。
2. **IO-aware 分析能否推广到多级、多设备的存储层次？** 单卡两级模型已经足够复杂，加上 L2、NVLink、跨节点网络之后，是否还存在一个可分析的框架？Ring Attention 等工作在做，但缺乏 Theorem 2 那样干净的复杂度刻画。
3. **除注意力之外还有哪些算子值得 IO-aware 重写？** 作者提到「网络的每一层都要碰 HBM」。实践中 fused LayerNorm、fused optimizer 等确实跟进了，但缺少一个系统性的判据来回答「哪些算子值得投入手写 kernel」。算术强度是候选判据，但论文没有把它发展成方法。
4. **推理侧的 IO 分析是什么样的？** 本文的分析针对训练/prefill（$\mathbf{Q}$ 有 $N$ 行）。自回归 decode 时 $\mathbf{Q}$ 只有一行，注意力从矩阵乘退化为矩阵-向量乘，算术强度极低，瓶颈完全变成读 KV Cache。**同一套 IO 视角在那里会得出完全不同的结论**（这正是 FlashDecoding 与 PagedAttention 的领域）。

### 6.4 后续可读

- **FlashAttention-2（2023）** —— 在相同 IO 复杂度下通过更好的并行划分再快约 2×，是「IO 复杂度不完备」的直接证据。
- **FlashAttention-3（2024）** —— 针对 Hopper 架构的异步与 FP8 重写，印证 §5 「不跨架构可移植」这条局限。
- **vLLM / PagedAttention（SOSP'23）** —— 正交的另一层访存优化，见 [笔记](../../llm-serving/2023-vllm-pagedattention/README.md) 与 Q6 的对比。
- **Online normalizer calculation for softmax（Milakov & Gimelshein, 2018）** —— 分块 softmax 的直接前作。
- **Self-attention Does Not Need $O(n^2)$ Memory（Rabe & Staats, 2021）** —— 独立提出的省显存注意力，论文附录 B.5 专门做了对比。
- **Ring Attention** —— 沿 §5 第 3 条「多 GPU IO-aware」方向的后续。
