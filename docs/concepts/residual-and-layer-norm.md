---
concept: "残差连接与 Layer Normalization"
aliases: ["Residual Connection", "LayerNorm", "Post-LN", "Pre-LN"]
tags: ["transformer", "training-stability", "normalization"]
papers: ["topics/foundations/2017-attention-is-all-you-need"]
---

# 残差连接与 Layer Normalization

> **一句话定义**：残差连接让每个子层学习「在输入基础上加什么」而不是「输出什么」，Layer Normalization 把每个样本每个位置的特征向量重新标准化——两者共同决定了深层堆叠能不能训得动。

## 为什么需要它

深层网络的两个老问题：

- **梯度传播**：不加捷径时，梯度要连乘穿过每一层的雅可比，层数一多就非指数放大即指数衰减。
- **内部分布漂移**：前面层的参数一变，后面层看到的输入分布就变，学习率难以统一设定。

残差解决第一个（提供一条恒等路径，梯度可以直通），归一化解决第二个（把每层输入拉回可控范围）。

选 LayerNorm 而非 BatchNorm 的原因：BatchNorm 沿 batch 维统计均值方差，对**变长序列**和**小 batch** 都不友好，且推理时需要维护 running statistics。LayerNorm 只在单个样本的特征维上统计，与 batch 大小和序列长度都无关。

## 它是怎么工作的

$$\mathrm{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

其中 $\mu$、$\sigma^2$ 是**沿特征维**（长度 $d_{\text{model}}$）计算的均值与方差，$\gamma$、$\beta$ 是可学习的缩放与平移。

### 两种摆放位置

**Post-LN**（归一化在残差相加之后）：

$$x_{l+1} = \mathrm{LayerNorm}\big(x_l + \mathrm{Sublayer}(x_l)\big)$$

**Pre-LN**（归一化在子层输入侧）：

$$x_{l+1} = x_l + \mathrm{Sublayer}\big(\mathrm{LayerNorm}(x_l)\big)$$

差别看似只是括号位置，实际影响很大：Pre-LN 中从 $x_0$ 到 $x_L$ 存在一条**未经归一化的恒等路径**，梯度可以完全直通；Post-LN 中每一层的输出都要过一次归一化，恒等路径被打断。

## 关键性质与代价

| | Post-LN | Pre-LN |
| --- | --- | --- |
| 恒等路径 | 被归一化打断 | 完整直通 |
| 训练初期稳定性 | 差，深层梯度量级偏大 | 好 |
| 是否需要 warmup | 需要 | 可选 |
| 最终质量 | 调好后通常略优 | 通常略逊，差距不大 |
| 层数可扩展性 | 差 | 好 |

原始 Transformer 采用的是 **Post-LN**。今天多数大模型实现改用 Pre-LN 或其变体（如 RMSNorm + Pre-LN）。

## 常见误解

- **误解**：残差连接是为了「保留原始信息」→ **实际**：主要作用是重构优化问题（学增量而非学映射）并提供梯度直通路径。信息保留是副产品。
- **误解**：Post-LN 与 Pre-LN 只是实现风格差异 → **实际**：它直接决定了是否必须使用学习率 warmup，以及模型能堆多深。
- **误解**：LayerNorm 沿 batch 维归一化 → **实际**：沿特征维。这正是它优于 BatchNorm 处理变长序列的原因。
- **误解**：残差要求子层输出维度与输入相同是个巧合 → **实际**：这是**强制约束**。Transformer 把所有子层与 embedding 层的输出统一为 $d_{\text{model}}$，正是为了让残差相加成立。

## 出现在哪些论文里

- [Attention Is All You Need](../topics/foundations/2017-attention-is-all-you-need/README.md) —— 每个子层外包 $\mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))$，即 Post-LN；并因此规定所有子层输出维度统一为 $d_{\text{model}} = 512$。这个结构选择是该文必须使用 warmup 的原因。

## 延伸阅读

- Pre-LN 相关分析：解释 Post-LN 为何依赖 warmup。
- RMSNorm：去掉均值中心化，只做缩放，计算更省。
