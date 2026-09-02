---
concept: "学习率 Warmup"
aliases: ["Learning Rate Warmup", "Noam schedule", "预热"]
tags: ["training", "optimization", "learning-rate"]
papers: ["topics/foundations/2017-attention-is-all-you-need"]
---

# 学习率 Warmup

> **一句话定义**：训练开始时从很小的学习率逐步升到目标值，而不是一上来就用目标学习率——用来度过参数与优化器状态都还没稳定的初期。

## 为什么需要它

训练最初的几百到几千步有两个不稳定来源：

- **优化器状态未收敛**。Adam 用滑动平均估计一阶矩与二阶矩。步数很少时二阶矩估计噪声极大，$\hat{m}/\sqrt{\hat{v}}$ 这个比值可能异常大，导致单步更新过大。
- **结构相关的梯度尺度问题**。在 Post-LN 结构下，初始化时深层的梯度量级偏大。直接使用目标学习率容易在前几十步就把参数推离良好区域，甚至发散。

warmup 用一段低学习率期把这两件事都缓过去。

## 它是怎么工作的

### Noam schedule

$$lrate = d_{\text{model}}^{-0.5} \cdot \min\!\left(step^{-0.5},\; step \cdot warmup^{-1.5}\right)$$

- $step < warmup$ 时取第二项，学习率**线性上升**。
- $step > warmup$ 时取第一项，按 $step^{-0.5}$ **衰减**。
- 两段在 $step = warmup$ 处取值相等，曲线连续，峰值为 $d_{\text{model}}^{-0.5} \cdot warmup^{-0.5}$。
- $d_{\text{model}}^{-0.5}$ 这个因子使调度对模型宽度自适应：模型越宽，学习率越小。

### 常见变体

- **线性 warmup + 余弦衰减**：当前最常见。warmup 段线性上升，之后按余弦曲线降到接近 0。
- **线性 warmup + 线性衰减**：实现最简单。
- **按 token 数而非 step 数计**：batch 大小可变时更稳健。

典型 warmup 长度：总步数的 1%–5%，或固定几千步。

## 关键性质与代价

| 收益 | 代价 |
| --- | --- |
| 抑制训练初期发散 | 多一个需要调的超参（warmup 长度） |
| 允许使用更大的峰值学习率 | 前期若干步的算力用于「升温」而非有效训练 |
| 对大 batch 训练几乎必需 | 长度设置不当会拖慢收敛或失去保护作用 |

## 常见误解

- **误解**：warmup 是 Transformer 的固有需求 → **实际**：它是对 **Post-LN** 结构的补偿。改用 Pre-LN 后 warmup 变为可选。这说明它修的是结构问题，不是模型族的本质需求。
- **误解**：warmup 越长越安全 → **实际**：过长会明显推迟有效学习，在固定 step 预算下反而损害最终质量。
- **误解**：warmup 只对 Adam 有必要 → **实际**：大 batch 下 SGD 同样受益。Adam 的二阶矩噪声只是原因之一，另一个原因来自结构与初始化。
- **误解**：峰值学习率与 warmup 长度可以独立调 → **实际**：在 Noam schedule 中峰值是 $d_{\text{model}}^{-0.5} \cdot warmup^{-0.5}$，改 warmup 会同时改峰值。

## 出现在哪些论文里

- [Attention Is All You Need](../topics/foundations/2017-attention-is-all-you-need/README.md) —— 提出 Noam schedule（$warmup = 4000$），是这类策略最常被引用的出处。论文只给出公式，未解释必要性；其 Post-LN 结构是需要 warmup 的结构原因。

## 延伸阅读

- Pre-LN 相关分析：论证何时可以省去 warmup。
- 大 batch 训练的学习率缩放规则（linear scaling rule 及其 warmup 要求）。
