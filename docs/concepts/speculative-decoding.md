---
concept: "Speculative Decoding"
aliases: ["投机解码", "speculative sampling", "draft-and-verify"]
tags: ["llm-serving", "sampling", "decoding", "latency"]
papers: ["topics/llm-serving/2023-speculative-decoding"]
---

# Speculative Decoding

> **一句话定义**：用廉价模型提出一段候选 token，再由目标模型并行验证并校正，使一次目标调用可以确认多个 token，同时保持希望采样的目标分布。

## 为什么需要它

自回归解码每个 token 都依赖前面的 token，目标模型因此要多次串行前向。在带宽受限、算力有余量时，多位置计算可以摊薄昂贵权重读取。困难在于预测不总正确，需要既丢弃无效路径，又保持输出概率不被草稿偏好改变。

## 它是怎么工作的

### 候选生成与连续前缀验证

给定已确认前缀，草稿模型按实际分布 $q_i$ 生成 $\gamma$ 个候选。目标模型在候选路径的各前缀上并行计算 $p_1,\ldots,p_{\gamma+1}$。顺序验证，直到第一次拒绝；后续候选依赖被拒绝的 token，不能继续接受。

固定某一步，若候选 $x\sim q$，接受概率为：

$$a(x)=\min\left(1,\frac{p(x)}{q(x)}\right).$$

若拒绝，修正 token 来自：

$$r(x)=\frac{[p(x)-q(x)]_+}{\sum_y[p(y)-q(y)]_+}.$$

全部候选接受时，从 $p_{\gamma+1}$ 再抽一个 token；否则输出已接受前缀加一个修正 token。一轮在未提前终止时产生 1 到 $\gamma+1$ 个 token。

### 为什么保持目标分布

接受草稿并输出 $x$ 的概率为 $q(x)a(x)=\min(p(x),q(x))$。记总接受概率 $\beta=\sum_x\min(p(x),q(x))$，拒绝后输出 $x$ 的概率为：

$$(1-\beta)r(x)=p(x)-\min(p(x),q(x)).$$

两条路径相加就是 $p(x)$。$p=q$ 时不会拒绝，无需对零残差归一化。

这里 $p,q$ 都必须对应真正使用的采样分布，包括 temperature、top-k / top-p、归一化等处理。即使草稿在某些 token 上概率为零，目标缺少的质量也可通过残差补回。

### 性能模型

接受概率与 total variation distance 的关系为：

$$\beta=1-\frac12\sum_x|p(x)-q(x)|.$$

设平均接受率为 $\alpha$，草稿单步成本相对目标为 $c$。在接受率近似 IID、验证 $\gamma$ 个草稿的时间约等于目标普通一步、输出足够长的条件下：

$$E[N]=\sum_{i=0}^{\gamma}\alpha^i,\qquad
S\approx\frac{E[N]}{1+\gamma c}.$$

草稿更强可能提高 $\alpha$，也可能提高 $c$；草稿更长提高产出上限，也增加串行草稿成本与被丢弃的后缀工作。应先用公式筛选，再测真实验证耗时。

### 状态和终止规则

草稿与目标通常各自保存 KV。拒绝后回退无效后缀，保留已接受前缀；修正 token 的 KV 需在正确路径上补算。遇到 EOS 或长度上限应立即按目标基线规则终止，不能继续返回该轮尚余的候选。

随机版本保证同分布，不保证同 seed 得到同一条样本。greedy 在数值与 tie-breaking 一致时可保持目标序列。

## 关键性质与代价

| 收益 | 代价与条件 |
| --- | --- |
| 减少目标模型串行调用 | 草稿本身仍需计算，可能成为瓶颈 |
| 严格算法保持目标分布 | 需要正确记录 $q$，执行接受与残差修正 |
| 利用带宽受限时的算力余量 | 验证和无效后缀增加 FLOPs，满载时可能降低容量 |
| 可用已有小模型，无需改目标模型 | 增加权重、KV 与状态管理成本 |
| 一轮可确认多个 token | 流式输出节奏改变，需重新衡量 TBT / TTFT |

## 常见误解

- **误解**：验证后拒绝的 token 直接从 $p$ 重抽即可 → **实际**：接受路径已分配部分概率，必须用残差补回，否则会偏离目标。
- **误解**：草稿后面猜对的 token 可以跳过中间错误继续保留 → **实际**：其条件前缀已不同，后续验证失效。
- **误解**：目标调用更少就一定更快 → **实际**：草稿、验证、通信与缓存管理时间都必须计入。
- **误解**：更快意味着总 FLOPs 更少 → **实际**：可以用更多并行计算换更短等待时间。
- **误解**：同分布等于同一随机种子逐 token 相同 → **实际**：随机数消费顺序不同，样本路径不一定相同。
- **误解**：放宽接受条件仍然无损 → **实际**：lenience 等变体改变分布保证，要与严格算法分开说明。

## 出现在哪些论文里

- [Fast Inference from Transformers via Speculative Decoding](../topics/llm-serving/2023-speculative-decoding/README.md) —— 推导随机接受与残差校正、成本模型及正确性证明。在目标 T5.1.1 XXL 11B、草稿 T5-small 77M、单 TPUv4、batch 1、WMT En–De、草稿长 7 的 Table 2 中，相对 T5X 标准解码，greedy / temperature 1 的实测速度比分别为 3.4× / 2.6×；这是单请求墙钟实验，非高并发服务容量。

## 延伸阅读

- [Self-Attention](self-attention.md) · [KV Cache](kv-cache.md) · [Continuous Batching](continuous-batching.md)。
- [GQA](grouped-query-attention.md)：减少每轮 KV 需求，改变投机验证的成本结构。
- [Chunked Prefill](chunked-prefill.md)：从调度层处理多位置计算与生成延迟。
- [独立的 Speculative Sampling 工作](https://arxiv.org/abs/2302.01318)：对照采样校正与实验边界。
