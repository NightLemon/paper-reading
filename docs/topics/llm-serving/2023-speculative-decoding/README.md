---
title: "Fast Inference from Transformers via Speculative Decoding"
authors: ["Yaniv Leviathan", "Matan Kalman", "Yossi Matias"]
affiliation: "Google Research, Mountain View, CA, USA"
venue: "ICML 2023"
year: 2023
arxiv: "2211.17192v2"
url: "https://proceedings.mlr.press/v202/leviathan23a.html"
topic: "llm-serving"
tags: ["speculative-decoding", "sampling", "autoregressive", "inference", "latency"]
concepts: ["speculative-decoding", "self-attention", "kv-cache", "continuous-batching"]
status: "done"
rating: 5
read_date: "2026-09-08"
---

# Fast Inference from Transformers via Speculative Decoding

> **一句话结论**：先用廉价模型生成一段草稿，再让目标模型并行计算这段路径上的条件分布，通过接受检验与残差重采样保持目标分布，在硬件有算力余量时减少昂贵模型的串行调用。

---

## 1. 元信息

| 项目 | 内容 |
| --- | --- |
| 作者 / 机构 | Yaniv Leviathan、Matan Kalman（同等贡献）、Yossi Matias；Google Research |
| 发表 | ICML 2023；首次预印本 2022，本目录按会议年份归档；所读版本 arXiv v2（2023-05-18） |
| 原文 | [PDF](paper.pdf) · [HTML](paper.html) · [PMLR](https://proceedings.mlr.press/v202/leviathan23a.html) · [固定 arXiv 版本](https://arxiv.org/abs/2211.17192v2) |
| HTML 阅读 | 样式与部分图像从 arXiv 联网加载；离线阅读使用同目录 PDF |
| 实现背景 | 在 [T5X](https://github.com/google-research/t5x) 基础上实现并与其标准解码比较；原文未给出涵盖全部实验的固定复现 commit |
| 关键词 | speculative sampling · draft / target model · acceptance rate · residual distribution |
| 前置知识 | 条件概率、[Self-Attention](../../../concepts/self-attention.md)、[KV Cache](../../../concepts/kv-cache.md)、自回归与 teacher forcing |

**阅读约定**：$p$ 表示目标模型的条件分布，$q$ 表示草稿模型的条件分布，均已按实际采样策略处理。随机采样时本文保证**输出分布相同**；不能由此推出同一随机种子会生成完全相同的文本。贪心解码在目标选择及 tie-breaking 一致的条件下可以保持相同序列。

---

## 2. 摘要速览（5 分钟版）

### 2.1 要解决的问题

标准自回归每生成一个 token 都要调用一次目标模型；下一次输入又依赖上次输出。大模型因此反复读取权重和 KV，算术资源却可能没有充分利用。

如果已有一个快速小模型能经常猜对后续 token，就可以先生成候选路径，把目标模型在该路径上的多个条件分布一起算出来。难点是：草稿错误时如何纠正，才能让结果仍服从原目标分布。

### 2.2 核心方法

草稿模型自回归生成 $\gamma$ 个 token，同时记录每一步分布 $q_i$。目标模型在已知草稿路径上并行计算 $p_1,\ldots,p_{\gamma+1}$。按顺序以概率 $\min(1,p_i(x_i)/q_i(x_i))$ 接受草稿；遇到第一个拒绝时停止，丢弃后面的猜测，从归一化的 $\max(p_i-q_i,0)$ 中抽取修正 token。若全接受，再从 $p_{\gamma+1}$ 抽一个额外 token。

```text
已确认前缀 → 小模型串行写 γ 个草稿 token
                     ↓
             大模型并行验证整条候选路径
                     ↓
     接受连续前缀 → 首个拒绝处修正，或全接受后补 1 token
                     ↓
             丢弃无效后缀，开始下一轮
```

一轮产生 1 到 $\gamma+1$ 个 token（尚未遇到 EOS / 长度终止时）。小模型仍然串行，大模型所需的串行轮数减少。

### 2.3 主要结果

Table 2 的墙钟实验：目标 **T5.1.1 XXL 11B**，草稿 **T5-small 77M**，已有相应任务 checkpoint，**单 TPUv4、batch size 1**，基线为 T5X 标准目标模型解码。

| 任务 | 采样 | 草稿长度 $\gamma$ | 接受率 $\alpha$ | 实测速度比 |
| --- | --- | --- | --- | --- |
| WMT En–De | greedy，temperature 0 | 7 | 0.75 | **3.4×** |
| WMT En–De | 标准采样，temperature 1 | 7 | 0.62 | **2.6×** |
| CNN/Daily Mail 摘要 | greedy，temperature 0 | 5 | 0.65 | **3.1×** |
| CNN/Daily Mail 摘要 | 标准采样，temperature 1 | 5 | 0.53 | **2.3×** |

上述均为严格保持目标分布的主算法，未启用 Appendix A.5 的 lenience。GPT-like / LaMDA 的结果主要是接受率测量；本文没有给它们相同形式的端到端墙钟实验。

### 2.4 我的评价

评分 ⭐⭐⭐⭐⭐。本文最有价值的是把“猜一段再验证”的工程直觉变成**随机采样下可证明正确**的算法。性能模型也很有用：草稿模型只要质量高还不够，必须足够便宜，而且验证时要有额外并行算力。

它提供的是延迟优化路径。高负载服务系统中，额外算术、草稿权重和临时 KV 会争用原本已紧张的资源，因此不能直接将单请求加速比当作服务容量提升。

---

## 3. 细读

> 3.1–3.6 对应原文 §1–§6，3.7 对应 Acknowledgments，3.8 对应 Appendix A.1–A.5。数学符号沿用原文；分布示例和工程边界明确标为本笔记展开。

### 3.1 Introduction

引言把既有加速分为两类：对所有位置统一降低成本，例如量化与架构修改；按位置难度自适应减少计算，例如 early exit。本文利用相似观察——有些 token 更容易预测——但允许目标模型在候选路径上提前执行，而不是仅靠启发式跳过验证。

硬件前提是大模型推理可能受带宽或通信限制，仍有算术余量。增加一些并行工作可以减少等待昂贵模型完成的串行次数。这与 [FlashAttention](../../foundations/2022-flashattention/README.md) 的 IO 视角相通：FLOPs 与墙钟时间不是一一对应的。

Figure 1 的示意来自 lm1b、8K 词表、97M 目标与 6M 草稿模型。图中绿色草稿被接受，红色被拒绝，蓝色修正；它展示算法路径，不能代替 Table 2 的 T5 墙钟测量。

> **批注**：本文的“不改变输出”在概率任务里应理解成不改变分布。标题和摘要的直观措辞容易让读者误以为每次抽样文本逐 token 一致，正文和 Appendix A.1 才给出准确保证。

### 3.2 Speculative Decoding

#### 2.1 Overview

目标模型为 $M_p$，廉价模型为 $M_q$，同一已确认前缀 $s$ 下分别得到 $p(x\mid s)$ 与 $q(x\mid s)$。草稿模型先提出 $x_1,\ldots,x_\gamma$，目标模型再计算每一个条件前缀对应的分布。

“并行计算”不要求复制 $\gamma+1$ 份模型权重。对 Transformer，候选序列已知后，可以用 causal mask 在一次多位置前向里评估候选路径，类似 teacher forcing；复用既有前缀 KV。具体 logits 和 KV 的缓存偏移需要实现处理。

一旦某个 $x_i$ 被拒绝，后续 $p_{i+1},\ldots$ 条件中含错误的 $x_i$，不能再用于修正后的序列。因此一轮只接受**连续草稿前缀**，不能跳过错误中间项继续捡后面的猜测。

#### 2.2 Standardized Sampling

原文把 temperature、top-k、nucleus 和 greedy 都视为先将 logits 转成目标概率分布，再标准抽样。于是算法操作的是**经过这些规则处理后的分布**，不是任意阶段的原始 logits。

对 greedy，目标分布可以写成确定的 argmax 位置概率 1，其余 0；平分时要有确定 tie-breaking。对 top-p 等策略，截断后要重新归一化。

正确性要求用于接受比率和残差修正的 $q$ 正是实际生成草稿的分布，$p$ 正是希望保留的目标分布。若从截断后的 $q$ 取样，却用未截断概率计算比率，证明的前提就被破坏了。

#### 2.3 Speculative Sampling

先看一个固定前缀。抽 $x\sim q$，以概率

$$a(x)=\min\left(1,\frac{p(x)}{q(x)}\right)$$

接受。若拒绝，从以下分布抽取一个替代 token：

$$r(x)=\frac{[p(x)-q(x)]_+}{\sum_y[p(y)-q(y)]_+},\qquad [z]_+=\max(z,0).$$

因为 $x$ 来自 $q$，其实际被抽中时理论上 $q(x)>0$。目标有而草稿没有的概率质量可通过 $r$ 补回，并不要求草稿分布覆盖目标的全部支持集。

把单步规则嵌入 Algorithm 1，得到下面的可读改写：

```text
输入：已确认前缀 s，草稿长度 γ
for i = 1..γ:
    q_i = 草稿模型在 s + x_1..x_(i-1) 下的实际采样分布
    x_i ~ q_i

并行求目标分布 p_1..p_(γ+1)，条件分别是同一候选路径的各前缀

for i = 1..γ:
    u ~ Uniform(0,1)
    if u > min(1, p_i(x_i) / q_i(x_i)):
        t ~ normalize(max(p_i - q_i, 0))
        return 已接受的 x_1..x_(i-1) + t

t ~ p_(γ+1)                         # 草稿全部接受
return x_1..x_γ + t
```

这里省略 EOS / 长度限制与缓存更新，以突出概率机制。实际终止规则和无效后缀处理见 Q6–Q7；它们必须与目标基线一致。

> **批注**：算法中的额外 token 有两个来源：拒绝时是修正，全部接受时是奖励式的追加。后者已经有 $p_{\gamma+1}$，因此不必再单独等一次目标模型推理才决定该 token。

### 3.3 Analysis

#### 3.1 Number of Generated Tokens

原文用 $\beta_s$ 表示给定前缀下的接受概率，以 $\alpha=E[\beta_s]$ 表示平均接受率。假设沿位置的这些接受率近似独立同分布，一轮产出数量 $N$ 的期望为：

$$E[N]=1+\alpha+\alpha^2+\cdots+\alpha^\gamma
=\frac{1-\alpha^{\gamma+1}}{1-\alpha}.$$

第一项 1 来自修正 / 额外 token；每多接受一个草稿，需要前面的草稿也全部通过。$\alpha=1$ 时使用极限 $E[N]=\gamma+1$，不直接代入产生 $0/0$。

**本笔记说明**：不作 IID 近似时，更一般的是 $E[N]=1+\sum_{i=1}^{\gamma}\Pr(\text{前 }i\text{ 个草稿全部接受})$。真实文本中容易或困难的位置常连续出现，因此用单个平均 $\alpha$ 的幂只是模型，而非算法正确性的条件。

Figure 2 展示草稿长度增加的边际收益：接受率有限时，越靠后的 token 越可能因前面拒绝而浪费。无限增长 $\gamma$ 不会线性增加有效产出。

#### 3.2 Calculating α

给定前缀，接受概率可以直接求出：

$$\beta=\sum_xq(x)\min\left(1,\frac{p(x)}{q(x)}\right)
=\sum_x\min(p(x),q(x)).$$

原文把距离记为 $D_{LK}$；按其定义，它等于 total variation distance：

$$D_{LK}(p,q)=\frac12\sum_x|p(x)-q(x)|,
\qquad\beta=1-D_{LK}(p,q).$$

这解释为什么“草稿质量”不能只用最终任务分数衡量：算法需要的是**相同前缀下概率质量的重叠**。两个模型各自任务表现相近，也可能以不同分布生成，从而接受率并不高。

#### 3.3 Walltime Improvement

设目标模型常规一步耗时为 $T$，草稿一步耗时为 $cT$。论文假设一次并行验证多个候选条件分布仍只需约 $T$，那么一轮耗时为 $(1+\gamma c)T$，速度比为：

$$S=\frac{1-\alpha^{\gamma+1}}{(1-\alpha)(1+\gamma c)}.$$

这个式子有三项前提：接受率的 IID 近似、验证并行拥有足够算力、输出足够长以摊薄边界和初始化成本。短生成还有天然上限，因为目标模型至少要被调用一次。

当 $\gamma=1$，$S=(1+\alpha)/(1+c)$，因此 $\alpha>c$ 时，在该理想模型中至少存在一个有收益的草稿长度。若所有草稿都拒绝，目标模型串行调用次数不增，但额外草稿计算仍会增加实际时间。

**部署时的扩展估算**〔本笔记推导〕：若实测验证一段草稿的耗时为 $v_\gamma T$，就应使用 $S\approx E[N]/(\gamma c+v_\gamma)$。原文的简式相当于 $v_\gamma\approx1$；高并发或大草稿长度时要重新测这个量。

#### 3.4 Number of Arithmetic Operations

令草稿每 token 算术量相对目标为 $\hat c$。原文 Theorem 3.11 给出总算术量相对基线的倍数：

$$F=\frac{(1-\alpha)(\gamma\hat c+\gamma+1)}{1-\alpha^{\gamma+1}}.$$

不被接受的草稿路径上，目标模型已经做了随后会丢弃的工作；所以墙钟变短与总 FLOPs 增加可以同时发生。并行计算多个位置能够摊薄权重与既有 KV 的读取，这是带宽受限时的机会。

Table 1 和 Figure 4 是**理论估算**，假设 $c=\hat c=0$，不能读作 measured benchmark。例如 $\alpha=0.8,\gamma=5$ 时，表中速度约 3.69×、算术量约 1.63×，它们没有计入真实草稿成本。

#### 3.5 Choosing γ

草稿越长，验证成功时的产出上限越高，但草稿串行时间也增长，后缀被浪费的概率增大。已知 $\alpha,c$ 后，对正整数 $\gamma$ 数值搜索上述速度函数即可；Figure 3 展示不同成本下的最优点变化。

Figure 5 将 encoder-decoder 的初始化和多轮草稿 / 验证画在时间线上。encoder 只在开头执行，其成本不能通过增加 decode 接受数无限摊薄，短输出尤其受影响。

原文还讨论用 oracle 或在线预测来动态改变草稿长度，但实际主实验固定草稿模型和 $\gamma$。关于 oracle 的更高上界依赖理想信息与充足算力，应与可部署结果分开。

#### 3.6 Approximation Models

概率正确性不要求草稿和目标是同样大的模型，甚至不要求草稿是神经网络；只要能在同一 token 空间中提供实际抽样分布。原文主实验使用同架构系列、相同采样标准化的小模型。

可能的草稿包括 n-gram、匹配前缀后的输入复制、非自回归模型和另一个自带投机加速的小模型。这些大多是探索方向。n-gram 的接受率可以非零，但原文把接受率代入理想成本得到的速度预测，不等于完成了全部端到端实现。

“模型越小越好”也不成立：若接受率过低，验证工作大量浪费。应联合比较接受率、草稿成本、额外存储与调度负担。

### 3.4 Experiments

#### 4.1 Empirical Walltime Improvement

**Setup**：T5.1.1 encoder-decoder，目标 XXL 11B，草稿分别为 small 77M、base 250M、large 800M；使用已有 WMT En–De 和 CNN/Daily Mail checkpoint，单 TPUv4、batch 1，temperature 0 / 1，基线 T5X 标准解码。

**Results（Table 2）**：除摘要列出的最优 small 配置，下面的对照最能说明“接受率更高不一定更快”：

| WMT En–De，greedy | 接受率 | 草稿长度 | 实测速度比 |
| --- | --- | --- | --- |
| T5-small 草稿 | 0.75 | 7 | 3.4× |
| T5-base 草稿 | 0.80 | 7 | 2.8× |
| T5-large 草稿 | 0.82 | 7 | 1.7× |

草稿更强提高了概率重叠，但执行成本增长抵消了收益。这是 §3 性能公式最直接的实证。

采样温度也影响接受率。在 T5-small / En–De 下，greedy 的 0.75 与 temperature 1 的 0.62 对应不同速度；这个趋势来自具体分布，不能当作温度对所有模型单调影响的定理。

> **批注**：Table 2 报告的是倍率，没有提供覆盖多种长度和并发水平的服务曲线。本次精读核对论文数据，未运行 TPU / GPU 性能复现。

#### 4.2 Empirical α Values

作者在目标模型生成的 10K tokens 上计算概率重叠，覆盖 GPT-like 97M、T5-XXL 11B 和 LaMDA 137B。GPT-like 测试使用 lm1b、BERT tokenization、8K 词表，6M 草稿为 2 层、256 维、4 heads；LaMDA 测试使用已有 100M、2B、8B 草稿。

Table 3 中，LaMDA 137B / 100M 在 temperature 0 / 1 的接受率分别为 0.61 / 0.57；其目标输出还经过 top-40 filter。数据可以说明小模型存在可用的概率重叠，**没有测得该配置的实际 speedup**。

还有一个需要保留的反例：GPT-like 97M / 6M 的接受率在 temperature 0 / 1 为 0.88 / 0.89，并不遵循“分布越尖，接受率总越高”的笼统趋势。论文本身的表格比该概括更精确。

### 3.5 Related work

原文比较蒸馏、量化、early exit 与 speculative execution。其主张是，在目标分布保持的条件下用额外并行计算换延迟，而不是单独减少所有算术量。

Blockwise Parallel Decoding 主要针对 greedy，并需要定制模型；Shallow Aggressive Decoding 主要面向输入与输出高度相似的复制情形。本文将概率修正推广到一般随机采样和更一般的草稿分布。

本版本也注明 Chen 等人的 [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) 为独立实现。两项工作应区分来源，不将另一篇的 Chinchilla 实验当作本篇自己的实验。

### 3.6 Discussion

作者明确承认：没有额外算力时，增加并发验证与浪费计算不一定有帮助。讨论的后续方向包括 beam search、定制草稿、直接优化接受率、层级草稿、在线调整模型和草稿长度，以及文本以外的模态。

这些方向说明算法与性能模型具有可扩展性，不能视为本篇已经实现并验证。主实验仍是固定草稿与固定 $\gamma$ 的 T5 文本任务。

### 3.7 Acknowledgments

致谢包含 LaMDA 数据计算和 XLA 实现方面的协助。它也帮助识别实验分工：LaMDA 接受率分析与 T5 墙钟实现是不同的证据来源。

### 3.8 Appendix A

#### A.1 Correctness of Speculative Sampling

固定一个前缀，记接受概率：

$$\beta=\sum_x\min(p(x),q(x)).$$

残差归一化常数等于 $1-\beta$，因为 $[p-q]_+=p-\min(p,q)$。输出某个 token $x$ 的概率由两条路径组成：

$$\Pr(\text{接受草稿并输出 }x)=q(x)a(x)=\min(p(x),q(x)),$$

$$\Pr(\text{拒绝后输出 }x)=(1-\beta)r(x)=p(x)-\min(p(x),q(x)).$$

二者相加恰为 $p(x)$。对正确的条件前缀依次使用该机制，构成目标模型的自回归联合分布；首个拒绝之后的草稿依赖错误条件，必须丢弃。

当 $p=q$，$\beta=1$，没有拒绝事件，因而无需计算分母为零的残差分布。这个零分母边界应该通过分支避免，而不是用任意均匀分布替代。

#### A.2 Speculative Sampling vs. Rejection Sampling

传统 rejection sampling 使用一个包络常数 $M\ge\sup_x p(x)/q(x)$，接受概率为 $p(x)/(Mq(x))$，通常还要求适当的支持集条件。本文按 token 最大限度接受重叠质量，再用残差一次性补齐。

附录比较二者接受效率。要点是这里已经算出了目标分布，所以可以利用 $p-q$ 的形状；不是简单重复“从草稿抽直到接受”的传统拒绝采样。

#### A.3 Theoretical Predictions vs. Empirical Runtimes

Table 4 根据 profiler 估计 $c$，比较 Theorem 3.8 与 Table 2。En–De greedy 的示例：

| 草稿 | $c$ | 理论速度比 | 实测速度比 |
| --- | --- | --- | --- |
| T5-small | 0.02 | 3.2× | 3.4× |
| T5-base | 0.04 | 3.3× | 2.8× |
| T5-large | 0.11 | 2.5× | 1.7× |

条件仍为目标 T5-XXL、单 TPUv4、batch 1、$\gamma=7$。作者将偏差归因于实现优化差异和 IID 近似。这里的 $c=0.11$ 也说明正文“$c$ 总小于 0.05”的概括不能覆盖表内所有草稿；实际估算应使用对应行的数值。

公式能辅助筛选参数，却不能替代 profiling；如果一个更大的草稿让验证资源或执行方式变化，简化模型的偏差还可能扩大。

#### A.4 Application to Beam Search

附录给出扩展设想：目标 beam width 为 $w$，草稿用 $u\ge w$，一次提出多步候选，再验证目标 top-$w$ 是否被草稿 top-$u$ 覆盖。其验证计算预算约与 $w+u\gamma$ 次目标求值相关。

beam search 维护的是多个路径与累计得分，不能直接套用单路径的残差采样伪代码。作者把更完整分析留作未来工作，本篇没有相应性能表证明一般 beam search 也有主实验同等加速。

#### A.5 Lenience

此附录主动放宽严格分布保证，通过较宽松接受规则增加接受率。原文讨论 $l\in(0,1]$ 时输出概率上界与 $p(x)/l$ 的关系；$l<1$ 已允许分布改变，不能继续沿用 Appendix A.1 的等分布结论。

Table 5 主要报告不同 lenience 下的接受率，并结合成本模型讨论可能速度。它与 Table 2 的严格主实验必须分开引用。本文的精读和概念页以严格算法为默认，不将这些额外倍数作为“无损加速”结果。

---

## 4. 关键问题解析

### Q1: 自回归有顺序依赖，大模型为什么能一次验证多个位置？

**A**：目标模型不能在未知前缀上准确知道未来 token，但草稿模型已经提供了一条候选序列。给定这条序列，各位置的条件前缀都已知，可以用 causal mask 同时算出沿候选路径的概率。

这与训练时 teacher forcing 的可并行性相似：输入位置已经给定，求每个位置的条件分布不需要等模型自己先抽出上一位置。只在检查完接受条件之后，才知道其中哪段计算真正有效。

因此它减少目标模型的串行调用，仍然保留候选提出和拒绝后重启的顺序性。一次拒绝会让后续依赖该错误 token 的验证作废。

### Q2: 为什么接受率是 min(1, p/q)，而不是只要求两个模型 top-1 相同？

**A**：top-1 相同比较适合解释 greedy，但随机采样需要保留整个分布。只要草稿在 token $x$ 上提供的质量 $q(x)$ 不超过目标 $p(x)$，可以全部接受；超过时只接受其中 $p(x)/q(x)$ 的比例。

这样接受路径贡献的质量正好是 $\min(p(x),q(x))$，即两个分布重叠部分。残余质量交给修正分布补齐，最后逐 token 得到目标概率。

不同草稿可以有完全不同的 top-1 但仍有较高分布重叠；相同 top-1 也不保证整个分布接近。因此一般随机采样下应看 total variation 与接受率，不能只看“猜中率”。

### Q3: 拒绝后为什么不能直接重新从目标分布 p 抽样？

**A**：因为一部分目标概率质量已经通过接受草稿的路径输出了。如果拒绝后再完整地从 $p$ 抽，会重复分配其中一些质量。

**本笔记三 token 示例**：

$$p=(0.6,0.3,0.1),\qquad q=(0.2,0.5,0.3).$$

接受路径分别贡献 $(0.2,0.3,0.1)$，总接受率 0.6。还缺的质量为 $(0.4,0,0)$，所以拒绝后应确定地选第一个 token，最终恰好得到 $p$。

若拒绝后从 $p$ 重新抽，最终会得到：

$$(0.2,0.3,0.1)+0.4(0.6,0.3,0.1)=(0.44,0.42,0.14),$$

已经偏离目标。这个例子说明残差重采样是正确性的核心，不是可随意简化的实现细节。

### Q4: “输出相同”是同分布，还是同一随机种子也逐字相同？

**A**：严格随机版本保证每个输出序列的概率与目标基线相同。两个算法消耗随机数的顺序和数量不同，使用同一 seed 并不自动生成同一条样本；要实现样本级耦合，需要额外设计，本文没有要求。

greedy 下，$p$ 在目标 argmax 处为 1，其他处为 0；草稿匹配就接受，不匹配就修正为目标选项。若数值计算与 tie-breaking 一致，最终序列可以与目标 greedy 一致。

实际浮点误差、截断策略和重复惩罚等变换必须在构造 $p,q$ 时保持正确语义。数学上的等分布不能替代实现验证。

### Q5: 选更强的草稿总能更快吗？如何选择 γ？

**A**：更强草稿通常提高 $\alpha$，但也提高 $c$。Table 2 的 En–De greedy 实验中，T5-small → large 让接受率 0.75 → 0.82，速度却从 3.4× 降到 1.7×，说明成本更重要。

固定草稿后，$\gamma$ 也有取舍：大一些能摊薄目标调用，但串行草稿成本线性增长，连续接受更多位置的概率递减。先用 $E[N]/(1+\gamma c)$ 筛选，再按实际验证时间和负载 profiling。

**教学计算**：取 $\alpha=0.8,c=0.02,\gamma=5$，则 $E[N]=3.68928$、理想速度约 $3.68928/1.1=3.35$。这是假设验证时间等于目标单步的模型预测，不是论文或本机实测结果。

### Q6: 第一项就拒绝、全部接受、p=q 或支持集不相交时会怎样？

**A**：第一项拒绝时接受前缀长度为零，从当前残差分布生成一个修正 token，所以仍前进；全部接受时，使用已计算的 $p_{\gamma+1}$ 多生成一个 token。

$p=q$ 时接受率为 1，没有拒绝分支，残差归一化不应执行。支持集不相交时接受率为零，残差就是 $p$，算法仍正确但草稿与无效验证成为性能负担。

实际请求遇到 EOS 应立即按目标基线的终止规则结束，不能因为本轮还能产生额外 token 就越过 EOS；最大生成长度同理。这会截短最后一轮的有效产出，因此理论期望式默认忽略或摊薄这类边界。

### Q7: 两个模型的 KV Cache 如何处理？拒绝后的缓存能继续用吗？

**A**：目标与草稿权重不同，一般各有自己的 KV；同样的 token 文本不能直接共享两套模型的激活。草稿模型还会缓存尚未确认的路径，目标验证也会产生这条路径上的临时状态。

若第 $i$ 个草稿被拒绝，其后缓存依赖错误的前缀，必须丢弃或逻辑回退。已接受前缀的状态可以保留；修正 / 额外 token 被抽出后，其相应 KV 需要在后续正确的模型计算中补齐，不能假定 logits 已确定就已经有该 token 的所有 KV。

以上是由条件依赖推出的实现要求，原文没有给出完整生产缓存管理方案。它也解释了为什么投机解码除了多一份草稿权重，还会消耗临时显存，并对分页和 batch 状态提出新要求。

### Q8: 与 continuous batching、GQA、Sarathi-Serve 的收益能直接相乘吗？

**A**：不能。GQA 减少每轮 KV，continuous batching 合并多个请求，Sarathi-Serve 限制每轮 prefill 工作量，投机解码为一个请求增加多个候选位置的并行计算。它们会共同改变算术强度、KV 占用和可并发请求数。

低并发且带宽受限时，投机验证可能利用闲置算力；高并发时，其他请求本来就能填满算力，额外草稿计算可能挤占容量。原文只在 batch 1 测墙钟加速，没有测完整到达负载下的 goodput。

此外，一轮批量确认多个 token 的输出时序不同于逐轮产生一个 token。对流式服务，需要重新观察 TTFT、TBT 和队列行为，不能用平均 tokens/s 代替全部体验指标。这个组合评估属于后续系统工作。

---

## 5. 可迁移的知识点

- [Speculative Decoding](../../../concepts/speculative-decoding.md) —— 候选路径、连续前缀接受、残差修正和性能条件。
- [Self-Attention](../../../concepts/self-attention.md) —— 已知候选序列时用 causal mask 并行求多个条件分布。
- [KV Cache](../../../concepts/kv-cache.md) —— 复用已确认状态、回退被拒绝路径，区分草稿与目标模型缓存。
- [Continuous Batching](../../../concepts/continuous-batching.md) —— 请求间并行与同请求的候选并行会争用同一硬件资源。

**可迁移的方法**：用廉价近似提出可能有用的工作，用严格校正保留语义；把预测模型的价值定义为“正确率与成本的组合”；分别计算额外算术、减少的 IO 和串行依赖长度。它们是比单个模型名称更持久的设计思路。

---

## 6. 批判与开放问题

### 6.1 作者承认的局限

- 需要额外并行算力，算力已饱和时不保证收益。
- 期望速度分析采用 IID 接受率与近似固定验证时间，A.3 承认与测量存在偏差。
- 主实验固定草稿与长度，动态选择和层级草稿留作后续。
- beam search 只有初步扩展讨论，实验只覆盖文本。

### 6.2 我的质疑

1. **服务场景证据有限**。单 TPUv4、batch 1 的延迟收益不能直接说明高并发容量；额外模型与临时 KV 对可接纳请求数的影响未量化。
2. **长度相关成本没有系统扫描**。长上下文会改变验证的 attention 时间，短输出会放大初始化和最后一轮浪费；主表不足以覆盖全部长度组合。
3. **接受率测量和速度测量容易混用**。LaMDA 和 GPT-like 的重叠率只是可行性证据；没有其墙钟计时就不能计算可靠的部署加速。
4. **理论误差不是统一比例**。A.3 中有理论高估和低估，尤其较大草稿偏差明显。公式更适合指导 profiling，而非给产品承诺精确倍率。
5. **分布等价对实现提出严格要求**。实际 $q$、目标采样变换、残差归一化、EOS 和缓存回退必须一致。原文没有详列所有框架层面的实现验证，工程接入时仍需补齐。

### 6.3 开放问题

- 能否按当前负载与接受率动态调整草稿长度，让单请求延迟与总体容量共同受益？
- 草稿训练直接优化 total variation / 接受率，是否优于只优化常规语言模型 loss？
- 与 prefix caching、GQA 和 chunked prefill 组合时，如何预测验证真实成本和临时 KV 峰值？
- 对随机采样，用哪些可计算的分布检验和边界案例能有效发现实现偏差？

### 6.4 后续可读

- [Attention Is All You Need](../../foundations/2017-attention-is-all-you-need/README.md) —— 对照原论文“making generation less sequential”的展望。
- [GQA](../../foundations/2023-gqa/README.md) —— 模型结构侧减少 KV 读取，与减少目标串行轮数形成互补。
- [Sarathi-Serve](../2024-sarathi-serve/README.md) —— 将单请求延迟放入在线批处理与 SLO 容量的框架。
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) —— 独立工作，对照概率校正和实验设置。
- [Blockwise Parallel Decoding](https://arxiv.org/abs/1811.03115) —— 了解 greedy 条件下的前作与候选提出方式。
