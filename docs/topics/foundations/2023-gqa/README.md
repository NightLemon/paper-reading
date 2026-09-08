---
title: "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
authors: ["Joshua Ainslie", "James Lee-Thorp", "Michiel de Jong", "Yury Zemlyanskiy", "Federico Lebrón", "Sumit Sanghai"]
affiliation: "Google Research; Michiel de Jong: University of Southern California, work done at Google Research"
venue: "EMNLP 2023"
year: 2023
arxiv: "2305.13245v3"
url: "https://aclanthology.org/2023.emnlp-main.298/"
topic: "foundations"
tags: ["attention", "gqa", "mqa", "kv-cache", "uptraining", "inference"]
concepts: ["grouped-query-attention", "multi-head-attention", "kv-cache", "tensor-parallelism"]
status: "done"
rating: 5
read_date: "2026-09-08"
---

# GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints

> **一句话结论**：让每组 query 头共享一组 K/V，在 MHA 与 MQA 之间保留可调的表示能力；再用组内 K/V 权重平均和少量继续预训练，把已有 MHA checkpoint 转成适合低成本推理的模型。

---

## 1. 元信息

| 项目 | 内容 |
| --- | --- |
| 作者 / 机构 | Joshua Ainslie、James Lee-Thorp、Michiel de Jong（前三位同等贡献）、Yury Zemlyanskiy、Federico Lebrón、Sumit Sanghai；Google Research，de Jong 的工作在 Google Research 完成、署名含 USC |
| 发表 | EMNLP 2023；本笔记读取 arXiv v3（2023-12-23） |
| 原文 | [PDF](paper.pdf) · [HTML](paper.html) · [ACL Anthology](https://aclanthology.org/2023.emnlp-main.298/) · [固定 arXiv 版本](https://arxiv.org/abs/2305.13245v3) |
| HTML 阅读 | 样式与部分图像从 arXiv 联网加载；离线阅读使用同目录 PDF |
| 实现背景 | T5.1.1，JAX / Flax / [Flaxformer](https://github.com/google/flaxformer)；原文链接此框架，未给出覆盖全部论文实验的独立固定版本复现包 |
| 关键词 | grouped-query attention · multi-query attention · checkpoint conversion · uptraining |
| 前置知识 | [Multi-Head Attention](../../../concepts/multi-head-attention.md)、[KV Cache](../../../concepts/kv-cache.md)、[张量并行](../../../concepts/tensor-parallelism.md) |

**阅读范围**：原文主要实验是 **T5 encoder-decoder**，计时硬件为 **TPUv4**。把它用于理解现代 decoder-only 推理很有价值，但不能把下列速度数字直接当作 GPU 上 decoder-only 模型的实测结果。

---

## 2. 摘要速览（5 分钟版）

### 2.1 要解决的问题

每步自回归解码都需要读取历史 K/V。MHA 的每个 query 头有自己的 K/V 头，缓存随上下文长度与头数增长。MQA 把所有 query 头的 K/V 合并为一组，显著缩小缓存，但压缩得越激进，质量和训练稳定性越值得担心。

已有 MHA checkpoint 还有迁移问题：如果每次为了更快推理都要从头训练一个新模型，成本过高。本文同时解决“共享到什么程度”和“怎样利用已有模型”两个问题。

### 2.2 核心方法

记 query 头数为 $H$，K/V 组数为 $G$。将 query 头分为 $G$ 组，每组共享一个 K 头和一个 V 头；query 头本身仍保留。$G=H$ 是 MHA，$G=1$ 是 MQA。

从 MHA 转换时，对每组原来的 K/V 投影矩阵分别取平均，得到共享权重；随后沿用原预训练数据和配方继续训练。作者把这一步称为 **uptraining**，主要设置使用原预训练步数的 5%，之后还要进行下游任务 fine-tuning。

```text
原有 MHA checkpoint
  → 按 query 头分组
  → 组内 W_K、W_V 分别 mean-pool，保留 W_Q 与 W_O 的结构
  → 原配方继续预训练，让模型适应共享 K/V
  → 下游 fine-tuning 与质量、速度评测
```

### 2.3 主要结果

Table 1 比较 T5.1.1 Large / XXL；MQA-XXL 与 GQA-8-XXL 均做 5% uptraining。推理使用 greedy decoding，在 **8 个 TPUv4、每 TPU 尽可能填充但最多 32 个样本、分别优化各模型并行配置**的条件下计时。报告口径为每样本、每 TPUv4 chip 的时间，不是在线单请求端到端延迟。

| 模型 | Table 1 时间（s） | 七项任务平均分 |
| --- | --- | --- |
| MHA-Large | 0.37 | 46.0 |
| MHA-XXL | 1.51 | 47.2 |
| MQA-XXL，5% uptraining | 0.24 | 46.6 |
| GQA-8-XXL，5% uptraining | 0.28 | 47.1 |

七项任务为五个摘要数据集的 ROUGE-1、WMT 的 BLEU、TriviaQA 的 F1。任务输入 / 输出长度设置分别为 512 / 256、2048 / 512 或 2048 / 32，详见 §3.3。

用表中数值计算，GQA-8 的时间相对 MHA-XXL 约缩短到 $0.28/1.51\approx18.5\%$，对应约 **5.39×** 的该计时口径速度比；相对 MQA 则多耗时约 **16.7%**，平均质量更接近 MHA。这两个百分比是本笔记根据 Table 1 计算。

**单位提醒**：PDF Table 1 明确写秒，Figure 3 横轴却标了 ms，并沿用同量级数值。本笔记采用 Table 1 的秒单位并保留这个不一致，不把 0.28 写成 0.28 ms。

### 2.4 我的评价

评分 ⭐⭐⭐⭐⭐。这篇篇幅短，但将模型结构、已投入的训练成本和服务硬件联系起来。特别值得学的是：性能优化可以提供一个连续可调的设计轴，既不必保留所有 K/V 头，也不必一口气压成一个。

核心证据支持“在本文 T5 实验中，GQA-8 取得了良好的质量与速度折中”。“8 组在所有模型和硬件上最好”则超出论文范围。

---

## 3. 细读

> 3.1–3.5 对应原文 §1–§5；3.6–3.7 对应未编号的 Limitations、Acknowlegements；3.8 对应 Appendix A。为便于复用，张量形状与容量公式按原文机制展开，推导部分明确标注。

### 3.1 Introduction

论文先将自回归推理瓶颈定位到每步读权重和历史 K/V 的带宽开销。减少 K/V 头可以降低其中一项，但会改变模型的表示能力；仅凭理论缓存大小无法决定方案优劣。

作者指出当时有些模型使用 MQA，有些公开模型仍使用 MHA，因此研究目标包含“从一个已有 MHA checkpoint 得到更快版本”。这解释了为什么本文的贡献包括转换训练流程，而不仅是一张共享结构示意图。

> **批注**：这里优化的是一次持续运营决策：已经花钱训练出的模型，能否以较小追加成本适应新的推理需求？“改 checkpoint 然后继续训练”的路线把历史投入纳入了方法设计。

### 3.2 Method

#### 2.1 Uptraining

转换包含两个步骤。先将 K/V 投影头 mean-pool，再用相同预训练配方继续训练原步数的一小部分 $\alpha$。主要实验 $\alpha=0.05$。它与下游任务 fine-tuning 是两个阶段，不能把 5% 解读成在目标任务上微调几轮。

**组内平均的形状推导**〔本笔记按方法展开〕：设模型宽度为 $d$，每个头维度为 $d_h$，原始投影矩阵为 $W_i^K,W_i^V\in\mathbb R^{d\times d_h}$。组 $\mathcal G_g$ 含 $H/G$ 个 query 头，则：

$$\bar W_g^K=\frac{1}{|\mathcal G_g|}\sum_{i\in\mathcal G_g}W_i^K,\qquad
\bar W_g^V=\frac{1}{|\mathcal G_g|}\sum_{i\in\mathcal G_g}W_i^V.$$

保留 query 头的划分与输出投影结构，把组内各头的 K/V 换成对应共享投影。此后继续训练时模型参数仍可以更新，不能把“保留初始权重”理解为永久冻结。

Figure 1 展示的是 MHA → MQA 的特例：所有头成为一组。Figure 4 才给出转换策略的对照，平均优于只取第一个头和随机初始化。

mean-pooling 能保留更多原始权重信息，是作者对结果的直觉解释。由于后面还存在 QK、softmax、加权 V，**平均投影不保证平均注意力输出，更不保证输出分布保持**，这也是 uptraining 的必要背景。

#### 2.2 Grouped-query attention

Figure 2 将三种结构放在同一图中。对第 $i$ 个 query 头，设其组号为 $g(i)$，单头输出可写为：

$$O_i=\operatorname{softmax}\!\left(\frac{Q_iK_{g(i)}^T}{\sqrt{d_h}}+M\right)V_{g(i)}.$$

这里 $M$ 是适用的 attention mask。所有 $O_i$ 仍按 query 头拼接并乘输出矩阵。因此有 $H$ 组 query 和输出，有 $G$ 组 K/V；同组 query 可以因为 $Q_i$ 不同而产生不同注意力分布。

若忽略 batch 维，query 长度为 $q$，历史 KV 长度为 $n$，形状为：

| 张量 | MHA | GQA | MQA |
| --- | --- | --- | --- |
| Q | $[q,H,d_h]$ | $[q,H,d_h]$ | $[q,H,d_h]$ |
| K / V | $[n,H,d_h]$ | $[n,G,d_h]$ | $[n,1,d_h]$ |
| 头输出 | $[q,H,d_h]$ | $[q,H,d_h]$ | $[q,H,d_h]$ |

于是单请求、$L$ 层 self-attention 的逻辑 KV 存储量为：

$$\mathrm{KVBytes}=2Lnd_hGs,$$

其中 $s$ 为每元素字节数；相对 MHA 的比例为 $G/H$。这是**逻辑缓存量**，还未包括张量并行副本、分页元数据、内存对齐与 allocator 预留。

原文还给出模型扩大的考量：如果头数随模型宽度增长，坚持 MQA 就会越来越激进地压缩表示；用 GQA 可以保持相近压缩比例。另一方面，权重参数与主要 FLOPs 随宽度近似平方增长，KV 规模随宽度近似线性增长，因此不同尺寸模型的收益不会只由头数比决定。

**分片的细节**：当 MQA 的单个 K/V 头在多个 model partitions 中复制时，逻辑压缩并未等比例转化为物理存储节省。GQA 可以让不同分片持有不同 KV 组，减少这种浪费。它依赖具体分片方式，见 Q6。

原文没有把 GQA 用于 encoder self-attention：encoder 的位置可并行处理，逐步反复加载 KV 的问题没有 decoder 那样突出。

> **批注**：模型结构与执行方式要一起评估。一个逻辑上更小的张量，如果为分片而重复存储，物理成本可能与稍大的可分片张量接近。

### 3.3 Experiments

#### 3.1 Experimental setup

**Configurations**：T5.1.1，JAX、Flax、Flaxformer；用 Adafactor 与 T5 的超参数和学习率调度。MQA / GQA 作用于 **decoder self-attention 和 cross-attention**，encoder self-attention 保持原样。后一点决定了本文速度数字包含 cross-attention 优化，不能直接搬到 decoder-only。

**Uptraining**：从公开 T5.1.1 checkpoint 转换，在原预训练数据与设置下继续训练。$\alpha=0.05$ 的实验约消耗 **600 TPUv3 chip-days**。5% 是相对于原预训练的比例；绝对成本仍很高。

**Data**：CNN/Daily Mail、arXiv、PubMed、MediaSum、MultiNews，外加 WMT 2014 English-to-German 与 TriviaQA。作者没有测 GLUE，理由是其关注的多步自回归推理在多数短输出分类任务中不够突出。

**Fine-tuning**：常数学习率 0.001、batch size 128、dropout 0.1；训练到收敛并按 dev 表现选择 checkpoint，使用 greedy decoding。

| 任务组 | 输入长度 | 输出长度 |
| --- | --- | --- |
| CNN/Daily Mail、WMT | 512 | 256 |
| 其他摘要任务 | 2048 | 512 |
| TriviaQA | 2048 | 32 |

**Timing**：xprof 测量，8 个 TPUv4；每模型分别优化并行方式，并使用能装下的最大 batch、上限每 TPU 32。因 GQA 改变内存需求，能用的 batch / 并行方案也可能改变。因此计时是经过配置优化后的系统效果，不能全归给一个 attention kernel。

#### 3.2 Main results

Table 1 与 Figure 3 的核心是两个比较：第一，uptrained MQA-XXL 比 MHA-Large 更快且质量更高；第二，GQA-8 在 MQA 附近的速度上将平均质量拉近 MHA-XXL。

但平均分会遮住细节。下表抽出 Table 1 的几个任务，沿用上面的训练、硬件和解码条件：

| 模型 | CNN/Daily Mail R1 | PubMed R1 | MultiNews R1 | WMT BLEU | TriviaQA F1 |
| --- | --- | --- | --- | --- | --- |
| MHA-XXL | 43.8 | 47.5 | 46.9 | 28.4 | 81.9 |
| MQA-XXL | 43.0 | 46.9 | 46.5 | 28.5 | 81.3 |
| GQA-8-XXL | 43.5 | 47.7 | 47.2 | 28.4 | 81.6 |

GQA 有的任务略高、有的略低。表格支持质量接近，却没有证明每个任务无损，也没有置信区间支持把这些小差异解释成稳定优势。

Figure 3 用质量–时间平面把折中画出来；其横轴单位与 Table 1 不一致，读取绝对数值时采用表格秒单位。速度比例不受这个单位排版冲突影响，但绝对延迟解释会受影响。

> **批注**：ROUGE、BLEU 和 F1 的算术平均可以用于本文内部摘要，却不是一个通用效用单位。“平均只差 0.1”不意味着每项能力都只损失同一比例。

#### 3.3 Ablations

消融只取 CNN/Daily Mail、MultiNews、TriviaQA 三项任务，不能把这些图的纵轴平均分和 Table 1 的七项平均分直接比较。

**Checkpoint conversion（Figure 4）**：T5-Large → MQA、5% uptraining，对照 Mean、First、Random。Mean 最好，First 次之，Random 最差。三者最终都经过训练，结果证明初始化选择会影响迁移质量；它并不是“转换瞬间哪个输出更接近”的直接测量。

**Uptraining steps（Figure 5）**：T5-XXL，比较 MQA 与 GQA-8 随 $\alpha$ 的变化。GQA 转换后已有较好起点，MQA 更依赖 uptraining；两者从 5% 获益，增加到 10% 的边际改善减小。该图不支持在所有模型上完全省去 GQA 的继续训练。

**Number of groups（Figure 6）**：T5-XXL，输入 2048、输出 512，扫描组数与推理时间。从 MQA 的一组增加到 8 组时，时间增长较温和，再靠近 MHA 时上升更明显。作者选择 8 组作为折中。

该图主要扫描**时间**，没有同时完整扫描每个组数的多任务质量与稳定性。因此“8 是经过全面质量–速度搜索的普适最优组数”比证据更强。

### 3.4 Related Work

原文将 MQA 的提出归于 Shazeer（2019），并明确提到 Rabe（2023）独立开发了 GQA 且有公开实现。讨论历史时应保留这一归属，避免把所有 K/V 共享思路都归到本文。

它与 [FlashAttention](../2022-flashattention/README.md) 改变的对象不同：FlashAttention 在给定注意力结构下重新组织计算与 IO；GQA 修改 K/V 参数结构，需要质量评估。量化减小每元素字节数，GQA 减少 KV 头数；二者作用于容量公式的不同因子。

原文也把 speculative sampling 列为相关工作：用一次目标模型的多位置计算来减少串行轮数，和缩小每轮 KV 读取的 GQA 可以组合，但组合速度需独立测量。

### 3.5 Conclusion

结论包含两项不可拆开的贡献：已有 MHA checkpoint 可以经转换与少量 uptraining 适应 MQA；GQA 提供中间组数，在本文实验中取得接近 MHA 的质量与接近 MQA 的速度。

如果只记住“GQA 共享 KV”，会漏掉部署一个既有模型时最需要付出的适应训练成本。

### 3.6 Limitations

作者明确列出三类限制。第一，长输出质量难测，ROUGE 并不能完整表达摘要质量。第二，由于计算资源有限，没有对 XXL GQA 做相同条件的从头训练对照，所以不能比较 uptraining 与从头训练的最终优劣。第三，实验只覆盖 encoder-decoder；作者预期 decoder-only 上 GQA 相对 MQA 可能更有优势，但本文没有相应实证。

这些限制直接影响采用决策：对延迟敏感的任务，既要看缓存量，也要用目标任务的质量指标评估转换后的模型。

### 3.7 Acknowlegements

原文感谢 Google Research 同事的建议和讨论。本节标题沿用原文拼写，不包含新的实验结果。

### 3.8 Appendix A: Training Stability

附录补上正文一笔带过的训练风险。作者从头训练多个 T5-Large MQA，预训练频繁出现 loss spikes，长输入任务 fine-tuning 时立即发散。uptrained MQA 更稳定，但仍有高方差；对不稳定任务报告三次 fine-tuning 的平均表现。

uptrained GQA 在他们的实验中看起来稳定，因此没有继续调查 MQA 不稳定的根因。这里支持的是**有限配置下的观察**，不能推出 MQA 必然不稳定、GQA 必然稳定。

> **批注**：附录改变了阅读 Table 1 的方式。MQA 与 GQA 不仅比较平均质量和时间，还可能有训练方差差异；原文没有统一列出各模型全部任务的多 seed 分布，因而小分差不宜过度解释。

---

## 4. 关键问题解析

### Q1: GQA 到底减少了哪些头？共享 K/V 后，各 query 头还不同吗？

**A**：减少的是 K 和 V 的独立投影头数，query 头数保持 $H$。同组 query 使用相同 K/V，但各自的 query 向量不同，内积与 softmax 权重通常也不同。

例如 $H=8,G=2$，前四个 query 头共享第一组 KV，后四个共享第二组；仍输出 8 个头的结果再拼接。它限制的是每组可访问的键和值表示空间，没有把 8 次 query 注意力直接变成 2 次。

缩放仍按每头维度 $\sqrt{d_h}$，不因组数变成 $\sqrt G$。组数改变共享方式，点积的维度仍由 $d_h$ 决定。

### Q2: 如何估算节省的 KV？为什么这不等于整个模型显存按同倍数减少？

**A**：对 decoder self-attention，逻辑容量为 $2Lnd_hGs$，固定其他条件时从 MHA 到 GQA 缩小 $H/G$ 倍。

**本笔记的容量示例，非论文实验**：32 层、32 query 头、每头 128 维、4096 个缓存 token、每元素 2 bytes、单请求：

| 结构 | KV 头数 | 逻辑 KV 占用 |
| --- | --- | --- |
| MHA | 32 | 2 GiB |
| GQA | 8 | 512 MiB |
| MQA | 1 | 64 MiB |

这里只计算 K/V 激活，不包括权重、临时激活、allocator 预留和并行副本。GQA 缩小 K/V 投影参数，但 Q、输出投影和 FFN 仍然存在，所以模型总显存不会按 4 倍或 32 倍同步缩小。

T5 还缓存 cross-attention 的 K/V，其长度取决于 encoder 输入长度；这一项应与 decoder self-attention 的增长项分别估算。本文改了两种 decoder attention，因此总体收益含两者。

### Q3: mean-pooling 为什么不是一个数学上等价的 checkpoint 转换？

**A**：权重平均可以得到平均的线性投影，但注意力还包含非线性 softmax，通常有：

$$\operatorname{softmax}\!\left(Q\frac{K_1+K_2}{2}^T\right)
\ne\frac{\operatorname{softmax}(QK_1^T)+\operatorname{softmax}(QK_2^T)}{2}.$$

这里省略缩放与 mask 以展示非线性来源。V 也被平均、Q 头仍不同，最终输出自然不保证等价。

平均是一种利用已有参数信息的初始化选择。Figure 4 的“优于 First / Random”是完成 uptraining 后的经验结果，而非证明信息无损保留。真正的迁移需要让网络通过继续训练适应新的约束。

### Q4: “只需 5% 训练”具体是什么？原模型还能直接转换后上线吗？

**A**：原文 $\alpha$ 是**原预训练步数的比例**，沿用原配方和数据；摘要将其概括为原预训练计算量的一小部分。主要实验还在 uptraining 后对每个下游任务 fine-tune。

在原文设置下，5% 约为 600 TPUv3 chip-days，属于相对原预训练较便宜、绝对成本仍显著的适应过程。它不等于 checkpoint 文件离线取个平均就完成了可用模型迁移。

Figure 5 中 GQA 的零 uptraining 起点较好，说明共享较温和；但实际采用仍要测目标任务质量。本文没有给出免训练迁移的普遍保证。

### Q5: 为什么 GQA-8 可以接近 MQA 的速度？8 组岂不是 8 倍 KV？

**A**：逻辑 KV 是 8 倍，但总时间还有权重读取、FFN、Q / 输出投影、attention 算术、通信及固定开销。只有其中部分随 KV 组数近似变化。

当这些其他成本占了相当比例时，KV 从一组到八组不会让总时间增加八倍。原文还指出大模型权重与 FLOPs 增长比 KV 更快，并讨论 MQA 在分片中的复制开销。

Table 1 的 GQA-8 0.28 s 对 MQA 0.24 s 是优化过批量与并行配置后的 TPUv4 结果；Figure 6 的固定输入 / 输出长度扫描则帮助观察组数趋势。这些结果支持“存在温和增加组数的区间”，具体宽度要重新 profiling。

### Q6: GQA 和张量并行怎么互相影响？

**A**：常见按 query 头切分的 TP 希望每个分片本地拥有所需 KV。MQA 只有一个逻辑 KV 头，多个分片可能各存一份；GQA 的多组 KV 则可以随 query 组一起分片。

**示意配置**：8 个 TP 分片，每分片处理一组 query，GQA-8 可以每片持一组独立 KV；若 MQA 在每片复制同一个头，聚合物理 KV 也有 8 份。此时逻辑上的 8:1 缓存差异不会出现在总物理存储量上。

这个例子假设实现使用复制式 MQA 且 GQA 组与分片对齐。其他实现可以通信共享，代价转移到通信；当 $G$ 少于分片数或无法整齐划分时，也可能需要复制。原文给的是这种系统考量，没有规定所有框架采用同一映射。

### Q7: GQA、FlashAttention、PagedAttention 和 Sarathi-Serve 可以叠加吗？

**A**：它们改变不同层面的对象，机制上可以组合：

| 方法 | 改变什么 | 主要验收问题 |
| --- | --- | --- |
| GQA | K/V 参数与共享结构 | 转换和训练后的质量、缓存大小 |
| FlashAttention | 注意力计算的分块、融合与 IO | kernel 支持的形状、数值与性能 |
| PagedAttention | KV 的物理分配、寻址与共享 | 碎片、容量和访问效率 |
| Sarathi-Serve | prefill 分块与每轮调度预算 | TTFT / TBT、容量与流水均匀性 |

GQA 可以让更多请求装下，分页减少浪费，kernel 高效计算，调度器决定这些请求何时运行。组合也会改变瓶颈：缓存不再是首要限制后，调度或通信可能先触顶。

因此不能把各篇论文的加速比相乘预测总收益。需要相同模型、硬件、负载和 SLO 下的组合评测。

### Q8: 为什么将论文归到 Foundations，而不是只归入推理系统？

**A**：核心方法改变了注意力结构与训练流程，它成为后续模型定义的一部分。推理系统读 checkpoint 时必须知道 $H$ 与 $G$，KV 容量、kernel 输入形状和 TP 分片都由这个结构决定。

反过来，读本文时也要带着系统视角：组数选择受部署方式影响，逻辑容量与物理复制要分开算。将其放在 Foundations，可以作为 KV Cache、PD 传输和调度论文共同使用的模型前提。

本文证据仍限定在 T5 / TPU；现代 decoder-only 模型采用 GQA 的事实不应替代原文的实验条件，也不能反向补出原文没有做过的测试。

---

## 5. 可迁移的知识点

- [Grouped-Query Attention](../../../concepts/grouped-query-attention.md) —— 在 query 头数不变时分组共享 KV，区分逻辑压缩、物理分片和质量代价。
- [Multi-Head Attention](../../../concepts/multi-head-attention.md) —— 本文保留 query 的多头分布，放宽每头拥有独立 KV 的约定。
- [KV Cache](../../../concepts/kv-cache.md) —— 容量公式中的 $H_{kv}$ 正是本文优化的变量。
- [张量并行](../../../concepts/tensor-parallelism.md) —— 逻辑头数与分片复制共同决定每设备占用。

**可迁移的设计思路**：在两个极端方案之间寻找有物理意义的可调参数；先用已有权重构造合理初始化，再为结构变化付出适应训练；性能评估时把逻辑数据结构与实际硬件映射分开。它们是从本文方法归纳的工程原则。

---

## 6. 批判与开放问题

### 6.1 作者承认的局限

- 只在 encoder-decoder 上实验；decoder-only 的相对优势是预期，需要单独验证。
- ROUGE 无法完整测量长摘要质量，质量–速度折中存在评估盲点。
- 没有 XXL GQA 从头训练对照，无法判断 uptraining 相对同成本从头训练的上限。
- Appendix A 观察到 MQA 训练不稳定，但没有解释根因，也未系统研究 GQA 稳定性的机制。

### 6.2 我的质疑

1. **硬件和配置影响归因**。时间来自 TPUv4、每模型分别优化并行与 batch，反映最终部署收益；论文没有把省 KV、改变 batch、通信与 kernel 效率的贡献全部拆开。
2. **小质量差异缺少统一不确定度**。附录仅对部分不稳定 MQA 任务三次平均，并非所有模型、所有任务统一多 seed。平均 47.1 与 47.2 的接近不等于统计意义上的等价。
3. **组数选择的证据偏向时间维度**。Figure 6 主要画组数–时间，8 组是作者在该设置下的选择，不能作为所有头数、TP 度和任务的固定默认答案。
4. **计时单位存在排版冲突**。Table 1 是 s，Figure 3 标 ms。相对位置仍可解释，引用绝对时间必须注明采用的来源。
5. **结构性收益不直接等于 SLO 容量收益**。本文没有服务到达过程、排队和尾延迟评测；要用于在线服务，还需走到 [Sarathi-Serve](../../llm-serving/2024-sarathi-serve/README.md) 那种容量口径。

### 6.3 开放问题

- 按头相似性分组能否优于固定分组？收益是否足以抵消转换和分片复杂度？〔本文未实验〕
- 不同层采用不同 KV 组数，能否在同一缓存预算下保留更多质量？〔本文未实验〕
- 当 KV 量化、TP 复制和长上下文同时存在时，最佳组数如何随部署改变？
- 如何用更贴近事实性、长文档一致性的指标验证 uptraining 后的质量？

### 6.4 后续可读

- [Attention Is All You Need](../2017-attention-is-all-you-need/README.md) —— 回看多头独立性的原始定义。
- [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) —— MQA 前作。
- [FlashAttention](../2022-flashattention/README.md) · [vLLM](../../llm-serving/2023-vllm-pagedattention/README.md) —— 与模型侧共享正交的计算和存储优化。
- [DistServe](../../llm-serving/2024-distserve/README.md) —— KV 头数也会影响跨实例传输量，重新审视其 MHA 模型实验的边界。
