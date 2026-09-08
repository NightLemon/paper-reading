---
title: "GPipe / PipeDream：流水并行的调度与更新语义（对比精读）"
authors: ["Yanping Huang", "Youlong Cheng", "Ankur Bapna", "Orhan Firat", "Mia Xu Chen", "Dehao Chen", "HyoukJoong Lee", "Jiquan Ngiam", "Quoc V. Le", "Yonghui Wu", "Zhifeng Chen", "Deepak Narayanan", "Aaron Harlap", "Amar Phanishayee", "Vivek Seshadri", "Nikhil R. Devanur", "Gregory R. Ganger", "Phillip B. Gibbons", "Matei Zaharia"]
affiliation: "GPipe: Google; PipeDream: Microsoft Research / Carnegie Mellon University / Stanford University"
venue: "NeurIPS 2019 / SOSP 2019"
year: 2019
arxiv: "1811.06965v5"
url: "https://papers.nips.cc/paper_files/paper/2019/hash/093f65e080a295f8076b1c5722a46aa2-Abstract.html"
topic: "llm-training"
tags: ["pipeline-parallelism", "gpipe", "pipedream", "microbatch", "1f1b", "weight-stashing", "training-semantics"]
concepts: ["pipeline-parallelism", "activation-checkpointing", "tensor-parallelism", "zero-state-sharding"]
status: "done"
rating: 5
read_date: "2026-09-08"
---

# GPipe / PipeDream：流水并行的调度与更新语义

> **一句话结论**：两者都把层切成 stage、让不同数据批次交错执行；GPipe 用 minibatch 内的梯度累积保留同步更新，PipeDream 用跨 minibatch 的 1F1B 和权重版本管理减少排空，取舍落在气泡、显存、陈旧梯度与达到目标质量的时间上。

---

## 1. 元信息

| 项目 | GPipe | PipeDream |
| --- | --- | --- |
| 论文标题 | 会议登记为 **GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism**；所读 arXiv v5 内文标题为 **GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism** | **PipeDream: Generalized Pipeline Parallelism for DNN Training** |
| 作者 / 机构 | Yanping Huang、Youlong Cheng、Ankur Bapna、Orhan Firat、Mia Xu Chen、Dehao Chen、HyoukJoong Lee、Jiquan Ngiam、Quoc V. Le、Yonghui Wu、Zhifeng Chen；Google | Deepak Narayanan、Aaron Harlap（同等贡献）、Amar Phanishayee、Vivek Seshadri、Nikhil R. Devanur、Gregory R. Ganger、Phillip B. Gibbons、Matei Zaharia；Microsoft Research / CMU / Stanford |
| 发表 / 所读版本 | NeurIPS 2019；本文按 arXiv **1811.06965v5，2019-07-25** 的章节与数据精读 | SOSP 2019；读取 Microsoft Research 官方页面提供的**会议 PDF**，DOI 10.1145/3341301.3359646 |
| 原文 | [PDF](gpipe.pdf) · [HTML](gpipe.html) · [arXiv 固定版本](https://arxiv.org/abs/1811.06965v5) · [会议登记页](https://papers.nips.cc/paper_files/paper/2019/hash/093f65e080a295f8076b1c5722a46aa2-Abstract.html) | [PDF](pipedream.pdf) · [官方论文页](https://www.microsoft.com/en-us/research/publication/pipedream-generalized-pipeline-parallelism-for-dnn-training/) · [DOI](https://doi.org/10.1145/3341301.3359646) |
| 实现背景 | Lingvo 中的 GPipe；[tensorflow/lingvo](https://github.com/tensorflow/lingvo) | 主要基于 PyTorch，论文亦展示 Caffe 接入；[msr-fiddle/pipedream](https://github.com/msr-fiddle/pipedream) |
| 原文格式 | PDF 与 HTML 均为所读 v5，HTML 样式和部分图像需联网 | 本地归档会议 PDF；未用早期 arXiv HTML 冒充会议版 |

**版本说明**：PipeDream 的 arXiv:1806.03377v1 是 2018 年早期稿，标题为 *Fast and Efficient Pipeline Parallel DNN Training*，与本笔记的 SOSP 2019 版不同。GPipe v5 对 PipeDream 的引文指向早期稿；两篇互相评论时要保留时间与版本边界。

本篇按待读队列建议合并精读，分别跟随两篇原文章节，再集中比较；原文以 `gpipe.*`、`pipedream.pdf` 区分，避免覆盖。索引中算作 **一篇对比笔记，覆盖两篇论文**。

前置知识：[张量并行](../../../concepts/tensor-parallelism.md)、同步 SGD、反向传播、[激活检查点](../../../concepts/activation-checkpointing.md)、[ZeRO 状态分片](../../../concepts/zero-state-sharding.md)。

---

## 2. 摘要速览（5 分钟版）

### 2.1 要解决的问题

模型按层分到多张卡后，一个样本要先从首层走到末层，再把梯度逆向传回。如果一次只处理一批数据，许多卡只能等待，显存虽然够了，利用率却很低。

让多批数据同时在流水中可以填补等待，但训练是双向流程：反向要用到前向的激活和权重。持续加入数据、持续更新参数，容易让同一批的前后向用到不同权重；反过来，等所有在飞数据结束再更新，又会产生周期性排空。

### 2.2 核心方法

| 维度 | GPipe | 原始 PipeDream |
| --- | --- | --- |
| 流水里的数据单位 | 一个 minibatch 拆出的多个 microbatches | 多个可独立触发更新的 minibatches |
| 调度 | minibatch 内交错执行，结束时汇总梯度并同步更新 | 启动后在 stage 上交替一前向一反向，即 1F1B |
| 权重版本 | 本 minibatch 的 microbatches 使用同一轮权重 | 前向取当时最新权重并 stash，反向取匹配旧版本 |
| 更新语义 | 满足 batch 可分解等条件时对应原 minibatch 同步梯度 | 梯度带有陈旧性，默认也可能跨 stage 使用不同版本 |
| 主要代价 | 排空气泡、较多在飞 microbatch 激活；用重算减存储 | 版本与激活存储、陈旧梯度、调度和分片实现复杂度 |

**1F1B 只描述执行顺序**。后来也存在使用 1F1B 但在 minibatch 结束才同步更新的方案；不能看见这个缩写就断定优化语义等同原始 PipeDream。

### 2.3 主要结果

| 来源 | 实验条件与基线 | 结果 |
| --- | --- | --- |
| GPipe v5 Table 2 | Transformer-48，TPU，32 个 microbatches；同表 $K=2,4,8$，batch 在必要时为适配显存调整 | 归一化吞吐 **1.8、3.4、6.3**；$K=8$ 相对 $K=2$ 约 **3.5×**，不是固定所有条件的 8 卡对单卡结果 |
| PipeDream Table 1 | VGG-16 / ImageNet，Cluster-A，4 台各 4×V100 16GB、机内 PCIe / 机间 10 Gbps，batch 单元 64；与同规模 DP 比，目标 68% top-1 | `15-1` 配置的 epoch time 与 time-to-target-accuracy 均约 **5.28×** 改善 |
| PipeDream Table 1 | 同 VGG-16 与质量目标，Cluster-B，2 台各 8×V100 16GB、机内 NVLink / 机间 25 Gbps | epoch time 约 **2.98×**，达到目标质量时间约 **2.46×**；两个指标不同 |

这些数字来自不同硬件、模型和更新方式，不能跨两篇直接比较倍率。PipeDream §5.4 的 GPipe 对照是作者自己的实现，并非将 GPipe 的 TPU 原始实验搬到同一表中。

### 2.4 我的评价

评分 ⭐⭐⭐⭐⭐。一起读的最大收益是把三个问题拆开：怎样切层、怎样安排前后向、什么时候以及用哪个版本更新。单独背一个“GPipe 气泡大、PipeDream 气泡小”的结论，会漏掉训练正确性和质量目标。

GPipe 的重点是以简单同步语义扩展模型容量，PipeDream 的重点是结合拓扑、划分与版本管理缩短达到目标质量的时间。适合采用哪种路线，要结合模型是否能装下、可用 batch 和可接受的优化语义决定。

---

## 3. 细读

> 3.1 的七个子节对应 GPipe v5 §1–§7；3.2 的六个子节及致谢对应 PipeDream SOSP 2019 版。引用 Table / Figure 时均显式写论文名；两份所读主文件均没有独立技术附录，GPipe 提及的 supplementary material 不自动算入 v5 的实验细节。

### 3.1 GPipe：按原文章节精读

#### 1 Introduction

引言以大模型在视觉与翻译上的质量收益为动机，希望提供跨架构的模型并行工具。接口要求模型可表达为一串层，将连续层合成 cell，放到不同 accelerator。

每个 minibatch 拆成多个 microbatches，使不同设备同时处理不同 microbatch；全部梯度累积后才更新。这个边界使模型规模变化时不必同时接受跨 minibatch 的旧梯度。

> **批注**：GPipe 的“通用”建立在顺序层表达和单层可执行等条件上。复杂分支、共享权重、跨样本算子会影响划分和微批语义，不能只凭接口简洁推断接入零成本。

#### 2 The GPipe Library

**2.1 Interface**：用户提供层 $f_i$、参数 $w_i$、可选计算成本 $c_i$，以及 partitions 数 $K$ 和 microbatches 数 $M$。连续层组成 cell $F_k=f_j\circ\cdots\circ f_i$，其反向通过自动微分得到；估计成本为层成本之和。

**2.2 Algorithm**：按成本尽量均衡地划分 cells，stage 边界插入激活和梯度通信。设 minibatch 有 $B$ 个样本、拆成 $M$ 份，每份 $b=B/M$；本轮所有 microbatches 在同一套参数下执行，再合并梯度更新。

**梯度等价性的推导**〔本笔记展开〕：若 loss 是各样本 loss 的平均，且前向没有跨 microbatch 的相互作用，则：

$$\nabla\mathcal L_B(w)=\frac1B\sum_{j=1}^{M}\sum_{x\in\mathcal B_j}\nabla\ell(w;x)
=\frac1M\sum_{j=1}^{M}\nabla\mathcal L_{\mathcal B_j}(w).$$

最后一个等式假设各 microbatch 等大；不等大时要按样本数加权。梯度累积的缩放、dropout 随机性、重算副作用都要一致。

**BatchNorm 例外**：原文训练时用 microbatch 的统计量，必要时跨 replica 计算；推理用的 moving statistics 在 minibatch 上累积。它并不等同于未拆分 batch 的训练前向，因此“任意网络拆微批都完全等价”比论文实际机制更强。

**2.3 Performance Optimization**：GPipe 保存 partition 边界激活，反向前重算该 cell 内部前向，减少全部微批内部激活长期驻留。若层数 $L$、大小近似均匀，原文峰值激活量级为：

$$O\!\left(B+\frac{L}{K}\frac{B}{M}\right).$$

第一项对应边界数据，第二项对应一个 microbatch 重算时的局部工作集；式子省略每样本激活尺寸等因子。它没有把参数和优化器状态算进去。

**气泡的简化推导**〔本笔记展开〕：令各 stage 前向耗时 $t_f$、反向 $t_b$，忽略通信和负载不均，则一个 minibatch 约需：

$$T\approx(M+K-1)(t_f+t_b).$$

每个 stage 的有用工作为 $M(t_f+t_b)$，故 walltime 内的气泡比例约为：

$$f_{bubble}\approx\frac{K-1}{M+K-1}.$$

若用有用工作时间作分母，则 overhead 为 $(K-1)/M$，与上式不是同一口径。原文的 $M\ge4K$ 是实验经验，不能解释成代入公式后气泡为零；例如 $K=4,M=16$ 的简化气泡比例仍约 15.8%。原文还指出重算可提前安排，真实执行会偏离最简单的时间线。

> **批注**：增加 $M$ 同时减少 bubble 和减小单微批矩阵乘。最优点取决于计算效率、边界存储和通信粒度，不能仅按气泡公式无限拆小。

#### 3 Performance Analyses

这一节分别测试容量、吞吐和弱互联下的行为，三类实验使用不同设置。

**Table 1 容量**：Transformer 使用 TPUv3 16GB、固定词表 32K、sequence length 1024、batch 32、model width 2048、FFN 8192、32 heads，通过增加层数扩展。128 partitions 可支持表列的 83.9B 参数。这是可运行容量展示，不是 83.9B 完整训练到翻译质量目标的结果。

AmoebaNet 的容量表在 v5 中表头写 NVIDIA GPUs 8GB，正文却写 TPUv2 8GB；部分参数内存数字也不能直接按表述的每参数 12 bytes 对齐。这里保留来源歧义，不把该表变成精确的硬件选型预算。

**Table 2 吞吐**：AmoebaNet-D(18,256) 与 Transformer-48，扫 $K=2,4,8$、$M=1,4,32$。Transformer 一组原始归一化数据为：

| Microbatches $M$ | $K=2$ | $K=4$ | $K=8$ |
| --- | --- | --- | --- |
| 1 | 1.0 | 1.07 | 1.3 |
| 4 | 1.7 | 3.2 | 4.8 |
| 32 | 1.8 | 3.4 | 6.3 |

$M=1$ 时几乎没有可重叠的不同微批，多设备主要提供容量；$M=32$ 时 Transformer 均匀层结构带来较好扩展。表注说明 batch 必要时为适配内存改变，所以这些数值不是严格固定 batch 的普遍强扩展结论。

AmoebaNet 的不同层成本更不均匀，同样 $M=32$ 的 $K=8$ 值仅为 3.48，体现了分区均衡的重要性。

**Table 3 弱互联**：单机多 P100、无 NVLink、走 PCIe，固定 $M=32$。从 2 增至 8 partitions，AmoebaNet-D(18,128) 吞吐约 2.7×，Transformer-24 约 3.3×。它证明所测模型的边界通信可以被有效摊销，不能推出任意网络带宽都不再是瓶颈。

#### 4 Image Classification

GPipe 在 4 partitions 上训练 557M 参数的 AmoebaNet-B(18,512)，输入 $480\times480$，ImageNet-2012 single-crop validation top-1 为 84.4%、top-5 为 97%。这些是模型训练质量，不能视作流水调度算法本身提高了同一模型精度。

迁移实验先用 ImageNet 预训练，再在其他数据集 fine-tune，Table 4 对结果作五次平均；部分比较对象从头训练，部分使用其他预训练数据。采用不同数据路径的分数不能只归因于模型并行机制。

> **批注**：系统让以前装不下的模型成为可训练对象，是贡献本身。对更大模型的质量结果，应保留架构、分辨率和预训练数据这些变化因素。

#### 5 Massive Massively Multilingual Machine Translation

原文章节标题有重复的 Massive，这里沿用以便定位。实验用内部 103 语言语料，将 102 种语言翻译到英语，共约 25B training examples，各语言规模差异很大。

定义 $T(L,H,A)$ 为 $L$ encoder 层加 $L$ decoder 层、FFN hidden 为 $H$、attention heads 为 $A$。因此 $T(64,16384,32)$ 是合计 128 层，并非 64 层总模型；最大质量实验为 6B，在 16 accelerators 上划分。

Figure 3 对照扩大宽度与深度：相近 1.3B 规模下，深模型在低资源语言上的收益更明显；更大模型继续改善高资源语言。该结果混合了多语言迁移和模型容量因素，不能从它推出每个场景都应该优先加深。

训练稳定性需要额外处理：作者缩小 FFN 初始化，并裁剪过大的 logits。GPipe 可以维持同步更新，但“同步”本身不能解决任意深模型的优化问题。

Table 5 在所述德英任务设置下把 batch 从 260K tokens 增到 1M、4M，BLEU 为 30.92、31.86、32.71，NLL 为 2.58、2.51、2.46。它说明这组实验能够从大 batch 获益；不足以证明增加 batch 对所有训练任务都有益。其私有语料和补充设置也限制完整复现。

#### 6 Design Features and Trade-Offs

作者比较按张量维度分片的 SPMD 与层间流水。前者能让单个矩阵更大，但通信更频繁；后者只在 stage 边界交换激活 / 梯度，但原实现要求单层能放到一个 accelerator。

对 PipeDream 的评论集中在陈旧权重和多版本内存。这个观察有助于理解目标差异，却不能替代 2019 会议版的 memory footprint 测量；所读 GPipe v5 引用的是 2018 早期稿。

GPipe 明确承认 BatchNorm 等跨 batch 计算需要特殊策略，划分不均也会限制效率。它选择同步更新与重算，把更多困难留给 batch、边界激活和划分。

#### 7 Conclusion

结论强调效率、灵活性与同步训练行为。应将“几乎线性”限定在适合划分、microbatches 足够、计算通信可以摊销的实验范围内，并区分容量测试与完整模型质量实验。

### 3.2 PipeDream：按原文章节精读

#### 1 INTRODUCTION

PipeDream 从训练通信开销出发，尝试通过跨 minibatch 流水将计算与通信重叠，并允许不同 stages 采用不同 replication factor。

训练双向依赖是其关键挑战：同一个 stage 在某批的前向后可能已更新多次，到其反向时不能随便使用最新权重。于是流水调度与版本保存必须同时设计。

引言将目标定为 **time-to-target-accuracy**，不仅是 samples/s。这比只看设备利用率更适合评价带陈旧性的更新方案，但质量阈值也决定了结果的范围。

#### 2 BACKGROUND AND RELATED WORK

**2.1 Intra-batch Parallelism**：DP、MP、hybrid 将一次迭代分到多个设备。本文对 MP 的用法较宽，包括朴素按层拆分；不能自动等同 [Megatron](../2019-megatron-lm/README.md) 的张量并行。

**2.2 Inter-batch Parallelism**：比较跨 minibatch 流水与 GPipe 的一批内部拆分。前者减少频繁排空，后者保留一次 minibatch 的梯度聚合边界。PipeDream 使用“microbatch”与“minibatch”的语境不同于 GPipe，比较时先确认一次 optimizer update 包含哪些样本。

**2.3 DNN Model and Hardware Diversity**：模型权重大小、边界激活和设备拓扑共同决定好的划分。例如全连接层权重大而输出小，切到独立 stage 可能大幅减少全参数同步；权重小、激活大的卷积网络则未必合适。

> **批注**：流水是否减少通信，取决于“切开后要跨边界传什么”。按层切分不天然优于 DP，后面的 ResNet-50 结果正好提供反例。

#### 3 PIPELINE PARALLELISM

**3.1 Challenge 1: Work Partitioning**。

Profiler 在单 GPU 上运行约 1000 minibatches，记录每层前后向计算时间 $T_l$、输出激活大小 $a_l$、参数大小 $w_l$。优化器结合设备内存和层次化带宽，选择 stage 边界及每个 stage 的副本数。

目标是最小化瓶颈 stage 的有效时间，同时控制边界通信。原文使用分层动态规划；将其核心递推写成统一符号：

$$A_k(i\!:\!j,m)=\min_{s,m'}\max\left\{
A_k(i\!:\!s,m-m'),\;\frac{2a_s}{B_k},\;T_k(s+1\!:\!j,m')\right\}.$$

其中 $s$ 是切分位置，$m'$ 是末段使用的副本数，$B_k$ 是当前拓扑层带宽。三个最大值项分别是前面子流水的瓶颈、边界前后向传输、末 stage 的有效处理时间。$T_k$ 又考虑本层通信与下层最优配置，再除以副本数。

**“最优”限定于模型**：顺序层分区、测得的稳定层耗时、层次化且同层带宽均匀的拓扑、给定通信重叠近似。动态 shape、共享带宽竞争和运行时抖动不一定能由这个代价模型精确表达。

**3.2 Challenge 2: Work Scheduling**。

输入 stage 先注入足够 minibatches 填流水，进入稳态后每个 stage 交替执行一个前向和一个较早批次的反向。1F1B 使两方向都持续推进，不必等待 epoch 的所有前向结束。

原文建议每个 input-stage replica 的最小活跃数为：

$$\mathrm{NOAM}=\left\lceil\frac{\#\mathrm{workers}}{\#\mathrm{input\ stage\ replicas}}\right\rceil.$$

这个 NOAM 是 active minibatches 的缩写，与 Transformer 的 Noam 学习率 schedule 无关。对无复制的 4-stage 流水，该值是 4。

若一个 stage 有多个 replicas，使用由 minibatch ID 决定的 round-robin 路由，即 **1F1B-RR**。同一批的反向必须回到执行其前向的 replica，因为那里保存着对应激活和权重版本。随意把梯度交给当前最空闲的副本，会破坏这份状态对应关系。

**3.3 Challenge 3: Effective Learning**。

**Weight stashing**：每个 stage 前向使用当时最新权重，并保存该版本；对应反向使用同一版本来计算参数梯度及输入梯度。更新产生时应用到当前最新参数。它解决同 stage 前后向不匹配的问题，但不会把旧梯度变成当前参数的梯度。

**Vertical sync**：可选地为 minibatch 指定跨 stages 统一的旧版本，使其前后向都沿同一份历史全模型计算。更新仍应用到更新后的当前参数，所以依旧是 stale gradient。原文默认 **不开 vertical sync**，因为会增加元数据管理。

原文对无复制、$K$ 个 stages 的说明可写为：

$$\begin{aligned}
\text{同步 SGD:}\quad w^{t+1}&=w^t-\eta\nabla f(w_1^t,\ldots,w_K^t),\\
\text{weight stashing:}\quad w^{t+1}&=w^t-\eta\nabla f(w_1^{t-K+1},w_2^{t-K+2},\ldots,w_K^t),\\
\text{再加 vertical sync:}\quad w^{t+1}&=w^t-\eta\nabla f(w_1^{t-K+1},\ldots,w_K^{t-K+1}).
\end{aligned}$$

这里是原文用于说明版本差异的简化时标。第二行是沿混合 stage 版本得到的梯度，第三行是统一旧版本梯度，两者都与第一行有别。

**Memory overhead**：均匀切成 $K$ 段后，每段只含约 $1/K$ 模型，但同时保存多个在飞批次的激活和权重版本。原文用最多 $K$ 份作粗略解释，说明可能接近完整 DP 的每卡占用。这个比较没有证明比只存一份 stage 参数的同步流水更省，也依赖层分布、在飞量和实际工作集。

> **批注**：“numerically correct gradient”在这里首先指前向函数与求导时所用版本匹配。它不是“与同步 SGD 同一条训练轨迹”，也不是所有任务都能保持同样收敛的定理。

#### 4 IMPLEMENTATION

实现约 3000 行 Python，使用 PyTorch 自动微分。Profiler 得到图后，分区器为层标记 stage ID，再生成独立 `torch.nn.Module`；runtime 管理前后向队列、模型版本与通信。

**Parameter State**：保存 stage 参数的多个版本，旧版在相关反向不再需要后才能回收。若 stage 被复制，其 replicas 的更新还要同步。

**Intermediate State**：用 blob ID 关联中间数据；前向激活保留到相应反向完成，反向临时数据用完并完成必要传输后释放。

**Stage Replication**：复用 DDP 支持 stage 级复制。论文当时为 DP 基线用 NCCL，为 pipeline 默认用 Gloo，理由包括小消息表现与不能在 stage 同时使用两个 backend 的实现限制。这个历史选择不能推广成所有现代通信库的普遍排名。

**Checkpointing**：本节指容错训练断点，默认每 epoch 末各 stage 在对应最后 minibatch 反向后保存；与 GPipe 的 activation checkpointing 是不同含义。失败恢复需要各 stage 找到共同成功的断点。

#### 5 EVALUATION

**5.1 Experimental Setup**。

| 集群 | 每服务器 GPU | 显存 | 机内 / 机间 |
| --- | --- | --- | --- |
| A | 4×V100 | 16 GB/GPU | PCIe / 10 Gbps |
| B | 8×V100 | 16 GB/GPU | NVLink / 25 Gbps |
| C | 1×Titan X | 12 GB/GPU | 单 GPU / 40 Gbps |

系统为 Ubuntu 16.04、CUDA 10.0、cuDNN 7.4。任务覆盖 ImageNet 分类、WMT16 En–De（newstest2014 验证）、Penn Treebank 语言模型、MSVD 视频描述；模型包括 VGG、ResNet、AlexNet、GNMT、AWD LM 和 S2VT，主要并非现代 decoder-only Transformer。

原文选取可放入设备的较大 batch 单元：VGG 64、AlexNet 256、ResNet 128、GNMT 64、LM / S2VT 80。DP 的全局 batch 是 worker 数乘该单元，PipeDream 的更新聚合与在飞 minibatches 则采用自己的语义，因此不能假设两者每次更新样本集合完全相同。

**5.2 Comparison to Data Parallelism**。

Table 1 同时报 epoch time 与 TTA，几组最有解释力的数据如下：

| 模型、数据、目标 | 硬件与选择的配置 | Epoch time / TTA 速度比，相对 DP |
| --- | --- | --- |
| VGG-16，ImageNet，68% top-1 | 16 V100，Cluster-A，`15-1` | 5.28× / 5.28× |
| VGG-16，同目标 | 16 V100，Cluster-B，`15-1` | 2.98× / 2.46× |
| ResNet-50，ImageNet，75.9% top-1 | 16 V100，A 或 B，单 stage 复制 16 份 | 1× / 1× |
| GNMT-16，21.8 BLEU | 16 V100，Cluster-A，straight pipeline | 2.34× / 2.92× |
| GNMT-16，同目标 | 16 V100，Cluster-B，straight pipeline | 3.14× / 3.14× |

`15-1` 指第一 stage 复制 15 份、第二 stage 一份，共两段、16 workers；不是“15 路 tensor parallel 加一段”。VGG 将较大权重的全连接部分放到未复制 stage，减少全参数同步。

ResNet-50 被分区器选回普通 DP，是重要的边界：其参数较小、输出激活较大，非 DP 划分反而增加通信。工具能够选择不使用流水，是其目标函数发挥作用的结果。

正文把 Cluster-B VGG 的 2.98× 描述成到目标精度加速，而 Table 1 将其列为 epoch time、TTA 为 2.46×；本笔记以表格区分。AlexNet 使用 synthetic data 只测 epoch time，TTA 标 N/A，不能据此声称获得相应收敛收益。

**5.3 Comparison to Other Intra-batch Parallelism Schemes**。

Figure 14 在 Cluster-A 4 GPUs 上对照朴素模型并行、加流水、再加 stage replication。VGG 和 AlexNet 同时从流水与复制受益，GNMT 无需复制；部分比较保持通信字节相同但通过重叠提高效率。

这些结果说明“安排通信的时机”与“减少通信量”是两条独立路径。其 MP 基线也不能未经说明地替换成 Megatron TP 来解读。

**5.4 Comparison to Inter-batch Parallelism**。

GNMT-16，16 GPUs，Cluster-A / B；使用作者实现的 GPipe，并沿用 PipeDream 分区。GPipe 在飞量设为 NOAM 时，作者报告相对 PipeDream 的 throughput slowdown 为 55% / 71%；增至不 OOM 的最大在飞量后为 35% / 42%。

这些数值说明排空与在飞深度在该实现中的影响。本节没有完整列出与原 GPipe 相同的重算、microbatch、batch 和所有调优设置，也没有提供相同更新语义下的公平隔离对照，因此不能从这组百分比宣布 PipeDream 在全部模型上优于 GPipe。

**5.5 Microbenchmarks**。

- **Optimizer**：Figure 15 的 VGG-16 / 16 workers 中，预测吞吐与真实吞吐相关，分区器选中测试配置中的最佳点；“最优”仍限定代价模型和候选空间。
- **Memory Footprint**：Figure 16 比较四阶段流水与 DP 的每 stage 占用，支持所测模型中 stash 开销仍可接受；它没有消除多版本存储本身。
- **Communication Overhead**：Figure 17 用每训练样本通信 bytes 比较，ResNet-50 的非 DP 方案更多，再次解释为什么被选回 DP。
- **Effect of Pipeline Depth**：Figure 18 在 GNMT-8 / Cluster-A 四张 V100 上，活跃量从 2 到 4、7 时吞吐改善趋于饱和，内存继续增长。这里“pipeline depth”指在飞 minibatches，需与 stage 数区分。

> **批注**：主表、通信消融和内存消融相互支撑，形成了比单个峰值倍率更有用的解释；但稳态利用率和 TTA 仍依赖模型、目标阈值和陈旧性可接受程度。

#### 6 CONCLUSION

结论将贡献概括为自动划分、跨 minibatch 流水、计算通信重叠和有效学习。最高约 5.3× 是特定任务、集群、质量阈值的 TTA 结果；默认语义并不回到同步 SGD。

**ACKNOWLEDGMENTS**：项目属于 Microsoft Research 的 Project Fiddle，并有 CMU、Stanford 等协作。致谢不增加新的收敛或性能证据。

### 3.3 跨论文证据对照

| 问题 | GPipe 已给出的证据 | PipeDream 已给出的证据 | 仍不能直接推出什么 |
| --- | --- | --- | --- |
| 容量 | 重算、分区、微批支持更大模型 | 多版本流水在所测模型的内存占用可接受 | 两者对任意大单层都可运行 |
| 利用率 | $K,M$ 扫描与不同架构的吞吐 | 1F1B、复制 stage、深度消融 | 消除所有运行时气泡 |
| 学习 | 同步聚合规则及大模型任务结果 | 指定目标精度的 TTA、部分 epoch 曲线 | 二者具有完全相同更新轨迹 |
| 网络 | P100 / PCIe 实验 | 多集群与每样本通信量分析 | 流水永远优于 DP / TP |
| 相互比较 | v5 对早期 PipeDream 的定性讨论 | SOSP 版对自实现 GPipe 的 GNMT 实验 | 统一配置下的普遍排名 |

---

## 4. 关键问题解析

### Q1: microbatch、minibatch、gradient accumulation 和在飞深度分别是什么？

**A**：在 GPipe 里，一个 minibatch 定义一次同步更新的数据集合，拆出的 microbatches 是其执行片段。梯度累积把这些片段的梯度合成一次更新；在飞深度是同一时刻尚未完成前后向的数据片段数。

PipeDream 原文的在飞单位通常是不同 minibatches，每批反向完成会触发相应 stage 的更新。它们并不是 GPipe 那种必须一起累积后才更新的同一大 batch。

因此比较时要写出一次 optimizer update 对应的样本数，以及未完成的批次有多少。只给一个 “batch size=64” 无法确定训练语义。

### Q2: GPipe 的气泡怎样估算？M 越大是否总越好？

**A**：在均匀 stage、忽略通信等简化条件下，$K$ 段、$M$ 微批的同步流水，总时间约 $(M+K-1)(t_f+t_b)$，气泡占比约 $(K-1)/(M+K-1)$。例如 $K=4,M=16$ 约 15.8%。

固定 minibatch 大小 $B$ 时，增大 $M$ 会减小 $b=B/M$，可能降低 GEMM 效率并增加小消息和调度开销；固定 microbatch 大小时，增大 $M$ 又增大全局 batch，改变优化条件。

原文 $M\ge4K$ 是其实现和实验的经验范围。真正要优化的是容量、吞吐和学习效果的组合，而不是单独把气泡公式压到零。

### Q3: weight stashing、vertical sync 和普通同步更新差在哪？

**A**：weight stashing 解决**同 stage** 的前后向版本一致；vertical sync 再约束**同 minibatch 跨 stages** 的版本一致；同步更新还要求梯度是在当前要更新的那轮参数下，由规定的完整 minibatch 计算出来。

前两项都可能使用旧版本。即使加 vertical sync 得到 $\nabla f(w^{t-d})$，更新仍是作用于当前 $w^t$，与 $w^t-\eta\nabla f(w^t)$ 不同。

所以“gradient 是某个一致前向函数的导数”是正确性的一层，“与目标优化算法完全相同”是另一层。PipeDream 默认只启用 weight stashing，必须依靠目标任务的学习结果验证其效果。

### Q4: 为什么不能在前向用旧权重，反向直接用最新权重？

**A**：反向是在对前向实际执行的函数求导。以两层 $y=W_2\sigma(W_1x)$ 为例，传回输入层的梯度要用前向所对应的 $W_2$ 和激活。如果换成另一个 $W_2$，这条链式求导不再对应原前向计算。

stash 保存匹配版本，使激活与导数在同一函数上计算；随后把旧梯度应用到新参数则成为显式的 stale-gradient 更新。这是一个可以研究收敛的算法变化，比在同次求导中混用不匹配状态更可控。

对 dropout 和有状态层，还需要保存或恢复相应状态。仅存一份权重版本并不能自动处理所有重算副作用。

### Q5: 哪个方案更省显存？为什么两篇会给人不同印象？

**A**：需要分开计权重与优化器、未完成微批的边界激活、当前重算工作集、权重版本和通信缓冲。GPipe 主要靠重算减少多个 microbatches 的内部激活；PipeDream 限制在飞量，但保存匹配的多版本参数和激活。

PipeDream 的“每卡接近 DP”以完整模型复制式 DP 为分母。它不等于比一个 stage 只保存一份参数的 GPipe 更小。GPipe 的最大模型展示也不能直接拿来反驳 PipeDream 在其所测网络上 stash 开销可接受。

**教学预算**：若某 stage 参数为 $W$ bytes，需同时保留 $v$ 个不同版本，那么单是这些版本就可能接近 $vW$；再加在飞批次激活。此式未算版本复用和优化器实现，适合提醒成本来源，不是测量值。

### Q6: 1F1B 为什么不等于异步训练？怎样与 ZeRO、Megatron 组合？

**A**：1F1B 只说一个 stage 交替做前向和反向。可以像原 PipeDream 一样跨批更新、保存旧版本，也可以在一个同步 minibatch 内按 1F1B 排程，直到全部梯度累积完才更新。要确定语义，还必须问 update barrier 放在哪里。

Megatron TP 切 stage 内部矩阵，ZeRO 在相应 DP 组分片状态，PP 切连续层。对常见规则的三维同步组合，总设备数为 $N_{DP}N_{TP}N_{PP}$，全局 batch 为本地 microbatch × 累积微批数 × DP 度。

PipeDream 允许每 stage 不同 replication factor，并不是上述均匀三维拓扑的直接同义词。把多版本异步流水与 ZeRO 组合还需要处理 owner 状态、参数版本和 gather 的一致性，不能只把配置数字相乘。

### Q7: 为什么 PipeDream 对 VGG 有收益，对 ResNet-50 却退回数据并行？

**A**：VGG 的全连接部分参数多、边界表示相对小，把它放到独立未复制 stage，可以减少大量全参数同步。ResNet-50 的权重较紧凑而中间激活较大，按层切分后边界通信可能比 DP 梯度同步更贵。

这正是 PipeDream Figure 17 的每样本通信量对照。优化器选择普通 DP，并不代表工具失败，而是说明在那个模型和硬件上流水不是有利选项。

因此阅读模型并行论文时，应分别看参数 bytes、边界 activation bytes、计算时间和拓扑。层数本身不足以预测哪种并行最好。

### Q8: 应该用吞吐、epoch time 还是 time-to-quality 比较两者？

**A**：吞吐解释系统资源利用，epoch time 进一步反映处理相同数据量的时间，time-to-quality 则包含达到目标所需的数据遍历次数。对更新语义或 batch 不同的方案，最后一项尤其重要。

PipeDream Table 1 的 Cluster-B VGG 就是例子：epoch time 改善 2.98×，TTA 仅 2.46×。若只报前者，会漏掉学习进度差异。TTA 也应注明质量阈值、验证方式和未达到目标的运行。

公平复现应固定或明确记录模型、数据、硬件总量、精度、优化器、有效 batch、重算设置、划分和目标质量，并分别报告容量、稳态吞吐与达到目标的时间。本次笔记做原文核对，未执行模型训练实验。

---

## 5. 可迁移的知识点

- [流水并行](../../../concepts/pipeline-parallelism.md) —— 将 stage 划分、前后向顺序、梯度累积边界与权重版本分开设计。
- [激活检查点](../../../concepts/activation-checkpointing.md) —— GPipe 用重算控制多微批的激活占用；注意与容错 checkpoint 的含义不同。
- [张量并行](../../../concepts/tensor-parallelism.md) —— 单层内部的切分，补足纯层间流水的单层容量限制。
- [ZeRO 状态分片](../../../concepts/zero-state-sharding.md) —— 本仓库的组合分析：分片持久状态，与流水的在飞激活和权重版本共同决定峰值；两篇原文未做这种组合评测。

**可迁移的方法**：用时间线解释依赖、用版本号解释梯度、用分项账本解释显存，最后用统一质量目标评价系统。单个设备利用率或最高 speedup 不足以替代这四项。

---

## 6. 批判与开放问题

### 6.1 作者承认的边界

- GPipe 原始实现要求单层可放入设备，BatchNorm 等跨 batch 操作需要特殊处理，划分不均会降低效率。
- GPipe 的更深模型还需初始化和 logits 处理来稳定训练，同步更新不保证任意深模型都容易优化。
- PipeDream 默认有梯度陈旧性，weight stashing 不保证跨 stage 版本一致；vertical sync 也不等价于当前版本同步更新。
- PipeDream 的分区器在给定层次拓扑和成本模型下优化，过多在飞 minibatches 会继续占用内存。

### 6.2 我的质疑

1. **两篇优化目标不同**。GPipe 的主张偏大模型容量与同步语义，PipeDream 的主表强调 TTA；直接比较摘要倍率容易错位。
2. **GPipe v5 的部分表述需要对账**。表头和正文的 AmoebaNet 硬件不一致，吞吐表允许改变 batch，部分 supplementary 设置未在该文件中给出。
3. **PipeDream 的收敛证据是有范围的**。主要是 CNN / LSTM 等模型、给定精度阈值，不能自动推广到大规模 Adam 训练的 decoder-only Transformer。
4. **直接比较的实现细节有限**。PipeDream §5.4 使用自己的 GPipe 实现和相同划分，结果受在飞量和重算等设置影响，缺乏完整统一配置表。
5. **稳态无气泡依赖理想条件**。分区均衡、输入计算时间稳定、通信能被覆盖时收益最好；负载波动和共享网络竞争可能重新出现等待。
6. **参数和图表单位必须区分**。stage 数、stage replicas、microbatch 数与“pipeline depth”在两篇中含义不同；未解释这些参数就复述结论会制造错误直觉。

### 6.3 开放问题

- 同步 1F1B 如何在保留 minibatch 更新语义的同时降低激活峰值？
- 划分器能否同时优化 time-to-quality、峰值显存、重算和网络干扰，而不只优化稳态瓶颈时间？
- ZeRO 参数物化与流水多版本怎样共存，如何限制瞬时内存峰值？
- 动态序列长度、MoE 和跨 stage 共享权重会怎样改变划分与调度的可行空间？

### 6.4 后续可读

- [Megatron-LM](../2019-megatron-lm/README.md) · [ZeRO](../2020-zero/README.md) —— 将层内计算、模型状态与层间执行的三条线串起来。
- [Efficient Large-Scale Language Model Training on GPU Clusters](https://arxiv.org/abs/2104.04473) —— 3D 并行、同步流水与交错调度的后续分析。
- [Memory-Efficient Pipeline-Parallel DNN Training（PipeDream-2BW）](https://arxiv.org/abs/2006.09503) —— 研究梯度累积与权重版本内存的后继工作，不能回填成 2019 版已经具有的行为。
- [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198) —— 接续激活复制与序列并行的方向。
- [Sarathi-Serve](../../llm-serving/2024-sarathi-serve/README.md) —— 推理没有训练的参数更新，却仍会因为微批运行时间差异出现流水气泡。
