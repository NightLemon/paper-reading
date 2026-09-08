---
title: "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"
authors: ["Samyam Rajbhandari", "Jeff Rasley", "Olatunji Ruwase", "Yuxiong He"]
affiliation: "Microsoft"
venue: "SC 2020"
year: 2020
arxiv: "1910.02054v3"
url: "https://arxiv.org/abs/1910.02054v3"
topic: "llm-training"
tags: ["zero", "deepspeed", "data-parallelism", "optimizer-state", "memory", "activation-checkpointing"]
concepts: ["zero-state-sharding", "activation-checkpointing", "tensor-parallelism"]
status: "done"
rating: 5
read_date: "2026-09-08"
---

# ZeRO: Memory Optimizations Toward Training Trillion Parameter Models

> **一句话结论**：把数据并行中重复存储的优化器状态、梯度和参数逐级分片，在需要时通信恢复计算所需数据，再处理激活、临时缓冲与碎片，使集群显存能够支撑更大的模型和更高效的 batch。

---

## 1. 元信息

| 项目 | 内容 |
| --- | --- |
| 作者 / 机构 | Samyam Rajbhandari、Jeff Rasley（同等贡献）、Olatunji Ruwase、Yuxiong He；Microsoft |
| 发表 | SC 2020；首次预印本 2019，按会议年份归档；本文读取 arXiv v3（2020-05-13） |
| 原文 | [PDF](paper.pdf) · [HTML](paper.html) · [固定版本页面](https://arxiv.org/abs/1910.02054v3) |
| 阅读说明 | HTML 适合按章节阅读，PDF 第 23–24 页还包含 HTML 未呈现的实验配置附表；HTML 的样式和部分图像需联网加载 |
| 代码 | [DeepSpeed](https://github.com/microsoft/DeepSpeed)；本笔记分析论文当时的 ZeRO-100B，不将后来的实现功能自动算入原始实验 |
| 前置知识 | 数据并行、Adam、混合精度、[张量并行](../../../concepts/tensor-parallelism.md)、[激活检查点](../../../concepts/activation-checkpointing.md) |

**三个边界先记住**：论文里的 MP 主要指 Megatron 式张量并行；三个 ZeRO-DP 分片阶段是累计启用的；实际评测的 **ZeRO-100B = 优化器与梯度分片 + ZeRO-R**，完整参数分片和万亿参数主要属于理论容量分析。

---

## 2. 摘要速览（5 分钟版）

### 2.1 要解决的问题

模型有多少参数，并不能直接告诉我们训练需要多少显存。在论文采用的混合精度 Adam 配置中，除 FP16 参数和梯度外，还要保存 FP32 主参数、一阶矩和二阶矩。每参数的持久模型状态合计 16 bytes，外加激活、通信缓冲和分配器开销。

传统数据并行在每个 rank 上都保存整套状态，加卡不能减少单卡存储需求；张量并行虽然切分了状态，却把层内计算也切细，需要频繁通信。ZeRO 寻找的是保留数据并行计算粒度、同时去掉存储副本的办法。

### 2.2 核心方法

**ZeRO-DP** 逐级改变数据并行组内的状态归属：先分优化器状态，再分梯度，最后分参数。每个 rank 负责更新自己的参数片段，计算时按需获得所需参数，反向时把梯度归约给对应 owner。

**ZeRO-R** 处理剩余显存：分片保存张量并行组内重复的 activation checkpoints、控制融合缓冲大小、将长寿命对象放进预分配连续空间。特别大的模型还可选择把 activation checkpoints 放到 CPU。

```text
传统 DP：每个 rank 都存 参数 + 梯度 + Adam 全部状态
ZeRO-1：每个 rank 仅持有一部分 Adam 状态，负责对应参数更新
ZeRO-2：梯度也只在负责更新的 rank 保留归约结果
ZeRO-3：参数平时也分片，逐计算单元取回，使用后释放副本

另外处理：activation checkpoints / 临时通信缓冲 / 显存碎片
```

这里“释放副本”指不再保留额外的完整参数副本，owner 分片仍在；计算当前层时还需要临时物化数据。

### 2.3 主要结果

| 证据 | 条件与比较对象 | 结果与边界 |
| --- | --- | --- |
| Figure 1、§5 | **理论模型状态账本**：7.5B 参数，64 路 DP，FP16 参数 / 梯度、FP32 Adam 主参数与两矩，十进制 GB | 标准 DP 120 GB/设备；分优化器约 **31.4 GB**，再分梯度约 **16.6 GB**，完整分片约 **1.875 GB**；不含激活和瞬时峰值 |
| Figure 2、§10.2、PDF Table 5 | GPT-2 类模型，ZeRO + Megatron MP，25 台 DGX-2 共 400×V100 32GB、节点间聚合 800 Gbps；100B 配置为 MP16、本地 batch 32 | 作者报告约 **38 TFLOP/s/GPU、聚合 15 PFLOP/s**；基线 MP128、本地 batch 2、384 GPU，配置并非只差一个 ZeRO 开关 |
| Figure 4、§10.4、PDF Table 10 | 128×V100，纯 ZeRO 数据并行，无 MP / PP；13B、62 层、hidden 4096、本地 batch 2、全局 batch 256；与 PyTorch DDP 比较 | 能运行 **13B** 模型；DDP 最大约 1.4B。正文“平均超过 40 TFLOP/s/GPU”是该组模型的概括，不特指 13B 那一个点 |

论文称可运行到 170B，指相应配置的训练执行与性能测试；明确展示端到端训练的模型是 17B Turing-NLG。标题里的 **1T 并非本篇完成的端到端训练实验**。

### 2.4 我的评价

评分 ⭐⭐⭐⭐⭐。ZeRO 的核心是区分**计算上的数据并行**与**存储上必须完整复制**这两个假设。更新一个参数只需要其梯度和优化器状态，没有必要让每个 rank 都执行并保存同一份更新。

与 [Megatron-LM](../2019-megatron-lm/README.md) 连读，可以看到训练显存优化的两种互补方式：张量并行切算子，ZeRO 切状态归属。本文的容量公式很有解释力；性能结论则必须结合 batch、模型形状和通信拓扑阅读。

---

## 3. 细读

> 3.1–3.11 对应原文 §1–§11；3.12 为 Acknowledgement；3.13 核对 PDF 末尾未编号的实验配置附表。原文存在部分图号引用错位，下文用图题、实验内容和附表号共同定位。

### 3.1 Extended Introduction

引言先比较 DP 与 MP 的两类限制，再将显存分成 **model states** 与 **residual states**。模型状态包括参数、梯度和优化器；残余包括激活、临时缓冲和碎片。这个分类直接决定后面的两套优化。

Figure 1 是容量分析，Figure 2–3 是实现评测，三者的证据性质不同。引言明确说 ZeRO-100B 实现的是 $P_{os+g}$ 与 ZeRO-R，后面才计划支持完整参数分片。因此不能用 Figure 2 的吞吐图证明 stage 3 已经达到同样效率。

另一个关键判断是：全部参数虽然贯穿训练周期，但每层参数仅在对应计算发生时需要。利用这一生命周期，可以用通信替代一直存在的副本。

> **批注**：论文一开始就把可实现的状态分片设计与当时实际完成的实现分开。阅读后来广泛使用的技术时，更需要遵守这个版本边界。

### 3.2 Related Work

#### 2.1 Data, Model and Pipeline Parallelism

DP 切样本，MP 切层内计算，PP 切连续层。ZeRO 指出 GPipe 用较多 microbatches 摊薄气泡，PipeDream 用旧权重版本维持流水；它们分别带来 batch / 激活需求或优化语义方面的取舍。

这些是作者的机制比较，本篇没有把 GPipe、PipeDream 与 ZeRO 放进统一硬件、模型和质量目标下直接排名。张量并行、流水并行和状态分片可以组合，各自负责不同资源维度。

#### 2.2 Non-parallelism based approach to reduce memory

**2.2.1 Reducing Activation Memory**：压缩、checkpointing、生命周期分析与状态分片互补。ZeRO-R 进一步分片保存 checkpoint，不能把它与“不存激活、反向重算”的基本机制混为一谈。

**2.2.2 CPU Offload**：本篇主要通过设备间去冗余降低模型状态存储；可选 CPU offload 针对 activation checkpoints。后来的 ZeRO-Offload / ZeRO-Infinity 并不是本篇已经实现并评测的同一功能集合。

**2.2.3 Memory Efficient Optimizer**：Adafactor 等通过改变统计量结构节省状态，可能改变优化算法；ZeRO 保持每个参数原本的优化器状态与更新规则，主要改变这些数据存在哪里、由谁更新。

#### 2.3 Training Optimizers

Adam 的逐参数一阶、二阶统计量带来额外存储，也提供很直接的分片机会。只要相应梯度正确归约到 owner，owner 可以执行与复制式实现相同的逐参数更新。跨参数操作则仍需适当通信，见 Q4。

### 3.3 Where Did All the Memory Go?

#### 3.1 Model States: Optimizer States, Gradients and Parameters

设参数个数为 $\Psi$，沿用原文混合精度 Adam 账本：

| 状态 | 精度 | 字节数 |
| --- | --- | --- |
| 用于前后向的参数 | FP16 | $2\Psi$ |
| 参数梯度 | FP16 | $2\Psi$ |
| 主参数 | FP32 | $4\Psi$ |
| Adam 一阶矩 | FP32 | $4\Psi$ |
| Adam 二阶矩 | FP32 | $4\Psi$ |
| 合计 | | $16\Psi$ |

原文把主参数和两矩一起记作优化器状态乘数 $K=12$。1.5B 参数仅 FP16 权重需 3 GB，而这套模型状态已需 24 GB，尚未计算激活。

**适用边界**：16 bytes/参数由这套精度和优化器决定。若梯度保留 FP32、使用不同优化器或其他主参数策略，系数会变；不能把 16 当作所有训练系统的固定物理常数。

#### 3.2 Residual Memory Consumption

训练反向还要用到前向保存的信息。checkpointing 减少保存量，但大模型的 checkpoints 本身仍可能很大。通信或梯度范数计算也常将张量融合成大 buffer；若每次额外分配一个全模型 FP32 buffer，就增加 $4\Psi$ bytes。

碎片是另一个层次：总空闲量足够，不等于存在满足请求的连续空间。论文观察到极端情况下仍有相当空闲显存却分配失败。这个现象解释 ZeRO-R 的动机，不能仅用持久模型状态公式预测 OOM。

> **批注**：训练容量至少要分三层计算：长期存储、当前算子的临时工作集、分配器可实际使用的空间。ZeRO 改善第一层后，后两层会更显眼。

### 3.4 ZeRO: Insights and Overview

#### 4.1 Insights and Overview: ZeRO-DP

三个观察串起设计：DP 的算子粒度较大、易扩展；DP 的状态复制浪费显存；许多状态只在特定时刻需要。于是让每个 rank 仍处理自己的完整样本前后向，但只持有和更新部分持久状态。

相较张量并行，这保留了更大的局部矩阵乘；相较传统 DP，它打破了“每设备要放下整套训练状态”的容量限制。不过 stage 3 在计算时仍需要物化当前计算单元的参数，不能由此认为任意大的单层都能无条件运行。

#### 4.2 Insights and Overview: ZeRO-R

**4.2.1 Reducing Activation Memory**：Megatron 式 MP 的某些输入激活在组内重复，保存 checkpoint 时只存分片，在重算前 all-gather。

**4.2.2 Managing Temporary buffers**：使用足以保持通信效率、但不随整个模型无限增长的缓冲上限。

**4.2.3 Managing fragmented Memory**：按生命周期将 checkpoints、梯度等长寿命数据移入预分配连续空间，减少它们与短寿命临时激活交错造成的碎片。

### 3.5 Deep Dive into ZeRO-DP

#### 5.1 P_os: Optimizer State Partitioning

将优化器状态分成 $N_d$ 份，每个数据并行 rank 负责一份。对应梯度归约好后，它更新这一片参数，再通过参数 all-gather 使下一步所需参数可用。

持久模型状态为：

$$M_1=4\Psi+\frac{12\Psi}{N_d}.$$

参数与梯度仍按完整规模预算；仅主参数和 Adam 两矩的长期保存被分摊。$N_d$ 很大时趋向 $4\Psi$，相对标准 DP 的 $16\Psi$ 最多接近 4 倍节省，不能随着卡数无限降低。

#### 5.2 P_g: Gradient Partitioning

每个 owner 只需要负责参数的归约梯度，因此按 bucket 将梯度 reduce 到对应 rank，其他副本及时释放。整体相当于 reduce-scatter。

累计启用优化器与梯度分片：

$$M_2=2\Psi+\frac{14\Psi}{N_d}.$$

未分片的 FP16 参数 $2\Psi$ 成为下限。bucketization 将通信做大以提升带宽利用率，也便于与反向计算重叠；bucket 本身仍是额外工作空间。

#### 5.3 P_p: Parameter Partitioning

参数也长期分片，每个 rank 在执行相应前后向前按需取得完整计算单元的参数，用完可以释放非 owner 副本。原文用分片广播说明机制，并将其视为沿执行过程展开的 all-gather。

长期模型状态理想值：

$$M_3=\frac{16\Psi}{N_d}.$$

与前两阶段不同，它没有一个完整参数副本造成的固定下限。但实际峰值至少还包含正在 materialize 的参数、激活和通信缓冲。这个公式描述状态去冗余收益，不能独自证明完整训练可行。

#### 5.4 Implication on Model Size

将 $\Psi=7.5\times10^9,N_d=64$ 代入，得到 120、31.40625、16.640625、1.875 GB，分别对应 baseline、stage 1、2、3。Table 1 的大规模配置进一步展示容量增长。

对 1T 参数、1024 路 DP，纯模型状态为 $16\times10^{12}/1024\approx15.625$ GB/设备。它为“容量上有可能”提供依据，完整峰值、通信和训练总时长仍需另算。

> **批注**：数字里使用十进制 GB，与 GiB 有区别。图里的约 1.9 GB 是长期模型状态；把它写成“7.5B 模型只需 1.9 GB 就能训练”会遗漏最重要的峰值条件。

### 3.6 Deep Dive into ZeRO-R

#### 6.1 P_a: Partitioned Activation Checkpointing

在 MP 组内，每个 rank 保存某个 checkpoint 的不同片段，反向需要重算时 all-gather 恢复输入。checkpointing 决定哪些中间数据可以不存，$P_a$ 则进一步减少仍需保存的 checkpoints 的重复副本。

原文估算：Table 4 的 100B 配置、sequence length 1024、batch 32、MP16，每层保存一个输入 checkpoint 时约 33 GB/设备，分片后约 2 GB。这是指定结构与保存策略下的激活账本，不是所有 activation tensor 的瞬时峰值。

$P_{a+cpu}$ 可把这部分保存量移到 CPU。数据返回设备时仍需缓冲和传输，所谓“接近零”的是设备上长期驻留的 checkpoints，而非整个反向阶段零激活显存。

#### 6.2 CB: Constant Size Buffers

融合许多小通信能提高带宽，但若 buffer 与模型一样大，会侵占可训练规模。ZeRO-R 选择一个较大的固定容量 buffer 分批处理。“constant”是对模型规模而言，不是说每次通信必须等长，更不是取消通信。

#### 6.3 MD: Memory Defragmentation

前向中的 checkpoints 长寿命、其他激活短寿命；反向中参数梯度较长寿命、许多 activation gradients 短寿命。按这种生命周期把长寿命对象放到预分配连续区，减少交错分配造成的洞。

这与 [vLLM](../../llm-serving/2023-vllm-pagedattention/README.md) 都关注显存浪费，但具体机制不同：这里组织和搬移训练对象，PagedAttention 为 KV 提供分页寻址；不能把 ZeRO-R 简称为训练版 PagedAttention。

### 3.7 Communication Analysis of ZeRO-DP

#### 7.1 Data Parallel Communication Volume

原文将大消息 all-reduce 拆成 reduce-scatter + all-gather，各约移动 $\Psi$ 个元素，总计约 $2\Psi$。这是忽略有限 rank 修正的每 rank 通信量口径。

**本笔记补全计数约定**：对 ring，每 rank 的发送量更精确为 $2(N_d-1)\Psi/N_d$ 个元素；接收量同量级。若同时累加发送与接收，会再多一个 2；比较基线和 ZeRO 时必须统一口径与元素字节数。

#### 7.2 ZeRO-DP Communication Volume

**7.2.1 Communication Volume with P_os+g**：标准 DP 的第二半步在每个 rank 收齐所有梯度；ZeRO 只让 owner 得到所需梯度，局部更新后再收齐**更新后的参数**。reduce-scatter 梯度约 $\Psi$，all-gather 参数约 $\Psi$，仍约 $2\Psi$。

“通信量相同”依赖这套重新组织的通信流程。若在原本完整梯度 all-reduce 之后额外加一轮参数 all-gather，就不是原文分析的最低通信流程。

**7.2.2 Communication Volume with P_os+g+p**：前向逐段收集参数一次，反向再收集一次，加上梯度 reduce-scatter，一步约 $3\Psi$，是基线的 1.5 倍。

| 流程 | 主要数据移动 | 论文近似量 |
| --- | --- | --- |
| 标准 DP | 梯度 reduce-scatter + 梯度 all-gather | $2\Psi$ |
| ZeRO 优化器 / 梯度分片流程 | 梯度 reduce-scatter + 更新参数 all-gather | $2\Psi$ |
| 完整状态分片 | 前向参数 gather + 反向参数 gather + 梯度 reduce-scatter | $3\Psi$ |

这个量没有刻画通信次数、调度延迟、重叠程度、反复重算或预取策略。论文并未证明 stage 3 的训练时间只增加 50%；它分析的是特定执行计划中的通信量。

### 3.8 Communication Analysis of ZeRO-R

以 Megatron 的 checkpointing 配置分析：每个 Transformer block 包括原前向、重算前向、反向，共六次 all-reduce；令每次消息大小 $A=bsh$，按论文口径总量约 $12A$。分片 checkpoint 额外增加一次约 $A$ 的 all-gather，因此相对该 MP 通信账本增加约 $1/12$。

原文部分简式省略 batch 因子，下文恢复 $b$ 以保持量纲一致。“少于 10%”针对这套计算和 checkpoint 策略，不是任意模型的通用额外通信率。

作者还说更大 batch 可减少 DP 通信。准确理解是：每步参数梯度通信仍取决于参数规模；对固定训练样本数，更大 batch 减少更新步数，或降低每样本分摊的通信。代价是全局 batch 变化可能影响优化。

### 3.9 Step Towards 1 Trillion Parameters

本节把容量与时间分开讨论。全分片可以让模型状态分摊到大量设备；与 MP 组合还可以约束每个数据并行组的样本 batch。

但参数装下之后，万亿模型的前后向仍需大量计算。作者通过相对 BERT-Large 的估算说明训练可能持续数月乃至更久，并未声称在已有集群完成了 1T 训练。

> **批注**：系统工作可以解决可行性的一个必要条件而尚未解决全部条件。本文最可靠的结论是扩大了可运行模型空间；最终的训练预算、数据和优化问题仍然存在。

### 3.10 Implementation and Evaluation

#### 10.1 Implementation and Methodology

实现为 PyTorch 上的 ZeRO-100B，包含 $P_{os+g}$ 和 ZeRO-R，可与 `torch.nn.Module` 及 Megatron MP 结合。测试平台是 25 台 DGX-2、400×V100 32GB、节点间聚合 800 Gbps。无 MP 时基线为 PyTorch DDP，有 MP 时基线为 2019-09 的开源 Megatron-LM。

不同实验使用不同模型层数、hidden size、头数和 batch。正文 Table 4 给概览，PDF 附表 5–10 给更细参数；两处个别数值与图号有冲突，应优先记录具体实验行，并保留不一致。

#### 10.2 Speed and Model Size

Figure 2 展示各模型规模的每 GPU 吞吐。ZeRO 让 MP 通常留在单节点内，基线大模型则需要跨节点 MP；这同时改变通信域、本地计算量和 batch 可选范围。

以 PDF Table 5 的 100B 为例：

| 配置 | GPU 数 | MP 度 | 层 / hidden | Attention heads | 本地 / 全局 batch |
| --- | --- | --- | --- | --- | --- |
| ZeRO | 400 | 16 | 125 / 8192 | 64 | 32 / 800 |
| Baseline | 384 | 128 | 125 / 8192 | 128 | 2 / 6 |

所以“最高约 10×”属于作者优化后的整套系统配置比较。它不等于固定模型头数、batch、并行方式后，单个 ZeRO 算法开关带来的孤立加速。

170B 的实验表明特定设置可以执行训练，正文又明确区分了 Turing-NLG 的端到端训练。不能将前者扩写为所有这些尺寸都已训练收敛。

#### 10.3 Super-Linear Scalability

Figure 3 固定 60B 参数模型，设备从 64 扩到 400。PDF Table 6 显示 MP16、75 层、hidden 8192、32 heads 固定，但本地 batch 和全局 batch 都在变：

| GPU | DP | 本地 batch | 全局 batch |
| --- | --- | --- | --- |
| 64 | 4 | 16 | 64 |
| 128 | 8 | 48 | 384 |
| 256 | 16 | 48 | 768 |
| 400 | 25 | 64 | 1600 |

卡更多 → owner 分片更小 → 有空间放更大 batch → 算术强度提高，因而每卡吞吐也可能上升。这条因果链合理，却不是固定全局 batch 的强扩展，也没有单独证明 time-to-quality 超线性改善。

#### 10.4 Democratizing Large Model Training

Figure 4 展示无 MP / PP 时可运行到 13B，§10.4 的文字误指 Figure 5；PDF Table 10 的配置按实验内容对应这条曲线。128 路 DP 的 13B 配置本地 batch 2、全局 256。

它减少模型开发者手工改造层内算子的工作，但仍依赖分布式训练设备，并不意味着单张 V100 能独自训练 13B。ZeRO 的易用性与硬件总投入需要分开衡量。

#### 10.5 Memory and Performance Analysis

Table 3 逐级组合五个配置：

| 配置 | 状态分片 | 其他优化 |
| --- | --- | --- |
| C1 | 优化器 | 固定 buffer + 碎片整理 |
| C2 | 优化器 | C1 + 分片 checkpoints |
| C3 | 优化器 + 梯度 | 固定 buffer + 碎片整理 |
| C4 | 优化器 + 梯度 | C3 + 分片 checkpoints |
| C5 | 优化器 + 梯度 | C4 + checkpoint CPU offload |

**Maximum Model Size**：Figure 6 图题为 Max model size；正文误写 Figure 10。固定 batch 和 MP16 下，C1 → C2 通过激活分片把可运行规模从 40B 推到 60B；C4 达到 140B，C5 进一步到 150B。它与其他实验的“170B”采用不同 batch，不能强行视为冲突。

**Max Cached Memory**：Figure 7 是 PyTorch allocator 的最大 cached memory，包含分配器行为，不能直接视作精确的活跃 tensor 字节和。模型状态与激活占比不同，C2、C3 的大小顺序也可能变化。

**Max Achievable Performance**：Figure 8 展示省显存如何支持更高效 batch，但 C4 → C5 的 60B 反而下降，因为 CPU 传输成本超过收益。170B 则需要 C5 才能运行。最省显存的配置未必最快，offload 只在需要或有利时启用。

#### 10.6 Turing-NLG, the SOTA language model with 17B parameters

本节报告 17B Turing-NLG 端到端训练和约 41.4 TFLOP/s/GPU，Figure 5 为其验证 perplexity 曲线。这里的“当时最大”限定为原文所述 2020 年时间点。

正文将数据集写为 “Webtext-103”，名称与常见 WikiText-103 容易混淆；本笔记不在未核对外部训练报告的情况下替作者消除歧义，也不把其 10.21 perplexity 当作纯系统优化导致的质量收益。

### 3.11 Concluding Remarks

结论强调让大模型训练像普通 DP 一样易用，并以 ZeRO-100B 的实验为已实现成果，将全分片支持更大模型作为继续推进方向。状态分片的数学设计与工程完成度再次需要分开。

### 3.12 Acknowledgement

作者感谢 Microsoft 内部研究和系统评测协作。此节没有增加新的性能配置或收敛证据。

### 3.13 PDF 实验配置附表：Tables 5–10

PDF 第 23–24 页列出各实验的设备数、MP、层数、hidden、头数、本地 batch 与全局 batch，HTML 没有这些表。附表标签沿用了旧图号，因此应以 “superlinear scalability”“only data-parallelism”等描述匹配正文。

附表说明基线部分配置只用了 384 或 256 张卡，受 MP 整除与集群上限限制；作者因此比较每 GPU 吞吐。但按 GPU 归一化并不会自动控制拓扑、batch 与模型头数的影响。本笔记将其列出，读者可以据此判断实验回答的是整套系统能力还是隔离后的算法效率。

本次为原文与数字核对，未运行分布式模型训练。后续复现还需记录具体软件 commit、sequence length、精度、累积策略和对应实验命令；原文未对所有吞吐图统一给出 sequence length，不将 §6 的估算设置自动推广到每张图。

---

## 4. 关键问题解析

### Q1: 为什么 ZeRO 仍属于数据并行？它和张量并行究竟切了什么？

**A**：ZeRO 中每个 DP rank 仍在自己的数据分片上执行整个模型的前后向，只是状态长期由不同 owner 保存。stage 3 计算到一层时需要先获取该层参数。

张量并行让同一个样本的一次算子计算由多个 rank 共同完成，各自输出部分结果后通信合成。二者切分的分别是持久状态归属与算子计算，因此可以组合。

如果同时用 TP 和 ZeRO，先从该 TP rank 负责的参数集合出发，再在相应 DP 组内分片；不能把全体设备数随意当成一个 ZeRO 分片组。

### Q2: ZeRO-1/2/3 的节省倍数为什么分别有不同上限？

**A**：在本文 16 bytes/参数的账本中，stage 1 还留完整参数和梯度，共 4 bytes/参数，因此大 DP 极限只能接近 4 倍；stage 2 留完整 FP16 参数的 2 bytes，因此接近 8 倍；stage 3 将全部长期模型状态都分摊，理想上按 $N_d$ 缩小。

这些是累计启用的阶段，stage 2 包含 stage 1。$N_d=1$ 时三式都回到 $16\Psi$，是一个方便的自检边界。

实际训练的激活、通信桶和物化层参数不会都按这个比例缩小。因此设备总显存的改善小于或不同于纯模型状态的改善。

### Q3: 谁拥有梯度，谁更新权重？为什么无需每卡执行完整 Adam？

**A**：每个参数片段有一个 owner。不同 DP ranks 在不同样本上算出的该片梯度，经 reduce-scatter 汇总给 owner；owner 持有对应主参数和 Adam 两矩，执行局部更新，再将需要的更新参数传播给其他计算参与者。

对于参数分量 $j$，Adam 的 $m_j,v_j$ 更新主要依赖该分量的正确梯度和自身历史；所有 rank 重复做这份工作没有额外信息价值。分片保留了状态数值，只改变计算地点。

stage 1 若仍保留完整本地梯度缓冲，也可以让只负责某片的 owner 使用归约结果；“缓冲是否完整保留”与“所有参数都由每卡更新”是两回事。

### Q4: 状态分片真的不改变优化语义吗？全局梯度裁剪怎么办？

**A**：在相同初始参数、样本 batch、损失归一化和更新规则下，分片归约得到同样的全局梯度，逐参数更新可保持同步优化语义。浮点归约次序仍可能产生数值差别。

全局 gradient norm 不能每个 owner 各算各的阈值。各片先求平方和，再跨 owner 求和得到全局 norm，使用同一个缩放系数；混合精度的溢出判断、loss scaling 等也需一致。以上是由同步更新要求推出的工程条件，原文没有逐项给出这些完整实现细节。

此外，省显存后主动增加全局 batch 是另一个变化。即使 ZeRO 本身不改变更新公式，改变 batch 的训练轨迹仍会变化。

### Q5: “通信量相同”或“只多 50%”意味着通信时间也一样吗？

**A**：不意味着。$2\Psi$ 与 $3\Psi$ 是指定通信计划的总量近似，运行时间还受消息次数、带宽、延迟、拓扑、重叠与物化时机影响。

stage 3 将一次参数收集分散到前后向各个计算单元，消息粒度与依赖位置变了。若重算、梯度累积或预取策略导致参数重复收集，实际量也可能不同于基本账本。

因此该公式证明的是“无需为去冗余付出随 DP 度线性爆炸的通信量”，不是固定的实际 slowdown 上界。本篇 stage 3 没有与 stage 2 同样完整的性能实测。

### Q6: ZeRO-R 的激活分片和 activation checkpointing 是同一项优化吗？

**A**：checkpointing 决定只保存少量边界输入，其余中间激活在反向时重算；激活分片进一步让 MP 组内多个 rank 分别保存边界输入的一部分。需要重算时，先 all-gather 恢复完整输入。

前者用计算换保存量，后者用通信换重复存储。两者可以叠加，也可能分别成为瓶颈。CPU offload 又是第三个维度，把仍需长期保存的数据搬到另一存储层。

所以评估时至少分开测重算时间、all-gather 时间、CPU 传输时间和显存峰值。Figure 8 中 C5 对 60B 变慢、对 170B 成为可运行条件，正说明不同优化的代价要分别记账。

### Q7: 超线性加速是不是违反了“卡数翻倍最多快两倍”？

**A**：没有，因为每张卡的工作效率也在变。Table 6 中更多 DP ranks 让每卡状态更小，进而把本地 batch 从 16 提高到 48、64，矩阵乘和通信摊销变得更高效。

但全局 batch 也从 64 增至 1600。该实验度量某模型在各设备规模的可达吞吐，不是固定全局 batch 的强扩展，也不是达到相同验证质量所需总时间的直接比较。

这种结果值得重视：容量优化能间接改善速度。不过要迁移到自己的训练任务，需要确认允许的全局 batch、学习率和收敛行为。

### Q8: 能把万亿参数模型状态分到 15.6 GB/卡，是否说明 1024 张 32GB 卡一定够？

**A**：这仅说明长期模型状态的理想容量条件。实际还需容纳当前计算单元的参数、输入 / 输出激活、临时通信桶、重算工作集和分配器开销；层很大时瞬时物化本身也可能超限。

即使显存满足，还需验证通信是否高效、数据与优化设置是否可用、总计算时间是否可承担。§9 明确讨论算力缺口，没有给出 1T 端到端训练完成的结果。

读此类容量论文时，可以按“长期状态能否放下 → 峰值能否放下 → 每步是否高效 → 能否以可接受成本收敛”四步检查。ZeRO 主要推进前两项，并对当时可运行规模给出吞吐证据。

---

## 5. 可迁移的知识点

- [ZeRO 状态分片](../../../concepts/zero-state-sharding.md) —— 在数据并行组内改变优化器、梯度和参数的归属，按需通信。
- [激活检查点](../../../concepts/activation-checkpointing.md) —— 重算、保存、分片和 offload 是可组合但代价不同的维度。
- [张量并行](../../../concepts/tensor-parallelism.md) —— 与状态分片组合，使单层计算和持久状态分别适配硬件。

**可迁移的分析方法**：按生命周期画显存账本；区分逻辑通信量与关键路径时间；容量改善引起 batch 改变时，将系统吞吐与优化收敛分开验收；用 PDF 附表核对摘要中省略的实验配置。

---

## 6. 批判与开放问题

### 6.1 作者给出的边界

- 实现与评测覆盖 ZeRO-100B，完整参数分片在所读版本中仍是计划扩展。
- 万亿参数是容量与计算需求分析，作者承认端到端训练时间可能过长。
- CPU offload 只在部分配置有益，60B 的相应消融就有性能退化。
- 更大 batch 可能影响收敛，论文明确讨论这是继续扩展 DP 的限制。

### 6.2 我的质疑

1. **主要速度图同时改变多个条件**。附表给出不同 GPU 数、MP、batch，部分模型还改变头数。它有力展示系统可达能力，但无法隔离每个组件的纯贡献。
2. **理论状态大小不等于训练峰值**。stage 3 的当前层物化、重算和预取需要单独建模；无限加 rank 的理想下界不能直接当成任意模型都能运行的保证。
3. **图号与配置存在错位**。正文、图题和 PDF 附表标签并不完全一致，HTML 还缺末尾表格。复现应按实验内容定位，不能只照抄一句 “Figure X”。
4. **超线性吞吐没有直接证明超线性 time-to-quality**。全局 batch 变化很大，论文没有为该扫描提供统一质量目标下的完成时间。
5. **硬件跨度有限**。主要结果来自 DGX-2 / V100 / 高带宽互联；较小消息、弱网络、不同精度下，容量公式可重算，效率需要重新测量。

### 6.3 开放问题

- 怎样联合选择 DP / TP / PP 与 ZeRO 阶段，使模型、激活、全局 batch 和通信同时可行？
- stage 3 的 gather 粒度、预取、参数驻留与重算如何共同影响峰值和吞吐？
- 分片后的 checkpoint 保存、恢复、世界大小变化怎样保持优化器状态一致？〔本文未完整研究〕
- 如何把容量优化产生的更大 batch 转化为可复现的 time-to-quality 收益？

### 6.4 后续可读

- [Megatron-LM](../2019-megatron-lm/README.md) —— 对照层内计算切分与状态分片。
- [GPipe / PipeDream](../2019-gpipe-pipedream/README.md) —— 从流水调度和权重版本理解另一种训练并行自由度。
- [ZeRO-Offload](https://arxiv.org/abs/2101.06840) · [ZeRO-Infinity](https://arxiv.org/abs/2104.07857) —— 后续将状态管理扩展到异构存储层；功能和结果与原始 ZeRO 分开看。
- [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198) —— 序列并行与重算的进一步取舍。
