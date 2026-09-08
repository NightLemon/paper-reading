---
title: "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve"
authors: ["Amey Agrawal", "Nitin Kedia", "Ashish Panwar", "Jayashree Mohan", "Nipun Kwatra", "Bhargav S. Gulavani", "Alexey Tumanov", "Ramachandran Ramjee"]
affiliation: "Microsoft Research India / Georgia Institute of Technology"
venue: "OSDI 2024"
year: 2024
arxiv: "2403.02310v3"
url: "https://arxiv.org/abs/2403.02310v3"
topic: "llm-serving"
tags: ["chunked-prefill", "stall-free-batching", "scheduling", "pipeline-parallelism", "tail-latency"]
concepts: ["chunked-prefill", "continuous-batching", "kv-cache", "slo-capacity", "prefill-decode-disaggregation", "tensor-parallelism"]
status: "done"
rating: 5
read_date: "2026-09-08"
---

# Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve

> **一句话结论**：将长 prompt 切成 prefill chunks，先安排正在运行的 decode，再用剩余 token 预算填入 chunks，让请求持续生成的同时接纳新请求；收益来自限制每轮计算量和降低流水批次间的运行时间差异。

---

## 1. 元信息

| 项目 | 内容 |
| --- | --- |
| 作者 / 机构 | Amey Agrawal、Alexey Tumanov（Georgia Tech）；Nitin Kedia、Ashish Panwar、Jayashree Mohan、Nipun Kwatra、Bhargav S. Gulavani、Ramachandran Ramjee（Microsoft Research India） |
| 发表 | OSDI 2024；本文按 arXiv v3（2024-06-17）精读 |
| 原文 | [PDF](paper.pdf) · [HTML](paper.html) · [固定版本页面](https://arxiv.org/abs/2403.02310v3) |
| HTML 阅读 | 样式与部分图像从 arXiv 联网加载；离线阅读使用同目录 PDF |
| 代码 | [microsoft/sarathi-serve](https://github.com/microsoft/sarathi-serve)；附录指定论文实验分支 `osdi-sarathi-serve` |
| 关键词 | chunked-prefill · stall-free batching · token budget · TBT · pipeline bubbles |
| 前置知识 | [Continuous Batching](../../../concepts/continuous-batching.md)、[KV Cache](../../../concepts/kv-cache.md)、[张量并行](../../../concepts/tensor-parallelism.md)、[SLO 容量](../../../concepts/slo-capacity.md) |

**版本与历史边界**：本文是 Sarathi-Serve（2024）。其前作 [SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills](https://arxiv.org/abs/2308.16369) 于 2023 年公开。DistServe 原文参考文献 [9] 引用的是这篇前作。下面区分前作、本文机制和论文实验中的基线实现；“vLLM”在结果比较中均指本文当时使用的基线。

---

## 2. 摘要速览（5 分钟版）

### 2.1 要解决的问题

Orca 已经把调度粒度降到每次迭代，vLLM 又让更多请求的 KV 装得下。然而，新请求的 prefill 可能包含数千个 token，一旦直接插入执行队列，正在生成的请求会等很久才吐出下一个 token。这种 **generation stall** 在交互界面上表现为回答突然停住。

prefill 优先有利于尽快补充 decode 批次，但损害 TBT；一直先跑现有 decode，则新请求迟迟进不来，随着旧请求退出，batch 越来越小。多 GPU 流水还有另一层问题：长 prefill 批与短 decode 批交错，前后 stage 的工作量失衡，GPU 会等待。

### 2.2 核心方法

1. **Chunked-prefills**：把一个长 prompt 的计算分散到多次迭代；后面的 chunk 读取前面 chunks 留下的 KV，保持完整的因果注意力。
2. **Stall-free batching**：先给每个正在 decode 的请求安排一个 token，再继续未完成的 prefill，最后接纳新请求；所有 prefill tokens 共用剩余预算。
3. **Token budget**：约束一轮的总 token 数，按模型、硬件、并行配置与 TBT SLO 做 profiling 后选择。
4. **近似均匀的流水批次**：让不同 microbatch 的计算量更接近，减少 stage 等待。

```text
每轮预算 τ
  ├─ 先安排所有运行中的 decode：每请求 1 token
  ├─ 继续已经开始的 prefill：占用剩余预算
  └─ 显存与预算允许时接纳新请求：只取装得下的 prompt chunk
                         ↓
             执行一个含两类 token 的 hybrid batch
                         ↓
              更新 KV、prefill 进度与完成状态
```

这里的 stall-free 表示避免整段 prefill 挤占生成进度。混批仍会增加单轮执行时间，预算本身也需要校准。

### 2.3 主要结果

下面优先列出基线与条件明确的结果；容量指满足尾延迟约束、且队列能稳定维持的最大请求到达率。

| 原文位置 | 模型、硬件、负载与比较对象 | 结果 |
| --- | --- | --- |
| §5.1、Figure 10a | Yi-34B，2×A100 80GB，TP2，openchat_sharegpt4 长度分布、Poisson 到达，p99 TBT ≤ 0.2 s；比较 Orca / vLLM | 正文报告容量分别为两者的 **4.0× / 3.7×** |
| §5.3、Figure 13b | Falcon-180B，2 节点共 8×A100 80GB，节点间 100 Gbps Ethernet，TP4-PP2，openchat_sharegpt4；比较同一并行配置的 vLLM | p99 TBT ≤ 1 s 时容量约 **3.6×**；放宽到 5 s 时约 **1.48×** |
| §5.4.2、Table 4 | Yi-34B，2×A100，128 个请求，预算 1024，openchat_sharegpt4；比较仅 hybrid batching | p50 TTFT **0.53 → 0.76 s**，p99 TBT **0.68 → 0.14 s**：TBT 改善伴随 TTFT 代价 |

所有容量实验还要求**中位 scheduling delay 不超过 2 s**。Table 4 是固定请求集合的机制消融，不能与最大容量实验直接拼接比较。

**数字阅读提醒**：摘要的 Falcon-180B“最高 5.6×”未在该句注明基线；Figure 11a 中约 5.62× 的柱顶标注对应与 Orca 的比较，不能写成相对 vLLM。§5.2 正文和 Figure 12 图注对“3.5×”对应的模型也存在不一致，本笔记不将这条作为无歧义的主要结果。

### 2.4 我的评价

评分 ⭐⭐⭐⭐⭐。本文补上了“每轮能调度”之后的下一步：**限制一轮到底可以做多少工作**。它尤其适合和 Orca、DistServe 连着读：三者分别改变调度粒度、同实例内的工作分配、以及阶段放置位置。

它的价值也在于提醒读者关注服务容量。显存允许更大的 batch，并不意味着尾延迟 SLO 允许系统实际使用它。先触及哪条约束，决定优化是否有效。

---

## 3. 细读

> 以下 3.1–3.8 对应原文 §1–§8；3.9 对应 Appendix A。子标题沿用原文。公式之外的调度示例和成本估算会明确标注为本笔记推导。

### 3.1 Introduction

开篇用 Figure 1 给出可见症状：Yi-34B 在两张 A100 上执行 arxiv_summarization trace 的 128 个请求，vLLM 的生成时间线存在持续数秒的空档。论文关心的是**相邻输出 token 之间的间隔**，因此即使请求最终能够完成，服务体验也可能已经不满足要求。

Figure 2 把调度策略放在吞吐与延迟的二维空间中，但图注明确说明它是示意图；图上的位置不能当作实验结果。真正的证据在后面的负载扫描与消融。

作者同时引入流水问题：训练中常见的前后向依赖在纯推理里消失了，但不同 microbatch 的运行时差异仍会造成 bubbles。“只有前向”不足以推出“流水没有气泡”。

> **批注**：本文把一个用户能感知的停顿，追溯到调度单位的工作量不受控。这条因果链连接了应用体验、调度策略和 GPU 利用率，解释了为什么只改 kernel 或显存分配还不够。

### 3.2 Background

#### 2.1 The Transformer Architecture

每层包含 self-attention 与 FFN。QKV 投影、输出投影和 FFN 的线性层可以在不同请求的 token 间共享权重读取；attention 的 QK 与加权 V 则必须保留请求边界。这个区分是后续混批提高算术强度的前提。

#### 2.2 LLM Inference Process

prefill 同时处理 prompt 中的多个位置，decode 每个请求每轮只提供一个新位置。论文实验所覆盖的模型与上下文下，前者通常能充分利用算力，后者的小 batch 通常受访存限制。随着 decode batch 增大、上下文增长，瓶颈占比会改变；这些分类是性能分析的起点。

PagedAttention、MQA/GQA 缓解 KV 容量限制，让更大的 decode batch 成为可能。此时“能否保持大 batch 且不违反延迟要求”就更值得单独优化。

#### 2.3 Multi-GPU LLM Inference

TP 在每层划分权重与计算，需要频繁 all-reduce；PP 按层划分，stage 边界传递激活。跨节点网络较慢时，后者的通信粒度更合适，但必须处理流水等待。

与 [Megatron-LM](../../llm-training/2019-megatron-lm/README.md) 的连接在于：训练里推导出的张量切分方式决定了推理通信量；服务调度又决定这些通信之间能否持续安排有用计算。

#### 2.4 Performance Metrics

设一个请求到达时刻为 $a$，第 $i$ 个输出 token 的时刻为 $t_i$。按定义可以写成：

$$\mathrm{TTFT}=t_1-a,\qquad \mathrm{TBT}_i=t_i-t_{i-1}\quad(i\ge2).$$

论文关注 p50 TTFT 与 p99 TBT。前者每请求一个样本，后者每个 decode token 一个样本，所以**长输出请求会贡献更多 TBT 样本**。这是指标的统计口径，不能直接理解成“99% 请求都完全满足 SLO”。

#### 2.5 Scheduling Policies for LLM Inference

Algorithm 1 展示请求级 batching：等当前批所有 decode 结束后再加新请求。Algorithm 2 展示 prefill 优先的迭代级策略：可以加请求时先做 prefill，否则做 decode。前者保住生成平滑性，后者更容易维持批大小。

Orca 还支持 hybrid batch，但完整 prompt 仍可能让混合迭代很长。于是本文需要同时解决**让两类请求进入同一轮**与**限制 prefill 工作量**两个问题。

> **批注**：这里只把两类调度作为解释框架。原文 §3.2 自己也进一步区分 Orca 的混批与 vLLM 的纯 prefill / 纯 decode 批，不能把它们画成完全相同的执行时间线。

### 3.3 Motivation

#### 3.1 Cost Analysis of Prefill and Decode

Figure 3 用 Mistral-7B / 单张 A100、prompt 长度 1024 扫 batch size：decode 吞吐随着 batching 明显上升，prefill 在一个请求就有许多 token，继续增加请求的边际收益小得多。两张图纵轴不同，应比较各自随 batch 的趋势。

Figure 4 把执行时间分为 linear、attention、others。在该图测量范围里，linear 占主要时间；这支持作者先分析权重加载和矩阵乘的算术强度，但不构成任意长上下文、任意模型都由 linear 主导的证明。

作者使用近似模型：

$$T\approx\max(T_{\mathrm{math}},T_{\mathrm{mem}}).$$

如果权重读取已经决定了时间，那么增加一些复用同一权重的 token，可能主要增加原本闲置的算术工作。Figure 5–6 用 LLaMA2-70B、A100 和不同 TP 配置展示这个区域：小 batch 到一定规模前，时间变化较小，之后才近似随 token 数线性上升。

**这个“余量”是可用算力，不是空闲时间槽**。prefill 与 decode 在同一批线性层中计算，二者共享权重读取成本。固定开销与并行配置会移动拐点，所以不能从 GPU 峰值 FLOPs / 带宽直接算出一个普适的 chunk size。

#### 3.2 Throughput-Latency Trade-off

Figure 7 比较四条时间线。请求 A、B 已在 decode，C、D 后到：vLLM 先做新 prefill，A、B 等待；Orca 将完整 prefill 混入本轮，A、B 虽参与执行，却仍要等长迭代结束；请求级策略让 C、D 等到旧请求结束；Sarathi-Serve 则把 C、D 的 prefill 分摊到后续多轮。

**“参与了一轮”与“及时拿到一个 token”是两个条件**。Orca 满足前一个条件，长 prefill 仍可能破坏后一个条件。这是从 continuous batching 到 chunked prefill 最重要的一步。

#### 3.3 Pipeline Bubbles waste GPU Cycles

Figure 8 区分三种运行时间差异：不同 prefill 长度、prefill 与 decode 交替、以及不同历史上下文长度导致的 decode attention 成本变化。只限制批内请求数无法让这三种差异消失。

给 microbatch 相近的 token 数能够缓解前两类差异。第三类仍受已缓存 KV 长度影响，所以本文表述宜为“减少 bubbles”，而非对任何负载彻底消除。

> **批注**：流水均匀性与单轮效率是两个目标。把每批做得非常大可能减少固定开销，却放大一次慢批对下游 stage 的影响。局部最优的 batch size 不必是整条流水最优的 batch size。

### 3.4 Sarathi-Serve: Design and Implementation

#### 4.1 Chunked-prefills

一个 prompt 的 chunk $j$ 要读取 chunks $1\ldots j-1$ 的 KV，以及本 chunk 内满足 causal mask 的位置。chunk 边界切分的是**执行进度**，上下文仍然连通。

因果性保证早期位置的隐藏状态不依赖后续 token。逐层完成前一段并缓存其 KV 后，就能继续处理后一段。每个请求需要额外记录“prompt 已计算到哪个位置”，直到整个 prefill 完成才进入常规 decode。

Figure 9 比较 decode-only、加入完整 prefill、加入 chunk 三种情况，说明限制 prefill token 数能显著控制新增延迟。它也显示混批成本依赖 decode batch size 与上下文，不能将 chunk 的成本固定写成零。

#### 4.2 Stall-free batching

Algorithm 3 的优先级为：**运行中的 decode → 未完成 prefill → 新请求**。下面按该机制改写成可读伪代码；原文排版中的批集合赋值写法不作为可直接执行的程序照抄。

```text
while 有运行中或等待中的请求:
    batch = 所有运行中的 decode 请求各取 1 个 token
    remaining = token_budget - batch.token_count
    按已有调度顺序继续部分完成的 prefill:
        取 min(剩余 prompt 长度, remaining) 个 token
        remaining -= 本次取出的数量
    在 remaining > 0 且 KV 分配允许时接纳新请求:
        只取能装进 remaining 的首个 chunk
        更新 remaining
    执行 hybrid batch
    更新 prefill 进度，移除已结束请求
```

该流程隐含“正在 decode 的请求能够放入当前预算”的可行条件。论文通过预算与最大 batch size 的配置保持工作区间；它没有为任意过载输入提供普遍的服务保证，见 Q4。

**本笔记示例**：预算 512，已有 64 个 decode 请求，则本轮留给 prefill 的空间为 448 tokens。一个剩余 1000 tokens 的 prompt 可以分多轮完成；每轮实际 chunk 长度还取决于当时有多少 decode token，而非永远等于 512。

#### 4.3 Determining Token Budget

预算越小，单轮 latency 越容易控制，但会降低部分矩阵乘的效率，增加 kernel 启动次数和历史 KV 读取。论文指出，若 prompt 被切为 $N$ 块，第一块的 KV 会被后续 $N-1$ 块读取，第二块被后续 $N-2$ 块读取，以此类推。

这没有增加因果注意力需要计算的合法 QK 对，但改变了数据在不同 kernel 调用之间被读取的次数。实际额外 HBM 流量还受 cache 与 kernel 分块影响，不能把抽象块访问数直接当作硬件计数器结果。

预算还要考虑 tile quantization：维度落在 tile 边界之外，可能多启动一个利用率很低的 tile。论文用 256 与 257 的例子说明非单调现象，但未在该句给出完整模型配置，本笔记仅保留机制结论。

作者的做法是一次性 profiling，并利用 Vidur profiler / simulator 挑选符合部署条件的预算；动态随负载变化的预算是未来工作。PP 下还必须把运行时方差带来的气泡算进去。

#### 4.4 Implementation

系统建立在 vLLM 开源实现上，加入 paged chunk prefill、调度策略、PP 与 telemetry。支持 FlashAttention v2 和 FlashInfer backend，本文实验采用 FlashAttention backend；TP 和 PP 通信均使用 NCCL。

> **批注**：调度算法很短，落地条件并不短。kernel 必须支持“已有一段历史 KV，再处理多个新位置”，调度器必须维护部分 prefill 的状态，显存分配器要与进度一致。只在队列上切 token 数无法单独实现这套机制。

### 3.5 Evaluation

#### 实验设置：Tables 1–3

| 模型 | 部署 | strict / relaxed p99 TBT |
| --- | --- | --- |
| Mistral-7B | 1×A100 80GB，GQA + sliding window | 0.1 / 0.5 s |
| Yi-34B | 2×A100 80GB，TP2，GQA | 0.2 / 1 s |
| LLaMA2-70B | 8×A40 48GB，TP4-PP2，GQA | 1 / 5 s |
| Falcon-180B | 2 节点各 4×A100 80GB，TP4-PP2，节点间 100 Gbps Ethernet，GQA | 1 / 5 s |

负载根据 openchat_sharegpt4 与 arxiv_summarization 的长度分布构造，使用 Poisson 到达；过滤总长度分别超过 8192、16384 的请求。Table 2 的 prompt 中位数分别为 1730、7059，output 中位数分别为 415、208。

SLO 以特定模型和硬件、4K prefill 长度、decode batch size 32 的无 prefill 干扰迭代时间为参照，strict / relaxed 分别取其 5 / 25 倍。因此跨模型的同名 strict 并不是相同的毫秒阈值。

#### 5.1 Capacity Evaluation

Figure 10–11 的横轴组合了模型与 SLO，纵轴为最大 QPS。很多基线首先触及 TBT 约束，而不是算力或显存的绝对上限。放宽 SLO 后，基线容量会明显上升，这恰好说明“峰值吞吐”与“可对外服务的容量”之间的距离。

严格 SLO 下，各模型使用预算 512；宽松时通常用 2048，LLaMA2-70B 为减轻流水气泡改用 1536。这个例外很有价值：即使延迟允许更大的 chunk，流水整体也可能更喜欢较小的 chunk。

对柱顶倍数必须先识别分母。例如 Figure 10a 的 Yi-34B strict 约 4.00× 对应 Orca，正文另报相对 vLLM 3.7×。Figure 11a 的 Falcon relaxed 5.62× 也不能不加区分地用作相对 vLLM 的收益。

#### 5.2 Throughput-Latency Tradeoff

Figure 12 扫描多个 p99 TBT 目标。vLLM 测最大 batch size 32、64、128；Sarathi-Serve 测预算 512、2048，最大 batch size 128。其要回答的是：**哪一个控制量能够有效移动延迟与容量的折中点？**

仅减少请求数仍可能遇到一个非常长的 prompt，因此对延迟毛刺的控制较粗。token budget 直接限制新增 prefill 工作量，对这个问题更有针对性。较大预算在宽松 SLO 下更有效率，较小预算在紧 SLO 下允许继续服务。

此版本正文将某处 3.5× 归到 Mistral-7B，图注则写 Yi-34B。这里保留曲线的定性结论；需要精确使用该数据时，应明确指定图中的模型、横坐标与对应基线，不用这句有冲突的概括代替。

#### 5.3 Making Pipeline Parallel Viable

Figure 13 比较 Falcon-180B 的 TP8、vLLM TP4-PP2、Sarathi-Serve TP4-PP2。在两节点 100 Gbps Ethernet 上，跨节点 TP 的频繁 all-reduce 抬高了 decode latency；改成节点内 TP、节点间 PP 可以减少这部分通信代价。

但 **换成 PP** 与 **改成均匀批次** 是两项独立变化。公平判断调度收益，应先比较两个 TP4-PP2 配置。严格 SLO 下约 3.6×、宽松 SLO 下约 1.48× 的容量增益是这一对照；相对 TP8 的收益还包含并行策略差异。

#### 5.4 Ablation Study

**5.4.1 Overhead of chunked-prefills**：Figure 14 在 Yi-34B / TP2、两张 A100 上，以无分块 prefill 为基准，测试不同 prompt 长度与 512 / 1024 / 2048 的 chunk。512 的额外时间最高约 25%，2048 则接近无额外开销。这说明低 TBT 是通过一个有代价的分块策略换来的。

**5.4.2 Impact of individual techniques**：Table 4 固定 Yi-34B / 两张 A100、128 个请求、预算 1024，时间单位为秒：

| 策略 | ShareGPT p50 TTFT | ShareGPT p99 TBT | arXiv p50 TTFT | arXiv p99 TBT |
| --- | --- | --- | --- | --- |
| 仅 hybrid batching | 0.53 | 0.68 | 3.78 | 1.38 |
| 仅 chunked prefill | 1.04 | 0.17 | 5.38 | 0.20 |
| 两者结合 | 0.76 | 0.14 | 3.90 | 0.17 |

组合方法相对“仅分块”同时改善两项指标；相对“仅混批”则主要改善 TBT，TTFT 略差。原文说“两维都改善”需要结合比较对象理解，不能写成它严格支配表内所有方案。

> **批注**：这是全文最适合检验因果解释的表。分块控制每轮长度，混批利用余量；两个组件解决不同问题。最终结果仍然有取舍，而不是消除了所有代价。

### 3.6 Related Work

原文对 DistServe、Splitwise、TetriInfer 的评价较平衡：PD 分离可以消除两阶段同实例执行的干扰，让 prefill 全速进行；代价包括 KV 迁移以及 prefill 实例的显存利用方式。作者明确把与这些方案的**定量比较留给未来工作**。

因此本文能支持的结论是：同实例混批存在有效的优化空间。本文没有证明它普遍优于 PD 分离，也没有用同预算硬件完成二者的全面评测。

公平性、抢占、kernel 优化和模型压缩分别改变请求次序、执行效率或资源需求；作者将它们视为可以组合的方向。组合后的尾延迟仍需单独验证。

### 3.7 Conclusion

作者将结果归纳为 chunked-prefills 与 stall-free batching 的组合，使接纳新请求时持续推进原有 decode，并让流水批次更均匀。摘要式的最高倍数需要带回 §5 的模型、SLO、负载与基线后再使用。

### 3.8 Acknowledgement

致谢 OSDI 评审与 shepherd，并列明 Georgia Tech / Microsoft 等支持来源。本节不包含额外算法或性能证据。

### 3.9 Appendix A: Artifact Appendix

附录给出代码组织：系统主体在 `sarathi`，自定义 kernel 在 `csrc`，论文实验脚本在 `osdi-experiments`，trace 在 `data`。测试环境包括 CUDA 12.1、A100 与 A40。

复现入口是 `osdi-sarathi-serve` 分支；主分支持续更新。若将来复现实验，应进一步记录具体 commit、模型配置、trace、SLO 和预算，避免把今天的主分支结果当成 2024 年论文结果。本次精读核对原文与图表，未运行 GPU benchmark。

---

## 4. 关键问题解析

### Q1: Orca 已有 continuous batching 和 hybrid batch，为什么还需要 Sarathi-Serve？

**A**：continuous batching 让调度器每轮重新选择请求，hybrid batch 让 prefill 与 decode 一起进入执行；这两者都没有限制新 prompt 的总计算量。一个很长的 prefill 即使和 decode 一起跑，也会让这轮结束得很晚。

Sarathi-Serve 增加的是工作量约束：先让每个运行中的 decode 占一个位置，再限制 prefill 最多补多少 token。Figure 7 中 Orca 与本文的差异就在这里。

因此三项能力可以逐层理解：**每轮可换成员 → 同轮可混合阶段 → 同轮工作量受控**。它们互相支撑，但不能互相替代。

### Q2: prefill 被切开后，后半段还能看到前半段吗？会改变模型输出吗？

**A**：后一个 chunk 的 query 读取前面 chunks 的 KV，并在自己的 chunk 内应用 causal mask。对 decoder-only 的因果 Transformer，早期位置不依赖后面位置，先算它们再缓存是合法的执行重排。

例如 prompt 长 6、chunk size 2，第三块的位置 5、6 可以读取位置 1–4 的缓存；位置 5 看不到位置 6，位置 6 则可以看 1–6。这里没有把注意力窗口缩成 2。

在相同模型和位置处理下，计算保持原有注意力语义；实际 kernel 的浮点归约顺序可能不同。本文没有提供逐位输出一致性证明，因此应把数学语义不变与 bitwise identical 区分开。

### Q3: 为什么把额外的 prefill token 放进去，decode 不会按 token 数同比变慢？

**A**：小 decode batch 的线性层往往由读权重的时间主导。多个 token 共用一份权重，把额外 token 放进同一个矩阵乘后，权重流量没有按 token 数重复，算术工作可以占用原本闲置的执行单元。

在 $T\approx\max(T_{\mathrm{math}},T_{\mathrm{mem}})$ 近似下，只要额外计算尚未超过访存时间，延迟增长就可能较小。一旦超过拐点，或 attention / 通信成为瓶颈，继续填入 token 就会明显变慢。

这解释的是为什么存在值得利用的区间，不是保证 prefill 免费。Figure 9 和 Table 4 都应在这个边界内理解。

### Q4: token budget 和最大 batch size 有什么不同？“stall-free”保证到什么程度？

**A**：最大 batch size 数请求，token budget 数**本轮要计算的新位置**。一个 decode 请求通常贡献 1，一个新 prefill 请求可能贡献数百个。相同请求数对应的执行时间可能相差很大。

设正在 decode 的请求数为 $D$，本轮选择的 prefill chunk 长度为 $c_j$，则本文预算约束可写为：

$$D+\sum_jc_j\le\tau.$$

这要求 $D\le\tau$，还需要显存分配可行。若仅 decode 已经耗尽预算，新 prefill 就无法在这一轮前进；若持续过载，队列仍会增长。论文的 capacity 实验主动限制队列增长，未声称任意到达负载下都能满足 SLO。

stall-free 是消除“整段 prefill 抢走多个 decode 周期”的机制，不是对网络、内存回收、所有上下文长度和任意到达过程的实时系统保证。

### Q5: chunk 越小，p99 TBT 越容易控制，那为什么不切成一个 token？

**A**：极小 chunk 会让 prefill 也落入低算术强度区域，增加 kernel 次数、调度次数，并反复读取历史 KV。它相当于放弃 prefill 本来可以并行处理多个位置的优势。

Figure 14 的具体证据是 Yi-34B / TP2 的 512-token chunk 在所测长度范围额外耗时最高约 25%，2048 则接近无开销。原文还提示 tile 对齐与 PP 气泡会让最优点发生变化。

正确的选择目标是**满足延迟约束时的服务容量**，而非单独最小化 chunk 长度。预算取值要和模型、硬件、TP/PP、负载一起记录。

### Q6: 相同 token 数就能消除 pipeline bubbles 吗？

**A**：相同 token 数让主要线性层的计算规模接近，因此通常比任意长度的完整 prefill / decode 交错更均匀。但 attention 仍要读取不同长度的历史 KV，不同请求的计算与访存成本不会完全一样。

例如两个批次各有相同数量的 decode token，其中一个已有较长上下文，它的 attention 访问量仍更大。这正是原文 Figure 8 的第三类 bubble 来源。

所以本文的技术提高了 PP 的可用性，没有给出 microbatch 等 token 数就等运行时间的定理。更精细的成本预算是否值得其调度开销，是一个合理的后续问题〔本笔记推断〕。

### Q7: Sarathi-Serve 与 DistServe 的关系是什么？哪一个更适合部署？

**A**：两者处理同一个干扰，但改变的资源安排不同。

| 维度 | Sarathi-Serve | DistServe |
| --- | --- | --- |
| 阶段放置 | 同一实例混合执行 | prefill / decode 分到不同实例 |
| 主要控制量 | 每轮 token budget、批组成 | 两类实例的数量、并行策略与放置 |
| 新增代价 | 分块开销、共享资源下的残余延迟取舍 | KV 传输、实例配置与资源配比 |
| prefill 的执行 | 分摊到多轮 | 在 prefill 实例独立执行 |
| 本文是否直接公平测过二者 | §6 明确留作未来工作 | 不能由本文柱状图决定胜负 |

只有少量 GPU、希望统一调度时，同实例分块是值得评估的方案；具备传输能力且需要分别扩展两阶段时，分离提供另一种自由度。以上是机制导出的部署判断，具体成本和 SLO 仍需测量。

历史上还需区分 SARATHI 2023 与 Sarathi-Serve 2024。DistServe 引用前者；其对 chunked prefill 的论述可以作为设计比较入口，但不能写成对本文所有新机制与实验的逐项反驳。

### Q8: 怎样读本文的 capacity，才能避免把它误当“GPU 快了几倍”？

**A**：capacity 是在特定延迟约束下最多能持续接多少 QPS。基线可能在尚有硬件余量时就因为长 prefill 触发 p99 TBT 违约；调度器降低这些毛刺后，允许更高负载进入，因此 capacity 可以大幅增加。

这不意味着单个 token 的 kernel 时间同比缩短。至少要同时记下：模型与硬件、并行策略、输入输出长度分布、到达过程、p99 TBT 阈值、队列稳定条件、基线实现。不同 SLO 下的倍数也不能互换。

本文 p50 TTFT 与 token 级 p99 TBT 的口径还不同于 [DistServe](../2024-distserve/README.md) 的双延迟目标下 goodput。跨论文比较时先统一指标，再讨论算法收益。

---

## 5. 可迁移的知识点

- [Chunked Prefill](../../../concepts/chunked-prefill.md) —— 把长输入计算拆成有依赖关系的调度单元，用预算限制单轮干扰。
- [Continuous Batching](../../../concepts/continuous-batching.md) —— 提供每轮重新组批的基础，本工作进一步约束批内计算量。
- [KV Cache](../../../concepts/kv-cache.md) —— 保存前面 chunks 的状态，使后面 chunks 保持完整因果上下文。
- [SLO 容量与 p99 尾延迟](../../../concepts/slo-capacity.md) —— 解释“硬件还没满，服务已经不能继续加负载”。
- [Prefill / Decode 分离](../../../concepts/prefill-decode-disaggregation.md) —— 用空间分离处理同一干扰，与同实例分块形成设计对照。
- [张量并行](../../../concepts/tensor-parallelism.md) —— TP 的通信成本影响是否需要 PP，以及预算如何取值。

**可迁移的方法**：当调度器按“请求个数”做决定而请求成本差异很大时，考虑引入更接近实际工作的预算；当流水吞吐低于单级吞吐时，除了平均时间，还应看批次运行时的方差。两条都是本文机制的归纳，适用性需结合具体工作负载验证。

---

## 6. 批判与开放问题

### 6.1 作者明确给出的边界

- §4.3 承认 chunking 的 KV 重读、kernel 启动和效率代价；预算依赖硬件与并行配置。
- §5.1 使用固定预算，动态随负载调整被留为未来工作。
- §6 未做与 PD 分离方案的定量比较，不能从本文得到统一部署排名。
- 附录给出的测试硬件为 A100 / A40、CUDA 12.1；其他硬件上的预算和收益需要重新验证。

### 6.2 我的质疑

1. **SLO 覆盖范围与产品体验仍有距离**。容量实验约束 p99 TBT 和中位 scheduling delay，未同时限定请求级尾 TTFT。新请求中的少数长等待可能不会被中位数暴露。这是指标口径的局限，不等于论文观测到了该退化。
2. **到达过程和长度过滤缩小了证据范围**。Poisson 到达、过滤超长请求有利于可控比较；突发流量和保留更长上下文时，固定预算与资源占用的组合需要另测。
3. **同 token 数是成本代理**。不同 KV 长度和 attention 变体会改变时间，论文已承认相应气泡来源。需要测量何时这个代理足够准确，何时值得引入长度相关成本模型。
4. **摘要、正文与图注需要对账**。最高倍数的比较对象，以及 §5.2 的模型归属不够统一。本笔记选择可定位的结果，不替作者消除未解释的不一致。
5. **消融显示的是折中改善**。Table 4 中组合方案的 TTFT 并未优于“仅混批”，因此将结果概括成任何条件下同时更快，超出了表格证据。

### 6.3 开放问题

- 如何在线调整预算，同时避免因频繁变化扩大流水运行时方差？
- 能否将 TTFT、TBT、KV 占用时间共同纳入预算选择，使长 prefill 的等待也受控？
- 同一硬件预算下，何种负载与网络条件使分块混批、PD 分离或两者组合更合适？
- 与 prefix caching、抢占和 speculative decoding 同时使用时，如何让“每轮 token 数”仍能准确表达实际成本？

### 6.4 后续可读

- [Orca](../2022-orca/README.md) —— 回看 iteration-level scheduling 的自由度和限制。
- [vLLM / PagedAttention](../2023-vllm-pagedattention/README.md) —— 显存容量如何支撑更大的活跃请求集合。
- [DistServe](../2024-distserve/README.md) · [Mooncake](../2025-mooncake/README.md) —— 阶段分离和 KV 存储设计如何改变这笔资源账。
- [SARATHI（2023）](https://arxiv.org/abs/2308.16369) —— 追溯 chunked prefills 与 decode piggybacking 的前作。
- [Vidur](https://arxiv.org/abs/2405.05465) —— 本文预算选择使用的推理系统模拟与配置探索工具。
