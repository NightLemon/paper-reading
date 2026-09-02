---
title: "Orca: A Distributed Serving System for Transformer-Based Generative Models"
authors: ["Gyeong-In Yu", "Joo Seong Jeong", "Geon-Woo Kim", "Soojeong Kim", "Byung-Gon Chun"]
affiliation: "Seoul National University / FriendliAI"
venue: "OSDI 2022"
year: 2022
url: "https://www.usenix.org/conference/osdi22/presentation/yu"
topic: "llm-serving"
tags: ["continuous-batching", "iteration-level-scheduling", "selective-batching", "osdi"]
concepts: ["continuous-batching"]
status: "done"
rating: 5
read_date: "2026-09-02"
---

# Orca: A Distributed Serving System for Transformer-Based Generative Models

> **一句话结论**：把调度粒度从「一整个请求」降到「一次迭代」，再用 selective batching 让形状不齐的请求仍能合批——在 GPT-3 175B / 16 张 A100 上，同等中位归一化延迟（190 ms/token）下吞吐从 FasterTransformer 的 0.185 req/s 提到 6.81 req/s，36.9×。

---

## 1. 元信息

| 项目 | 内容 |
| --- | --- |
| 作者 / 机构 | Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, Byung-Gon Chun · Seoul National University / FriendliAI |
| 发表 | OSDI 2022（16th USENIX Symposium on Operating Systems Design and Implementation, pp. 521–538） |
| 原文 | [USENIX 页面](https://www.usenix.org/conference/osdi22/presentation/yu) · 本地 [paper.pdf](paper.pdf)（无 arXiv 版，仅 PDF） |
| 关键词 | iteration-level scheduling · selective batching · continuous batching · Attention K/V manager |
| 前置知识 | [KV Cache](../../../concepts/kv-cache.md)、[Continuous Batching](../../../concepts/continuous-batching.md)、[分页式 KV 管理](../../../concepts/paged-kv-memory.md)、[张量并行](../../../concepts/tensor-parallelism.md) |
| 代码 | 未开源（FriendliAI 的商业产品 PeriFlow / Friendli Engine 是其延续） |

---

## 2. 摘要速览（5 分钟版）

### 2.1 要解决的问题

生成式 Transformer 的推理有一个 ResNet、BERT 都没有的性质：**处理一个请求要跑模型很多次**，每次迭代只产出一个 token，产出的 token 又作为下一次迭代的输入。而当时的推理服务系统（Triton、TensorFlow Serving）与执行引擎（FasterTransformer、TensorRT）之间的接口是**请求粒度**的：调度器把一批请求交给引擎，引擎要等这批里**所有**请求都生成完才返回。

论文把由此产生的问题命名为 **C1: early-finished and late-joining requests**：

- **提前结束的请求出不去**。同一批里各请求需要的迭代数不同，先生成出 `<EOS>` 的请求不能提前返回客户端，必须陪着最长的那个跑完。论文 Figure 3 给的例子里，$x_2$ 在 iter 2 就结束了，但引擎在 iter 3、iter 4 仍要为它做「无效计算」。
- **新到达的请求进不来**。批执行期间到达的请求必须等整批结束，排队时间被显著拉长。

论文特别指出，**这个问题在训练时不存在**：训练用 teacher forcing，整批在一次迭代内就处理完了。它是推理独有的。

### 2.2 核心方法

两个技术，缺一不可：

**S1: Iteration-level scheduling（迭代级调度）**。调度器不再「交出一批请求等结果」，而是循环执行：(1) 从 request pool 选一组请求；(2) 调引擎只跑**一次迭代**；(3) 收回这一次迭代的结果。因为每次迭代后调度器都拿到控制权，它能立刻发现完成的请求并返回客户端，新请求也最多等一次迭代就有机会入选。

**S2: Selective batching（选择性合批）**。迭代级调度带来一个新麻烦（论文的 **C2**）：被选中的请求彼此形状不齐，无法整体合批。论文列了三种不能合批的情形：

1. 两者都在 initiation 阶段但输入 token 数不同；
2. 两者都在 increment 阶段但处理的 token 下标不同（KV 张量形状不同）；
3. 一个在 initiation、一个在 increment（一个吃全部输入 token，一个只吃 1 个）。

selective batching 的解法是**按算子区别对待**：Linear、LayerNorm、Add、GeLU 这些不需要「请求」概念的算子，把 batch 拉平成 $[\sum L, H]$ 的二维张量做 token 级合批；只有 Attention 需要区分请求（只能在同一请求的 token 之间算注意力），就在它前面插一个 Split、后面插一个 Merge，逐请求单独算。

关键洞察在这里：**Attention 不带模型参数**，所以合批它拿不到「复用已加载权重」的好处，不合批的代价因此很小。

### 2.3 主要结果

实验环境：Azure ND96asr A100 v4，每台 8 张 40GB A100（NVLink 互联），最多 4 台，机间 8 张 Mellanox 200Gbps HDR InfiniBand（1.6 Tb/s）。模型为 GPT-3 系列，fp16，最大序列长度 2048。基线是 NVIDIA FasterTransformer。

| 实验 | 配置 | 结果 |
| --- | --- | --- |
| 引擎微基准（关闭调度器） | 13B / 1 GPU、101B / 8 GPU | Orca 引擎与 FasterTransformer **持平或略差**（因为不合批 Attention） |
| 引擎微基准 | 175B / 16 GPU，双方均关闭流水 | Orca 引擎**快至多 47%**，作者归因于控制面/数据面分离 |
| 端到端 | 175B / 16 GPU，输入长度 $U(32,512)$、生成长度 $U(1,128)$、Poisson 到达 | 中位归一化延迟 190 ms/token 处，FT 0.185 req/s vs Orca **6.81 req/s（36.9×）** |
| 端到端 | 101B / 8 GPU，低负载 | **唯一的例外**：低负载下两者都凑不满 batch，Orca 不占优 |
| 同构负载 | 175B，所有请求输入/生成长度相同 | FT 的 max_bs 终于开始有正收益；Orca 仍领先，但 `max_bs=1` 时退化 |

另有一个易被忽略的结果：FasterTransformer 在 13B 模型上**用不了 batch size ≥ 8**（101B 上 ≥ 16 就 OOM），因为它按模型最大序列长度 2048 为每个请求**固定预分配** KV 显存。Orca 则按每请求的 `max_tokens` 属性分配。

### 2.4 我的评价

这篇的价值不在于哪个算子写得快，而在于它**指出了一处抽象边界画错了**。Triton / TensorFlow Serving 那套「服务层负责调度、引擎层负责算」的分层，在单次前向就出结果的模型上是干净的；一旦模型变成多迭代，请求粒度的接口就成了性能天花板。论文的选择是**把这层边界拆掉**——调度器与引擎紧耦合。作者在 §7 坦承他们没有研究「如何在保住抽象分离的前提下支持这两个技术」，把它留给了未来工作。

这是系统论文里一个反复出现的模式：**新的工作负载特征让旧的抽象边界失效，第一篇论文往往先把边界砸掉换性能，抽象怎么重建是后话。** 今天 vLLM、SGLang、TensorRT-LLM 全都把调度器和引擎做在一起，这个边界至今没有重建起来。

评分 ⭐⭐⭐⭐⭐。iteration-level scheduling 已经是所有现代推理引擎的默认设定，continuous batching 这个术语就是从这里出来的。

---

## 3. 细读

> 章节编号与原文一致（原文共 8 节）。

### 3.1 Introduction

开篇的因果链很紧凑：生成任务重要 → Transformer 是主力架构 → 推理要交给独立的 serving service → 现有 serving 系统（Triton + FasterTransformer）在这类负载上不行。

不行的根源被归结为一句话：这些模型**要跑多次**才能处理完一个请求，而系统在**请求粒度**上调度。由此推出提前结束的请求返回不了、后到的请求排队时间被拉长。

论文在这里就把两个技术的名字与关系讲清楚了：iteration-level scheduling 是主张，selective batching 是让主张能落地的补丁。还有一处脚注值得注意——论文明确声明它用的是 Attention 的**窄定义**（不含 QKV Linear 与 Attn Out Linear），这直接决定了后面「Attention 不带参数」这个论证成不成立。

> **批注**：这个脚注不是学究气，它是全文最关键的定义之一。如果按宽定义（把 QKV Linear 算进 Attention），「Attention 没有模型参数所以不合批的损失很小」这句话立刻就不成立了。读系统论文时，**作者特意划定的术语边界通常正好卡在论证的承重墙上**。

### 3.2 Background

分两部分。

**GPT 的推理过程**。论文在这里立了全文的术语：一次 **iteration** = 跑完模型所有层、产出一个 token。首次迭代吃下全部输入 token 并产出第一个输出 token，称 **initiation phase**；之后每次吃一个 token 的迭代称 **increment phase**。（今天更常见的叫法是 prefill / decode，但 Orca 用的是这套词。）

接着解释为什么必须缓存 KV：Attention 要用到**所有前序 token** 的 key 与 value（causal masking 下所有前序 token 都参与）。不缓存就要每次迭代重算全部 KV。论文把这个技术溯源到 fairseq 的 **incremental decoding**，并指出 FasterTransformer、Megatron-LM 也都这么做。

Figure 1c 是这一节最有价值的一张图，它把 Transformer 与 LSTM 的状态使用模式并排放：**LSTM 的状态尺寸恒定（$c_{l,t}$、$h_{l,t}$），Transformer 的状态尺寸随迭代增长（$k_{l,1:t}$、$v_{l,1:t}$）**。因此 Attention 必须在形状随已处理 token 数变化的张量上做计算。

**ML 推理服务系统**。批处理是 GPU 利用率的关键：多请求的输入张量拼成一个大张量，加速器偏好大张量；此外还能**复用从片外显存加载的模型参数**，对访存密集的算子尤其重要。

> **批注**：Figure 1c 那个对比是全文的地基。「状态尺寸恒定 vs 随时间增长」这一条差别，同时解释了三件事：为什么 Attention 不能像 Linear 那样直接合批、为什么 KV 显存管理会成为独立难题（→ 后来的 [vLLM](../2023-vllm-pagedattention/README.md)）、以及为什么 BatchMaker 那套 RNN 的做法搬不过来（见 §3.7）。一张背景图能承载后续三条独立的论证线，这是很高的密度。

### 3.3 Challenges and Proposed Solutions

论文把挑战与方案交错编排（C1 → S1 → C2 → S2），读起来像一次现场推导。

**C1: Early-finished and late-joining requests**。见 §2.1。这里补一个细节：论文强调服务系统与引擎**只在两个时刻交互**——调度器给空闲引擎派下一批、或引擎跑完当前批。这个「只有两个交互点」的观察，就是后面把交互点改成「每次迭代」的靶子。

**S1: Iteration-level scheduling**。调度器循环：选请求 → 调引擎跑一次迭代 → 收结果。Figure 4 给了系统总览：Endpoint 收请求放进 **request pool**，Scheduler 监视 pool、选一组请求、调 Execution Engine 跑一次迭代、把返回的 token 追加回 pool 里对应的请求。请求结束时 pool 移除它并通知 endpoint 发响应。

论文用了一句很准确的话概括收益：**调度器对「每次迭代处理多少个、哪些请求」有了完全的控制权**。

**C2: Batching an arbitrary set of requests**。三种不能合批的情形（见 §2.2）。论文这里给了一个我认为很重要的定量判断：合批的前提是两请求同阶段、且输入 token 数相同（initiation）或 token 下标相同（increment），这**极大降低了真实负载中合批的概率，而且概率随 batch size 增大呈指数下降**，导致大 batch size 根本用不起来。

**S2: Selective batching**。见 §2.2。Figure 5 走查了 $(x_1,x_2,x_3,x_4)$ 这批共 7 个 token 的执行：非 Attention 算子上是 $[7,H]$ 的二维张量，QKV Linear 后变成 $[7,3H]$，经 Split 拆成 $[3,3H]$（$x_1$）、$[1,3H]$（$x_2$）、$[2,3H]$（$x_3$）、$[1,3H]$（$x_4$）逐请求做 Attention，再 Merge 回 $[7,H]$。

**Attention K/V manager** 在这里登场：它按请求分别保管已生成的 key/value，直到调度器**显式通知**该请求已结束才回收。increment 阶段的请求做 Attention 时，从 manager 取历史 KV，与 Split 出来的当前 token 的 Q/K/V 一起算。

> **批注**：selective batching 的分界线不是「哪个算子快」，而是「**哪个算子需要请求的概念**」。Linear/LayerNorm/Add/GeLU 是逐 token 独立的，把 batch 维拍扁不影响语义；Attention 必须知道 token 属于谁，否则会跨请求算注意力。这条判据可以直接迁移：**做变长合批时，先按「算子是否需要样本边界」分类，不需要的一律拍平**。今天的 varlen / packed 实现（FlashAttention 的 `cu_seqlens`）本质上是同一条思路的延续——只不过后来 Attention 也学会了在拍平的布局上用偏移量区分请求，不必再 Split/Merge。

### 3.4 Orca Design

#### 4.1 Distributed Architecture

Orca 组合了两种已有的并行策略，二者都源自训练系统：

- **Intra-layer parallelism**（层内并行，即[张量并行](../../../concepts/tensor-parallelism.md)）：把矩阵乘（Linear 与 Attention）及其参数切到多张 GPU 上。论文明确说**略过切分细节**，直接引用 Megatron-LM 与 Shoeybi 等人的工作。
- **Inter-layer parallelism**（层间并行，即流水并行）：把 Transformer 层切到不同 GPU，Orca 给每张 GPU 分**同样多**的层。

Figure 6 的例子：4 层模型切成 2 个层间分区，每个分区再切成 3 个层内分区，共 6 张 GPU。

引擎架构（Figure 7）是这一节的重点。每个 worker 进程负责一个层间分区，可以在不同机器上；每个 worker 管着若干个 CPU 线程，每线程驱动一张 GPU，线程数等于层内并行度。

执行流程：engine master 把「这次迭代的 token + 控制消息」发给 Worker1。**控制消息**含批内请求的 id、当前 token 下标（increment 阶段用）、输入 token 数（initiation 阶段用）。Worker1 的 controller 把信息交给各 GPU 线程，线程解析后向自己的 GPU 发射 kernel——例如 Attention kernel 用 request id 与当前 token 下标去 Attention K/V manager 查历史 KV 的显存地址。**与此同时，controller 不等本机 kernel 完成就把控制消息转发给下一个 worker**。只有最后一个 worker 才同步等待 kernel 完成，取出输出 token 回传 engine master。

这里有一个作者相当自豪的设计：**控制面与数据面分离**。论文观察到 FasterTransformer、Megatron-LM 每次收控制消息都会引发一次 CPU-GPU 同步，因为它们**用 NCCL（GPU-to-GPU 通道）传控制消息**；而控制消息每次迭代都要传，开销不可忽略。Orca 把两条通道拆开：NCCL 只传 GPU 产出、GPU 消费的中间张量，控制消息与 token 走 gRPC 这类不经过 GPU 的通道。

> **批注**：「控制消息不要走数据通道」这一条，收益在 §6.1 被单独量化了（175B 上快 47%），是全文最容易被迁移到其他系统的一条工程结论。它的一般形式是：**当控制信息被塞进为大块数据优化的通道时，你会为每一条小消息付一次同步代价**。判断方法也简单——看这条消息的生产者与消费者是不是 CPU；是的话就不该走 GPU 通道。

#### 4.2 Scheduling Algorithm

调度策略出人意料地简单：**保持先来先服务，不改变请求的处理顺序**。论文给出了迭代级 FCFS 的精确定义：对 pool 中任意一对请求 $(x_i, x_j)$，若 $x_i$ 比 $x_j$ 先到，则 $x_i$ 已跑的迭代数应**大于等于** $x_j$。（注意这不等于先到先返回：需要迭代更少的后到请求可能先完成。）

在 FCFS 之上叠加两个约束：

**约束一：max batch size（`max_bs`）**。理由是**批大小的边际收益递减**——batch 越大吞吐涨得越少，但延迟一直在涨。所以要有个上限，由运维按延迟预算调。

**约束二：GPU 显存**。中间结果的 buffer 可以跨算子立即复用，但 **Attention K/V manager 的 buffer 不能**——必须等调度器通知请求结束才能回收。论文明确点出一个风险：**朴素实现会死锁**——pool 里所有请求都因为「没地方存下一个 token 的 KV」而发不出去。

解法是**预留（reservation）**：请求首次被调度时，按它的 `max_tokens` 属性（每请求可配的上限，指处理完后最多有多少 token）一次性预留 `max_tokens` 个 slot；一个 slot 是存一个 token 的 K 和 V 所需的显存量。`n_slots` 是分配给 manager 的总槽位数，由运维配置。因为请求的 token 数不会超过 `max_tokens`，预留成功就保证它能一路跑到结束。

**Algorithm 1** 转写如下：

```text
Params: n_workers（worker 数）, max_bs（最大批大小）, n_slots（K/V 槽位总数）

 1  n_scheduled ← 0
 2  n_rsrv ← 0
 3  while true do
 4      batch, n_rsrv ← Select(request_pool, n_rsrv)
 5      调度引擎为 batch 跑一次迭代
 6      foreach req in batch do
 7          req.state ← RUNNING
 8      n_scheduled ← n_scheduled + 1
 9      if n_scheduled = n_workers then
10          等待某个已调度批次返回
11          foreach req in 返回的批次 do
12              req.state ← INCREMENT
13              if finished(req) then
14                  n_rsrv ← n_rsrv − req.max_tokens
15          n_scheduled ← n_scheduled − 1

17  def Select(pool, n_rsrv):
18      batch ← {}
19      pool ← { req ∈ pool | req.state ≠ RUNNING }
20      SortByArrivalTime(pool)
21      foreach req in pool do
22          if batch.size() = max_bs then break
23          if req.state = INITIATION then
24              new_n_rsrv ← n_rsrv + req.max_tokens
25              if new_n_rsrv > n_slots then break
26              n_rsrv ← new_n_rsrv
27          batch ← batch ∪ {req}
28      return batch, n_rsrv
```

论文特别说明：`n_slots` 比 `max_bs` **好配得多**。`max_bs` 需要实测延迟/吞吐折中，而 `n_slots` 在模型规格与并行度确定后，显存占用几乎只取决于它——直接取显存允许的最大值即可。

**Pipeline parallelism**。第 9–10 行是流水的关键：调度器**不等**返回，直到在飞批次数 `n_scheduled` 达到 `n_workers`。这样每个 worker 手上总有一个批在处理，不会空转。Figure 8a：3 个 worker、`max_bs=2`，调度器先后注入 AB、CD、EF 三批，然后才等 AB 返回；AB 回来后 A、B 因为到达最早又被重新选中。

对比 FasterTransformer（Figure 8b）：请求级调度不允许在当前批结束前注入新批，所以它只能把一批**切成 microbatch** 来填流水。这就产生了一个 Orca 没有的两难——**microbatch 越大合批效率越高，但越大流水气泡越多**。论文这里的措辞很到位：Orca「free of such a tradeoff」。

> **批注**：预留式显存管理是这篇最经得起推敲、也最先被推翻的一处设计。它对了一半：作者**正确识别了死锁风险**，也**正确判断了 `n_slots` 比 `max_bs` 好配**。但按 `max_tokens` 预留意味着按**最坏情况**占显存——请求实际只生成 10 个 token，也要占着 `max_tokens` 个槽位直到结束。一年后 vLLM 正是拿这一点开刀：把「一次性预留连续的最坏情况空间」换成「按需分配定长块」，量出来的有效利用率只有 20.4%–38.2%。
>
> 值得说清楚的是：Orca 已经比 FasterTransformer 好了一档（后者按模型最大序列长度 2048 预分配，Orca 按每请求 `max_tokens`），vLLM 是在 Orca 的基础上又走了一步。**这是一条清晰的演进链：固定最大长度 → 按请求上限预留 → 按需分页**。

### 3.5 Implementation

13K 行 C++，基于 CUDA 生态。控制面 gRPC，数据面 NCCL（层间与层内通信都用）。支持原始 encoder-decoder Transformer、GPT 以及 Raffel 等人讨论的变体。

融合 kernel 方面：LayerNorm、Attention、GeLU 都做了融合——Attention 的 QK 点积、Softmax、加权平均融进单个 CUDA kernel。此外论文还多走了一步：**把 Split 出来的多个 Attention kernel 再融成一个**，做法是把不同请求的 kernel 的 thread block 直接拼接。作者承认这让同一 kernel 内的 thread block 特征与生命周期都不同（「often discouraged by CUDA programming practice」），但实测有利：提升 GPU 利用率、减少 kernel 启动开销。

> **批注**：这个「反最佳实践」的融合，正是 selective batching 的实现代价被压下去的地方。Split 逐请求算 Attention 在概念上很干净，但字面实现会变成 batch size 个小 kernel 依次启动，启动开销直接吃掉合批的收益。把它们拼成一个 kernel 相当于**在 kernel 内部重建了批的概念**——这也解释了为什么 §6.1 里 Orca 引擎「只是略差于」FasterTransformer 而不是差很多。

### 3.6 Evaluation

#### 6.1 Engine Microbenchmark

**目的是把调度器摘出去**，单独比引擎。做法：不跑 Orca 调度器，测试脚本反复把**同一批**请求注入引擎直到全部结束，模拟请求级调度的行为；批内所有请求输入 token 数相同（32 或 128）、生成 32 个 token。测的是处理整批的时间。

结果（Figure 9）：

- **13B / 1 GPU**（Figure 9a）：Orca 与 FT **持平或略差**。论文归因明确——Orca 不合批 Attention，FT 合批所有算子。但差距小，因为 Attention 不带参数，合批拿不到复用权重的好处。
- **101B / 8 GPU**（Figure 9b）：结论类似。作者据此下结论：两者在 CUDA kernel 实现与层内分区通信上**效率相当**。
- **175B / 16 GPU**（Figure 9c，双方均关闭流水，FT 的 microbatch size 设为等于 batch size）：Orca **快至多 47%**，归因于控制面/数据面分离。341B 因结果类似而省略。

**显存侧的观察**：FT 在 13B 上 batch size ≥ 8、101B 上 ≥ 16 会 OOM，因为它按 max 序列长度 2048 为每个请求固定预分配 KV。Orca 按每请求 `max_tokens` 分配，避开了这份冗余。

> **批注**：这一节的实验设计值得单独学。作者主动构造了一个**对自己不利**的场景——关掉调度器，只比引擎，结果是持平或略差。这反而让 §6.2 的 36.9× 可信度大幅提升：既然引擎本身不占便宜（175B 上的 47% 除外），端到端的巨大差距就只能来自调度。**把自己的核心贡献隔离掉再测一次，是归因实验的标准做法**。

#### 6.2 End-to-end Performance

**负载合成**。没有公开的生成式语言模型请求 trace，所以自己合成：输入 token 数采样自 $U(32,512)$，`max_gen_tokens` 采样自 $U(1,128)$（两者之和即 `max_tokens`），到达时间按 Poisson 过程，通过调 Poisson 参数改变负载。

有一个必须记住的简化：**假定所有请求都一直生成到 `max_gen_tokens`，模型永不吐 `<EOS>`**。作者给的理由是他们既没有真实的模型 checkpoint 也没有真实输入文本，无从判断 `<EOS>` 该在何时出现。

FT 没有自带调度器，作者为它实现了一个：从队列取至多 max batch size 个请求动态组批——即 Triton、TensorFlow Serving 的常见做法。

**结果（Figure 10）**，报的是中位端到端延迟**按生成 token 数归一化**（ms/token）与吞吐：

- **101B / 8 GPU**（Figure 10a）：**低负载下是唯一的例外**——两者都凑不满 batch，延迟主要由引擎性能决定（即 Figure 9b）。负载加重后 Orca 拉开差距，论文的说法很形象：Orca 调度器让后到的请求「hitch a ride」搭上正在跑的批。FT 峰值吞吐只有 **0.49 req/s**。
- **175B / 16 GPU、341B / 32 GPU**（Figure 10b/c，均有多于一个层间分区）：Orca 在**所有负载水平**上延迟与吞吐双赢。头条数字出自这里：为匹配 175B 的中位归一化延迟 **190 ms**（这个值是 Figure 9c 中 `orca(128)` 归一化执行时间的两倍），FT 吞吐 **0.185 req/s**，Orca **6.81 req/s**，即 **36.9×**。

**批大小配置的影响**。Orca 增大 `max_bs` 提吞吐而**不影响延迟**——因为迭代级调度已经消解了早结束/晚加入的问题。论文自己加了限定：不保证在任意硬件、模型、负载下增大 batch 都不伤延迟，`max_bs` 仍须按延迟与吞吐需求谨慎设定。

FT 这边反直觉：**增大 max batch size 不一定提吞吐**。穷举所有 (`max_bs`, `mbs`) 组合后，最优的是 (1,1) 或 (8,8)。原因有两层：(a) microbatch 流水下引擎实际只按 `mbs` 合批，所以 `mbs` 取到最大（等于 `max_bs`）才好；(b) `max_bs` 越大，越容易把输入/输出长度差异大的请求凑到一批，而 FT 在这种批上表现很差——**第一次迭代按批内最短的输入长度处理所有请求**，且早结束的请求无法立即返回。

**同构请求 trace**（Figure 11，175B）。所有请求输入长度与 `max_gen_tokens` 都相同，早结束问题不复存在。此时 FT 的 `max_bs` 终于有明显正收益。Orca 仍然领先（对比 FT `max_bs=8`），**唯一的例外是 Orca 自己 `max_bs=1` 的配置**——此时 Orca 退化成一条不做合批的 worker 流水线。

> **批注**：`max_bs` 增大而延迟不变，这个结果初看很反直觉——批大了单步不是该更慢吗？答案在归一化上：报的是 **ms/token**，批增大确实让单次迭代变慢，但它同时消灭了大量为已结束请求做的无效计算，两者相抵。这也提示了这套机制的收益边界：**收益主要来自「消除无效计算与排队」，而不是「让每步算得更快」**。所以负载越同构、请求长度越接近，收益越小——Figure 11 就是这个论断的直接证据。

#### 6.2 附：实验条件速查

| 参数 | 取值 |
| --- | --- |
| 硬件 | Azure ND96asr A100 v4，8× A100 40GB / VM，NVLink；机间 8× 200Gbps HDR IB（1.6 Tb/s），最多 4 VM |
| 模型 | 13B（40 层 / hidden 5120 / 1×1）、101B（80 / 10240 / 1 inter × 8 intra）、175B（96 / 12288 / 2×8）、341B（120 / 15360 / 4×8） |
| 精度 | fp16（参数与中间激活） |
| 最大序列长度 | 2048（沿用 GPT-3 论文设定） |
| 基线 | NVIDIA FasterTransformer（＋作者自实现的调度器） |
| 端到端负载 | 输入 $U(32,512)$、生成 $U(1,128)$、Poisson 到达、永不吐 `<EOS>` |

### 3.7 Related Work and Discussion

**RNN 的细粒度批处理：BatchMaker**。论文把 BatchMaker 认作最相关的前作：它在 **RNN cell** 粒度上调度与合批。因为 RNN 每个 cell 做的计算完全相同，无论 token 下标是多少都能合批，所以新请求可以随时加入正在执行的批。

但这套搬不到 Transformer 上，理由很硬：Transformer 里**不同 token 下标 $t$ 的 cell 是不同的 cell**（每个要用不同的 Attention K/V 集合），所以图里有 $L$ 种互不相同的 cell（$L$ 为输入加生成的 token 数）。同一时刻恰好存在多个「同一种 cell」的概率极低——论文给了 Figure 10 里的实测范围：$L$ 从 33（=32+1）到 640（=512+128）。结果是 BatchMaker 会退化成串行执行。此外它也不支持需要模型并行与流水并行的大模型。

论文顺势把自己的设计原则讲出来了，这句话是全文的题眼：

> 我们设计 Orca 的核心原则是**每读一轮模型参数就尽可能多算**（perform as much computation as possible per each round of model parameter read），因为对大模型而言，从 GPU global memory 读参数是端到端时间的主要瓶颈。

正是这条原则让 iteration-level scheduling 与 selective batching 统一起来：**一轮参数读里处理掉所有「就绪」的 token，不管这些 token 能不能合批**（非 Attention 能，Attention 不能）。

**Transformer 专用引擎**。FasterTransformer、LightSeq、TurboTransformers、EET 都是 Triton / TF Serving 的后端引擎，都把调度权交给服务层，因而都停留在请求级调度。其中只有 FasterTransformer 支持分布式执行。Megatron-LM、DeepSpeed 虽然也能分布式，但主要为训练优化。

**服务系统与引擎之间的接口**。这一段是作者的自我审视：现有分层（Triton、Clipper 作为抽象层）有其好处——服务层与执行层可以独立设计实现。但对 GPT 这种多迭代模型，**这个接口太受限了**。Orca 的选择是把调度器与引擎紧耦合，以简化两个技术的落地。作者明确写道：本文**不研究**如何在不丢失抽象分离的前提下支持这两个技术，留给未来工作。

> **批注**：BatchMaker 那段对比是我认为全文写得最好的一处。它没有停在「前作不行」，而是精确指出**不行在哪个性质上**——RNN 的 cell 与位置无关，Transformer 的 cell 与位置强相关（因为 KV）。这条性质差异同时解释了 BatchMaker 为什么搬不过来、以及 Orca 为什么必须发明 selective batching：**既然按 cell 合批的路被 Attention 堵死了，那就只在没被堵死的算子上合批**。§3.2 Figure 1c 埋的伏笔在这里收线。

### 3.8 Conclusion

简短复述两个技术与结论：同等延迟下比当时的 SOTA 高一个数量级的吞吐。

---

## 4. 关键问题解析

### Q1: iteration-level scheduling 相比 request-level batching 具体省掉了什么？

省掉三样，且三样性质不同：

**1. 为已结束请求做的无效计算**。请求级调度下，批内请求需要的迭代数不同，先结束的请求在剩余迭代里仍被算。Figure 3 的例子：$x_2$ 在 iter 2 结束，引擎在 iter 3、iter 4 仍为它计算（论文用 `-` 标注这些「extra computation」）。迭代级调度下它在结束的那一刻就被移出。

**2. 已结束请求的返回延迟**。这是**纯粹的等待**，不是计算浪费。请求级调度下引擎只在整批结束时才向服务层返回，所以 $x_2$ 生成完 `<EOS>` 后还要在引擎里躺到 $x_1$ 也结束。迭代级调度下调度器每次迭代都拿到返回，能立刻放它走。

**3. 新请求的排队时间**。请求级调度下新请求要等当前整批结束；迭代级调度下最多等**一次迭代**就有机会入选。

三者中，**(2) 和 (3) 是延迟收益，(1) 是吞吐收益**。Figure 10 里 Orca 增大 `max_bs` 能提吞吐而不伤延迟，正是因为 (1) 省下的算力抵消了大批带来的单步变慢。

反过来说，这也界定了收益的边界：如果批内所有请求**同时开始、同时结束**，(1)(2) 都不存在，只剩 (3)。Figure 11 的同构 trace 实验证实了这一点——那种负载下 FT 增大 `max_bs` 也开始有正收益了。

### Q2: selective batching 是什么？为什么 Attention 不能像 Linear 那样直接合批？

**机制**：不把整个模型「批化」，只对一部分算子合批。具体地：

- **非 Attention 算子**（Linear、LayerNorm、Add、GeLU）：把批内所有请求的 token 拉平成 $[\sum L, H]$ 的**二维**张量，没有显式的 batch 维，做 **token 级**合批。
- **Attention**：前插 Split、后插 Merge，逐请求单独计算。

**为什么 Attention 不行**，要分两层说：

**第一层（语义）**：Attention 需要「请求」这个概念，因为它只能在**同一请求内部**的 token 之间计算注意力。如果像 Linear 那样把 batch 维拍平，不同请求的 token 会互相算注意力，结果是错的。而 Linear、LayerNorm、GeLU 都是**逐 token 独立**的——每个 token 的输出只取决于自己，所以完全不需要知道 token 属于谁，拍平不改变语义。

**第二层（形状）**：即使按传统方式保留 batch 维，Attention 的输入形状也对不齐。increment 阶段处理 token 下标 $t$ 的请求要吃进 $k_{l,1:t-1}$、$v_{l,1:t-1}$，形状随 $t$ 变化（§3.2 Figure 1c）。不同请求的 $t$ 不同，就凑不成一个规则的 $[B, \cdot, \cdot]$ 张量。传统实现靠 cuBLAS 的 batch matmul，那要求各样本形状一致。

**为什么这个妥协可以接受**：因为 **Attention 不带模型参数**。合批的两大收益是「喂给 GPU 更大的张量」和「复用已加载的权重」，对于不带参数的算子，第二项直接消失。论文用的正是这个论证，§6.1 的微基准（Orca 引擎仅持平或略差于全合批的 FT）是它的实测证据。

注意这个论证依赖 §3.1 那个脚注里的**窄定义**：QKV Linear 与 Attn Out Linear **不算** Attention 的一部分。它们带参数，而且它们仍然是合批的。

### Q3: 调度器如何决定每次迭代放哪些请求？`max_bs` 与 `n_slots` 如何确定？

**选取规则**（Algorithm 1 的 `Select`）：

1. 从 pool 里剔除 `state = RUNNING` 的请求（正在被某个在飞批次处理）。
2. **按到达时间排序**——这是 iteration-level FCFS 的落点。
3. 顺序扫描，遇到以下任一条件就停：批已满 `max_bs`；或当前请求处于 INITIATION 阶段且预留后 `n_rsrv > n_slots`。
4. INITIATION 阶段的请求入选时，按其 `max_tokens` 预留槽位并累加 `n_rsrv`。INCREMENT 阶段的请求不需要新预留（首次调度时已经一次性留够）。

**迭代级 FCFS 的定义**：对任意 $(x_i, x_j)$，$x_i$ 先到则 $x_i$ 跑过的迭代数 $\geq x_j$。论文提醒这**不等于**先到先返回——需要更少迭代的后到请求可能先完成。

**两个参数的调法完全不同**，论文明确对比了：

- **`max_bs` 难调**。它是延迟与吞吐的折中点，理由是**批大小的边际收益递减**：批越大吞吐涨得越少，延迟却持续上升。必须实测，由运维按延迟预算定。
- **`n_slots` 好调**。模型规格（hidden size、层数）与层内/层间并行度确定后，Orca 的显存占用几乎只取决于 `n_slots`，所以**直接取显存允许的最大值**即可，不需要任何实验。

第 22 行 `break` 而非 `continue` 值得注意：一旦某个请求因预留不足被挡住，扫描**直接终止**，不会跳过它去找后面预留量更小的请求。这是 FCFS 的严格贯彻，代价是可能损失一些装箱效率。

### Q4: Orca 如何管理 KV Cache 的显存分配？

论文的原文依据集中在 §4.2 与 §5，共四点：

**1. Attention K/V manager 是独立组件，且回收由调度器显式驱动**。§3.3 原文：manager「按请求分别保管这些 key 与 value，直到调度器显式要求移除某个请求的 key/value，即该请求处理完成之时」。这与中间结果的 buffer 形成对比——后者可以跨算子立即复用，前者不行。

**2. 预留式分配，粒度是 slot**。§4.2 原文：请求首次被调度时，用它的 `max_tokens` 属性**预先**预留 `max_tokens` 个 slot；一个 slot 定义为「存一个 token 的 Attention key 与 value 所需的显存量」。因为请求的 token 数不会超过 `max_tokens`，预留成功即保证它能一路跑到结束不再需要额外分配。

**3. 动机是防死锁，不是省显存**。§4.2 原文明确：朴素实现「会让调度器陷入死锁」——pool 里所有请求都因为没地方存下一个 token 的 KV 而发不出去。预留机制是为了消除这个风险。

**4. 相比 FasterTransformer 已是改进**。§6.1 原文：FT 为每个请求**按模型最大序列长度**（此处 2048）固定预分配，所以 13B 上 batch size ≥ 8、101B 上 ≥ 16 就 OOM；Orca「根据 `max_tokens` 属性为每个请求分别设定 buffer 大小」，避开了这份冗余分配。

**这正是 vLLM 批评的落点，需要说清是在批评什么**。Orca 的分配是**按请求上限、一次性、连续**的。若某请求 `max_tokens = 640` 但实际只生成了 10 个 token，那 630 个 slot 从它被调度到它结束为止都被占着且不可他用。vLLM 论文量化的 20.4%–38.2% 有效利用率就是在打这一点。

但要注意**演进关系而非对错**：FT 按模型最大长度 2048 预分配 → Orca 按请求 `max_tokens` 预留 → vLLM 按需分定长块。三步中每一步都在缩小「预留量」与「实际用量」的差距，Orca 是中间那一步。另外，Orca 论文根本没把显存效率当作自己的贡献点，它只需要显存管理**不出死锁、不成为调度的瓶颈**即可。

### Q5: 分布式执行部分与 Megatron 式张量并行是什么关系？

**关系是「直接复用，不做贡献」**。论文 §4.1 原文说 Orca「composes known parallelization techniques」，intra-layer parallelism 引用的正是 Megatron-LM 的工作，并明确写道「**我们略过这个策略如何切分每个矩阵乘的细节**」。

所以要分清三层：

| 层次 | 来自哪里 | Orca 的贡献 |
| --- | --- | --- |
| 层内并行（张量并行）怎么切矩阵 | Megatron-LM 等训练系统 | 无，直接用 |
| 层间并行（流水）怎么切层 | 训练系统 | 极小——只规定「每张 GPU 分同样多的层」 |
| 多 worker 的**控制流与流水编排** | — | **这里才是 Orca 的贡献** |

Orca 真正新的东西有两条：

1. **控制面/数据面分离**。FasterTransformer 与 Megatron-LM 用 NCCL 传控制消息，导致每次收消息都有一次 CPU-GPU 同步，而控制消息每次迭代都要传。Orca 让 NCCL 只传 GPU 间的中间张量，控制消息与 token 走 gRPC。§6.1 里 175B 上 47% 的提升就归因于此。
2. **无需 microbatch 的流水**。请求级调度不能在当前批结束前注入新批，所以 FT 必须把一批切成 microbatch 来填流水，从而陷入「microbatch 大则合批效率高、小则流水气泡少」的两难。Orca 靠迭代级调度直接注入多个**独立批次**（Algorithm 1 第 9–10 行，在飞批次数保持等于 `n_workers`），绕开了这个折中。

对本仓库其他笔记的意义：CacheRoute 里「一个 destination 是一个 TP2 组」中的 TP 就是这里的 intra-layer parallelism，具体切法要读 [Megatron-LM 笔记](../../llm-training/2019-megatron-lm/README.md)；Orca 只告诉你它被当作既定基础设施使用。

### Q6: 这套调度在什么负载下收益最小？

论文自己给出了两个实测的边界，我再补两个推断的。

**论文实测的：**

**1. 低负载**（Figure 10a，101B / 8 GPU）。论文原文承认这是「唯一的例外」：负载低时 Orca 与 FT 都凑不满一个批，延迟主要由引擎单批性能决定（即 Figure 9b 的结果，那里 Orca 略差于 FT）。**没有排队就没有排队可省**——迭代级调度的三项收益里，(2)(3) 在低负载下几乎不发生。

**2. 同构负载**（Figure 11，175B，所有请求输入长度与 `max_gen_tokens` 相同）。所有请求需要的迭代数相同，早结束问题**根本不存在**。此时 FT 增大 `max_bs` 开始有明显正收益，Orca 的相对优势收窄。极端情况是 Orca 自己 `max_bs=1`——论文承认此时 Orca「退化成一条不做合批的 worker 流水线」，表现不如 FT `max_bs=8`。

**我的推断（论文未测）：**

**3. 生成长度极短的负载**。若绝大多数请求只生成 1–2 个 token（例如分类任务改写成生成式），那么每个请求的生命周期就只有一两次迭代，「批内成员随迭代变化」这件事几乎没有发生的余地，迭代级调度退化为请求级。论文的 trace 里 `max_gen_tokens ~ U(1,128)`，均值约 64，落在对 Orca 有利的区间。

**4. 显存极度受限、`max_tokens` 又设得很大的负载**。Algorithm 1 第 25 行的预留检查会先于 `max_bs` 触发 `break`，实际批大小被显存卡死在一个很小的值。这时限制吞吐的是显存而不是调度，改进调度收益有限——**这正是 vLLM 一年后要解决的问题**。论文没有做这个方向的敏感性实验（既没有扫 `n_slots`，也没有扫 `max_tokens` 分布），是评测上的一处空白。

### Q7: Orca 与 vLLM 是什么关系？读完这篇再读那篇会有什么不同？

**两者正交，且 vLLM 论文自己是这么定位的**：Orca 通过**调度**让更多请求能同时在批，vLLM 通过**显存管理**让更多请求装得下。

更准确地说，二者是**因果相继**的：迭代级调度让批的成员每一步都在变化，这恰恰**加剧**了显存管理的难度——请求以不可预测的顺序加入与离开，连续预分配的碎片问题被放大。所以 Orca 越成功，分页式管理就越必要。

读完 Orca 再读 vLLM，有三处会读出不同的味道：

1. **vLLM 的批评对象具体了**。「已有系统预分配连续显存，有效利用率仅 20.4%–38.2%」——现在你知道被批评的是 Algorithm 1 第 23–26 行按 `max_tokens` 的预留，也知道 Orca 这么做是为了防死锁而不是偷懒。
2. **「Orca (Oracle)」这个基线看得懂了**。vLLM 论文构造了一个「假设预先知道输出长度」的 Orca 变体作为上界基线。知道 Orca 是按 `max_tokens` 预留后就明白：这个 oracle 变体等于把预留量从上限收紧到真实值，是在剥离「预留过量」这一个因素。
3. **continuous batching 的出处清楚了**。vLLM 论文 §2.3 讲的批处理技术就是 Orca 的迭代级调度，vLLM 把它当既有背景全盘继承。

详见 [vLLM / PagedAttention 笔记](../2023-vllm-pagedattention/README.md) 与 [分页式 KV 管理](../../../concepts/paged-kv-memory.md)。

### Q8: 为什么这套机制在训练时不需要？

论文 §3 结尾专门点了这句：早结束/晚加入的问题「**在语言模型的训练中不会发生**」，因为训练用 **teacher forcing**——整批在**一次迭代**内就处理完了。

展开说：训练时下一个位置的输入是**数据集里的真实 token**，不是模型上一步的输出，所以整个序列的所有位置可以并行计算，一次前向就得到全部位置的 loss。推理没有这个便利：第 $t+1$ 步的输入必须等第 $t$ 步真的算出来，序列依赖是硬的。

**「多迭代」这个性质是推理独有的，这正是为什么训练系统的那套批处理假设搬到推理上会失效**——而当时的推理服务系统（Triton、TF Serving）恰恰是按「一次前向出结果」的 ResNet/BERT 类模型设计的。论文 §1 就是从这个对比切入的。

这也解释了为什么 Orca 能直接复用 Megatron 的并行策略却必须重做调度：**并行策略处理的是「一次前向怎么切到多卡」，训练与推理没有本质差别；调度处理的是「多次前向之间怎么编排」，这在训练里根本不存在。**

---

## 5. 可迁移的知识点

- [Continuous Batching](../../../concepts/continuous-batching.md) —— 本文是 iteration-level scheduling 的**出处**，该页的一手细节（selective batching 机制、Algorithm 1 的调度策略、实验数字）均来自本文。
- [KV Cache](../../../concepts/kv-cache.md) —— 本文 §2 Figure 1c 给出了 Transformer 与 LSTM 的状态对比，是理解「为什么 KV 状态尺寸随迭代增长」最清楚的一张图。
- [分页式 KV 管理](../../../concepts/paged-kv-memory.md) —— 本文的 `max_tokens` 预留式分配是分页方案要取代的前一代做法，两者构成清晰的演进链。
- [张量并行](../../../concepts/tensor-parallelism.md) —— 本文 §4.1 的 intra-layer parallelism 直接复用自 Megatron-LM，本文不展开切分细节。
- [SLO 容量与 p99 尾延迟](../../../concepts/slo-capacity.md) —— 本文用「中位归一化延迟（ms/token）vs 吞吐」曲线来表达容量，是这类系统的标准画法；但只报中位数不报尾延迟是它的一处缺陷（见 §6.2）。

**跨领域可迁移的三条：**

1. **调度粒度应当匹配工作负载的最小可决策单元**。负载从「一次前向」变成「多次迭代」后，请求粒度的接口就成了天花板。找到「系统在什么时刻能重新做决定」，往往就找到了性能瓶颈。
2. **控制消息不要走为大块数据优化的通道**。判据是看消息的生产者与消费者是不是 CPU；是的话走 GPU 通道就会为每条小消息付一次同步代价（本文实测 47%）。
3. **做变长合批时，先按「算子是否需要样本边界」分类**。不需要的一律拍平成 $[\sum L, H]$，需要的单独处理。这条比「哪个算子更快」是更本质的分界线。

---

## 6. 批判与开放问题

### 6.1 作者承认的局限

1. **抽象边界被牺牲了，且没有重建方案**（§7）。Orca 把调度器与引擎紧耦合，放弃了 Triton / Clipper 那套「服务层与执行层独立设计」的好处。作者明确写道：本文**不研究**如何在保住抽象分离的前提下支持这两个技术，「留给未来工作」。

2. **不合批 Attention 确实有性能代价**（§6.1）。13B / 1 GPU 与 101B / 8 GPU 的微基准上，Orca 引擎相比全合批的 FasterTransformer「持平或**略差**」。作者论证代价小（Attention 不带参数），但没有否认代价存在。

3. **低负载下没有优势**（§6.2）。101B / 8 GPU 低负载是论文自陈的「唯一例外」——两者都凑不满批，延迟由引擎性能决定，而引擎侧 Orca 略差。

4. **`max_bs` 增大不伤延迟这个结论不保证普适**（§6.2）。作者主动加了限定：不保证在任意硬件、模型、负载下成立，`max_bs` 仍需按延迟与吞吐需求谨慎设定。

5. **`max_bs=1` 时退化**（§6.2 同构 trace）。此时 Orca 变成一条不做合批的 worker 流水线，不如 FT `max_bs=8`。

6. **实验只覆盖语言模型**（§1）。作者声称方法适用于任何基于 Transformer 且自回归生成的模型（图像、视频、语音），但「只在语言模型上做了实验」。

### 6.2 我的质疑

1. **合成 trace 里「模型永不吐 `<EOS>`」这个假设，方向上对 Orca 有利，论文没有讨论其影响**。§6.2 明确写了：假定所有请求都生成到 `max_gen_tokens`，不提前结束。理由（没有真实 checkpoint 与输入文本）是诚实的，但后果需要看清——真实负载里请求会在 `max_gen_tokens` **之前**随机结束，这会同时影响两侧：对 FT 而言早结束的请求更多，无效计算更严重（**对 Orca 更有利**）；对 Orca 而言按 `max_tokens` 预留的过量更严重（**对 Orca 不利**）。两个方向哪个占优取决于负载，论文既没做敏感性分析也没讨论。**36.9× 这个数字应当理解为在特定合成负载下的结果**。〔推断，论文未讨论〕

2. **只报中位延迟，尾延迟完全缺席**。Figure 10、11 报的都是 median latency。但 Orca 严格的 FCFS + 第 22 行 `break` 的组合，理论上可能让某个 `max_tokens` 很大的请求长期占着预留、阻塞后面的请求扫描。这种效应恰恰只在 p95/p99 上显现。对一篇服务系统论文而言，**没有尾延迟数据是一处实打实的评测缺口**。〔推断：阻塞机制是我从 Algorithm 1 推的，论文没有讨论也没有测量〕

3. **FasterTransformer 的调度器是作者自己实现的，公平性无法核验**。§6.2 说 FT 没有自带调度器，所以作者「实现了一个自定义调度器」，做法是从队列取至多 max batch size 个请求动态组批。作者称这是 Triton、TF Serving 的常见做法，这个说法可信；但**基线的调度器由被比较方实现**，且论文没有给出这个实现的细节或代码。考虑到 36.9× 的差距主要来自调度而非引擎（§6.1 已证明引擎侧基本持平），基线调度器的质量**直接决定了头条数字**。这是全文最需要独立复现来验证的一点。

4. **`n_slots` 与 `max_tokens` 两个参数完全没有敏感性实验**。论文声称 `n_slots` 好配（取显存上限即可），这个论证在逻辑上成立，但 §4.2 的预留机制意味着 **`max_tokens` 的分布直接决定了实际能达到的批大小**。端到端实验固定用 $U(32,512) + U(1,128)$ 一种分布，没有扫过 `max_tokens` 更大或方差更大的情形。而这正是一年后 vLLM 攻击的靶心——**Orca 自己没有测量这个薄弱点**。

5. **175B 上 47% 的提升归因于控制面/数据面分离，但没有做消融**。§6.1 原文是「we attribute this performance improvement to the control-data plane separation」——attribute（归因）不是 measure（测量）。要坐实这个归因，应该做一个「Orca 引擎但控制消息走 NCCL」的变体来对比。论文没做。考虑到这是全文最容易被其他系统直接借鉴的一条工程结论，缺消融比较可惜。〔推断：论文只给了归因表述，未给消融数据〕

6. **13K 行 C++ 的系统未开源，全部结果不可复现**。这在 2022 年的系统论文里已经不算常态。FriendliAI 后续把它做成了商业产品，这个背景让不开源可以理解，但不改变「论文的所有数字都无法被第三方核验」这个事实。相比之下 vLLM 一年后开源，很大程度上决定了两者在工业界的实际影响路径不同——**Orca 贡献了思想，vLLM 贡献了实现**。

### 6.3 开放问题

1. **抽象边界能否重建？** 作者自己留下的问题。有没有一个通用的 serving/engine 接口，既支持迭代级调度又保住分层？今天的现状是没有——vLLM、SGLang、TensorRT-LLM 都是紧耦合的。这可能说明这个边界**本来就不该存在**，也可能只是还没有人找到正确的抽象。

2. **严格 FCFS 是不是对的？** Orca 选择不改变处理顺序，好处是没有饥饿、公平性清晰。但这放弃了所有基于长度或 SLO 的调度机会（短请求优先能大幅改善平均延迟；按 SLO 分级能保尾延迟）。后续工作（如 FastServe 的抢占式调度、各类优先级调度）正是在这个方向展开的。

3. **selective batching 的分界线会移动吗？** 当年 Attention 因为形状不齐必须 Split/Merge。今天的 varlen kernel（FlashAttention 的 `cu_seqlens`）已经能在拍平的布局上用偏移量区分请求，不必物理拆分。这说明**「哪些算子需要样本边界」是随 kernel 能力变化的工程边界，不是固定的数学边界**。

### 6.4 后续可读

- **vLLM / PagedAttention（SOSP'23）** —— 直接攻击本文的预留式显存分配。见 [笔记](../2023-vllm-pagedattention/README.md)。
- **Sarathi-Serve（OSDI'24）** —— chunked prefill，处理本文没碰的「initiation 阶段的长请求打断 increment 阶段」问题。
- **DistServe / Splitwise** —— 把 initiation 与 increment 分离到不同实例，从根本上回避两阶段混排的取舍。
- **FastServe** —— 抢占式调度，直接挑战本文的严格 FCFS。
- **Megatron-LM（arXiv 2019）** —— 本文 §4.1 直接复用其层内并行，本文略过的切分细节在那里。见 [笔记](../../llm-training/2019-megatron-lm/README.md)。
