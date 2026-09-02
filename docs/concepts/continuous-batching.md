---
concept: "Continuous Batching"
aliases: ["连续批处理", "iteration-level scheduling", "in-flight batching", "动态批处理"]
tags: ["llm-serving", "scheduling", "batching"]
papers: ["topics/llm-serving/2022-orca", "topics/llm-serving/2023-vllm-pagedattention"]
---

# Continuous Batching

> **一句话定义**：把批处理的调度粒度从「一整个请求」降到「一次解码迭代」——每步结束后移出已完成的请求、加入新到达的请求，而不是等整批请求全部结束。

## 为什么需要它

批处理是提升 LLM 推理吞吐的基础手段：batch 内所有请求共享同一份权重，权重从显存搬到片上的成本被摊薄。由于 decode 阶段是访存受限的，batch 越大，单请求分摊到的权重搬运成本越低。

但请求级（request-level）批处理有两个硬伤：

- **到达时间不同**。朴素做法要么让先到的请求等后到的，要么让后到的请求等当前批结束，两者都产生显著排队延迟。
- **长度差异巨大**。同一批里输入输出长度可能相差一个数量级。对齐长度需要 padding，浪费算力与显存；不 padding 则整批要等最长的那个请求跑完，短请求的槽位在剩余时间里空转。

根源在于**这类负载的「多迭代」性质**：处理一个请求要跑模型很多次，每次只产出一个 token。而 ResNet、BERT 这类模型一次前向就出结果，为它们设计的服务系统（Triton、TensorFlow Serving）与引擎之间的接口天然是请求粒度的——调度器交出一批，引擎跑完整批才返回。

值得注意的是**这个问题是推理独有的**。训练时用 teacher forcing，下一位置的输入是数据集里的真实 token 而非模型上一步的输出，所以整个序列可以并行计算，一次迭代就处理完整批。「多迭代」在训练里根本不存在，因此训练系统的批处理假设搬到推理上会失效。

## 它是怎么工作的

调度单位从 request 变成 **iteration**（一次跑完模型所有层、产出一个 token）：

```
每次迭代结束后：
  1. 把已生成结束符的请求移出 batch，释放其资源
  2. 从等待队列取新请求填入空出的槽位
  3. 组装下一次迭代的输入（prefill 请求的全部 token + decode 请求的最新 1 个 token）
```

由此得到三个直接效果，性质各不相同：

| 省掉了什么 | 类型 |
| --- | --- |
| 为已结束请求做的无效计算（它们本来要陪跑到整批结束） | 吞吐收益 |
| 已结束请求的返回等待（本来要躺到整批结束才能出去） | 延迟收益 |
| 新请求的排队时间（从「等整批」降到「等一次迭代」） | 延迟收益 |

配合能处理变长序列的 kernel，padding 也一并消除。

### selective batching：让形状不齐的请求也能合批

迭代级调度带来一个新麻烦：被同时选中的请求彼此形状不齐，无法整体合批。有三种情形合不了批：

1. 两者都在 prefill 阶段但输入 token 数不同；
2. 两者都在 decode 阶段但处理的 token 下标不同（KV 张量形状不同）；
3. 一个在 prefill、一个在 decode（一个吃全部输入 token，一个只吃 1 个）。

传统合批要求「同阶段 + 输入 token 数相同（prefill）或 token 下标相同（decode）」。这个条件在真实负载里很难满足，而且**满足的概率随 batch size 增大呈指数下降**——大 batch 因此根本用不起来。

**selective batching** 的解法是按算子区别对待，分界线是「**这个算子需不需要知道 token 属于哪个请求**」：

- **不需要的**（Linear、LayerNorm、Add、GeLU）：这些算子逐 token 独立，每个 token 的输出只取决于它自己。于是把批内所有请求的 token 拉平成 $[\sum L, H]$ 的**二维**张量，去掉显式 batch 维，做 **token 级**合批。语义完全不变。
- **需要的**（Attention）：注意力只能在同一请求内部的 token 之间计算，拍平会让不同请求的 token 互相算注意力，结果是错的。于是前插 Split、后插 Merge，逐请求单独算。

举例：一批 4 个请求共 7 个 token，非 Attention 算子上走 $[7,H]$；QKV Linear 后变成 $[7,3H]$，Split 成 $[3,3H]$、$[1,3H]$、$[2,3H]$、$[1,3H]$ 逐请求做 Attention，再 Merge 回 $[7,H]$。

**为什么不合批 Attention 的代价可以接受**：合批的两大收益是「喂 GPU 更大的张量」与「复用已加载的权重」，而 **Attention 不带模型参数**，第二项直接消失。注意这里用的是 Attention 的**窄定义**——QKV Linear 与 Attn Out Linear 不算在内，它们带参数且仍然合批。

> 这条分界线是工程边界而非数学边界，会随 kernel 能力移动。今天的 varlen kernel（如 FlashAttention 的 `cu_seqlens`）已能在拍平的布局上用偏移量区分请求，不必再物理 Split/Merge。

### 调度策略：选谁进这一批

一个具体的做法（Orca 的 Algorithm 1）：

1. 剔除正在被某个在飞批次处理的请求；
2. **按到达时间排序**，保证迭代级 FCFS——对任意 $(x_i,x_j)$，$x_i$ 先到则 $x_i$ 跑过的迭代数 $\geq x_j$；
3. 顺序扫描，遇到「批已满 `max_bs`」或「KV 槽位预留不足」就停；
4. 首次被调度的请求，按其 `max_tokens` 一次性预留 KV 槽位。

两个参数的调法差别很大：

- **`max_bs`（最大批大小）难调**。它是延迟与吞吐的折中点，因为**批大小的边际收益递减**——批越大吞吐涨得越少，延迟却持续上升。必须实测。
- **KV 槽位总数好调**。模型规格与并行度确定后，显存占用几乎只取决于它，直接取显存允许的最大值即可。

注意「迭代级 FCFS」**不等于**先到先返回：需要迭代更少的后到请求可能先完成。

**防死锁是预留机制的动机**。KV 的 buffer 与中间结果的 buffer 不同——后者可跨算子立即复用，前者必须等请求结束才能回收。不做预留的朴素实现会死锁：pool 里所有请求都因为「没地方存下一个 token 的 KV」而发不出去。

### 与 prefill 的关系

一次迭代里同时存在两类请求：处于 prefill 阶段的（一次输入整个 prompt）和处于 decode 阶段的（每次输入一个 token）。如何把两者组织进同一次迭代，是这套机制的主要实现难点，也衍生出后续的调度策略分支：

- **prefill 优先**：新请求尽快拿到首 token，TTFT 好，但会打断正在 decode 的请求，抬高 TPOT。
- **decode 优先**：正在生成的请求更平滑，但新请求的 TTFT 变差。
- **chunked prefill**：把长 prompt 的 prefill 切成小块，与 decode 混排，平衡两者。

## 关键性质与代价

| 收益 | 代价 |
| --- | --- |
| 排队延迟从「等整批」降到「等一次迭代」 | 调度器每步都要运行，CPU 侧开销随 batch 增大 |
| 消除 padding 浪费 | 需要支持变长序列的自定义 kernel |
| 提前结束的请求立即让出资源 | batch 组成不断变化，显存管理显著变难 |
| 消除为已结束请求做的无效计算 | Attention 无法合批（但因其不带参数，代价小） |
| 吞吐大幅提升 | 长 prompt 的 prefill 会打断 decode，制造延迟毛刺 |
| 无需 microbatch 即可填满流水 | 调度器与执行引擎必须紧耦合，牺牲分层抽象 |

**它与显存管理正交**：continuous batching 决定「哪些请求同时在批」，分页式 KV 管理决定「这些请求的 KV 能不能装得下」。前者提高了对后者的要求——batch 组成频繁变化会让连续预分配的碎片问题更严重。两者叠加才能同时拿到调度与容量两方面的收益。

**它对流水并行的额外好处**：请求级调度不允许在当前批结束前注入新批，所以要填满多个层间分区的流水，只能把一批切成 microbatch，从而陷入「microbatch 大则合批效率高、小则流水气泡少」的两难。迭代级调度可以直接注入多个**独立批次**（保持在飞批次数等于 worker 数），绕开这个折中。

### 收益边界：什么时候它不划算

- **低负载**。凑不满批就没有排队可省，延迟主要由引擎单批性能决定。此时不合批 Attention 的代价反而显出来。
- **同构负载**。若所有请求输入长度与生成长度都相同，它们同时开始、同时结束，「早结束/晚加入」根本不存在，请求级批处理已经够用。
- **生成长度极短**。若请求只生成 1–2 个 token，请求生命周期就一两次迭代，「批内成员随迭代变化」几乎没有发生余地，迭代级调度退化为请求级。
- **`max_bs=1`**。退化成一条不做合批的流水线。

一句话概括：**收益主要来自「消除无效计算与排队」，而不是「让每步算得更快」**。负载越同构，收益越小。

## 常见误解

- **误解**：continuous batching 就是「动态调整 batch size」→ **实际**：核心是**调度粒度**降到迭代级，batch 成员在每一步都可以变化。
- **误解**：它能替代显存优化 → **实际**：它决定调度自由度，可批量的上限仍由 KV 显存决定。两者互补，而且迭代级调度让显存管理**更**难，因此更需要分页。
- **误解**：batch 越大越好 → **实际**：批大小的边际收益递减——批越大吞吐涨得越少，延迟却持续上升。对交互式场景抬高 TPOT 是负面的。
- **误解**：它对所有阶段一视同仁 → **实际**：prefill 是 compute-bound、decode 是 memory-bound，把两者放进同一次迭代本身就是一个需要专门处理的取舍。
- **误解**：Attention 不合批会带来明显性能损失 → **实际**：Attention 不带模型参数，合批拿不到「复用已加载权重」的好处，实测代价很小（见下方 Orca 的微基准）。
- **误解**：它在所有负载上都有大幅收益 → **实际**：低负载、同构负载、极短生成长度这三种情形下收益都很有限，见上文「收益边界」。

## 出现在哪些论文里

- [Orca（OSDI'22）](../topics/llm-serving/2022-orca/README.md) —— **出处**。提出 iteration-level scheduling 与 selective batching，本页的机制细节均来自该文。实验条件：GPT-3 175B、16 张 A100 40GB（Azure ND96asr A100 v4，NVLink + 1.6Tb/s InfiniBand）、fp16、输入长度 $U(32,512)$、生成长度 $U(1,128)$、Poisson 到达、基线为 NVIDIA FasterTransformer 加作者自实现的动态组批调度器。在中位归一化延迟 190 ms/token 处，FasterTransformer 吞吐 0.185 req/s，Orca 6.81 req/s，即 **36.9×**。
    - 补充两个界定收益边界的数字：**引擎微基准**（关掉调度器，13B/1 GPU 与 101B/8 GPU，批内请求输入 token 数相同、均生成 32 token）下 Orca 引擎相比 FasterTransformer **持平或略差**——说明端到端的巨大差距来自调度而非 kernel；**101B/8 GPU 低负载**是论文自陈的唯一例外，Orca 不占优。
- [vLLM / PagedAttention（SOSP'23）](../topics/llm-serving/2023-vllm-pagedattention/README.md) —— 把它作为既有背景（明确注明出自 Orca），并说明 PagedAttention 与它是**互补**的：Orca 通过调度让更多请求并行，vLLM 通过提高显存利用率让更多请求装得下；细粒度调度反而让显存管理更困难，因此分页更关键。vLLM 量出已有系统（含 Orca 的按 `max_tokens` 预留）的 KV 有效利用率仅 20.4%–38.2%。

## 延伸阅读

- Sarathi-Serve（OSDI'24）：chunked prefill，缓解 prefill 打断 decode 的问题。
- DistServe / Splitwise：把 prefill 与 decode 分离到不同实例，从根本上回避混排取舍。
- FastServe：抢占式调度，挑战严格 FCFS。
- BatchMaker（EuroSys'18）：RNN 上的 cell 级细粒度批处理，是 Orca 的最相关前作。因为 RNN 的 cell 与位置无关而 Transformer 的 cell 与位置强相关（每个 token 下标要用不同的 KV），这套做法搬不到 Transformer 上。
- [分页式 KV 管理](paged-kv-memory.md) · [KV Cache](kv-cache.md) · [SLO 容量与 p99 尾延迟](slo-capacity.md)
