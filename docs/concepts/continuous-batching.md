---
concept: "Continuous Batching"
aliases: ["连续批处理", "iteration-level scheduling", "in-flight batching", "动态批处理"]
tags: ["llm-serving", "scheduling", "batching"]
papers: ["topics/llm-serving/2023-vllm-pagedattention"]
---

# Continuous Batching

> **一句话定义**：把批处理的调度粒度从「一整个请求」降到「一次解码迭代」——每步结束后移出已完成的请求、加入新到达的请求，而不是等整批请求全部结束。

## 为什么需要它

批处理是提升 LLM 推理吞吐的基础手段：batch 内所有请求共享同一份权重，权重从显存搬到片上的成本被摊薄。由于 decode 阶段是访存受限的，batch 越大，单请求分摊到的权重搬运成本越低。

但请求级（request-level）批处理有两个硬伤：

- **到达时间不同**。朴素做法要么让先到的请求等后到的，要么让后到的请求等当前批结束，两者都产生显著排队延迟。
- **长度差异巨大**。同一批里输入输出长度可能相差一个数量级。对齐长度需要 padding，浪费算力与显存；不 padding 则整批要等最长的那个请求跑完，短请求的槽位在剩余时间里空转。

## 它是怎么工作的

调度单位从 request 变成 **iteration**（一次解码步）：

```
每次迭代结束后：
  1. 把已生成结束符的请求移出 batch，释放其资源
  2. 从等待队列取新请求填入空出的槽位
  3. 组装下一次迭代的输入（prefill 请求的全部 token + decode 请求的最新 1 个 token）
```

由此得到两个直接效果：

- 新请求最多等待**一次迭代**即可开始被处理，而非整批结束。
- 提前结束的请求立刻让出槽位，不再占着位置空转。

配合能处理变长序列的 kernel，padding 也一并消除。

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
| 吞吐大幅提升 | 长 prompt 的 prefill 会打断 decode，制造延迟毛刺 |

**它与显存管理正交**：continuous batching 决定「哪些请求同时在批」，分页式 KV 管理决定「这些请求的 KV 能不能装得下」。前者提高了对后者的要求——batch 组成频繁变化会让连续预分配的碎片问题更严重。两者叠加才能同时拿到调度与容量两方面的收益。

## 常见误解

- **误解**：continuous batching 就是「动态调整 batch size」→ **实际**：核心是**调度粒度**降到迭代级，batch 成员在每一步都可以变化。
- **误解**：它能替代显存优化 → **实际**：它决定调度自由度，可批量的上限仍由 KV 显存决定。两者互补。
- **误解**：batch 越大越好 → **实际**：batch 增大会抬高单步时延（TPOT），对交互式场景是负面的；同时 prefill 与 decode 的混排会制造延迟毛刺。
- **误解**：它对所有阶段一视同仁 → **实际**：prefill 是 compute-bound、decode 是 memory-bound，把两者放进同一次迭代本身就是一个需要专门处理的取舍。

## 出现在哪些论文里

- [vLLM / PagedAttention](../topics/llm-serving/2023-vllm-pagedattention/README.md) —— 把它作为既有背景（出自 Orca），并明确说明 PagedAttention 与它是**互补**的：Orca 通过调度让更多请求并行，vLLM 通过提高显存利用率让更多请求装得下；细粒度调度反而让显存管理更困难，因此分页更关键。

## 延伸阅读

- Orca: A Distributed Serving System for Transformer-Based Generative Models（OSDI'22）—— iteration-level scheduling 的出处。
- Sarathi-Serve：chunked prefill，缓解 prefill 打断 decode 的问题。
- DistServe / Splitwise：把 prefill 与 decode 分离到不同实例，从根本上回避混排取舍。
- [分页式 KV 管理](paged-kv-memory.md) · [SLO 容量与 p99 尾延迟](slo-capacity.md)
