---
concept: "Chunked Prefill"
aliases: ["分块预填充", "chunked-prefills", "prefill chunking"]
tags: ["llm-serving", "scheduling", "batching", "tail-latency"]
papers: ["topics/llm-serving/2024-sarathi-serve", "topics/llm-serving/2024-distserve"]
---

# Chunked Prefill

> **一句话定义**：把一个长 prompt 的 prefill 拆到多轮计算，后续块通过 KV Cache 读取前面块的状态，使每轮处理的新 token 数可以受控。

## 为什么需要它

连续批处理允许请求在每轮加入和退出。但新请求的 prompt 可能很长，直接插入整个 prefill 会拉长当前轮，使已有请求迟迟得不到下一个输出 token。限制请求个数无法限制这份工作量：一个请求也可能包含很长的 prompt。

分块让调度器在 prompt 尚未全部处理完时重新获得调度机会，从而控制 prefill 对 decode 的干扰。它也提供了一个更接近计算量的预算单位：**本轮处理的 token 数**。

## 它是怎么工作的

### chunk 边界切分进度，KV 保留上下文

对于因果 Transformer，前面位置的结果不依赖后面位置。先计算前一块，保留其各层 KV；下一块的 query 读取所有需要的历史 KV，并对块内位置应用 causal mask。

```text
prompt: [A B | C D | E F]
第 1 轮: 计算 A B，缓存 KV(A B)
第 2 轮: 计算 C D，读取 KV(A B)，追加 KV(C D)
第 3 轮: 计算 E F，读取 KV(A B C D)，追加 KV(E F)
```

E 可以关注 A–E，F 可以关注 A–F。块大小为 2 不意味着注意力窗口只有 2。调度器需记录已完成的 prompt 长度；通常只有整个 prompt 的 prefill 完成后，才能生成对应的首个输出 token。

### 与 decode 混批

一种策略是先安排运行中的 decode，再用 prefill chunks 填满剩余预算。若 decode 请求数为 $D$、每个请求本轮一个 token、chunk 长度为 $c_j$、预算为 $\tau$：

$$D+\sum_jc_j\le\tau.$$

例如预算 512、64 个 decode 请求，可为 prefill 安排至多 448 个新位置。该示例用于解释计数，与特定模型的性能无关；实际还受 KV 显存和最大活跃请求数限制。

小 decode batch 的线性层通常受权重读取限制。混入一些 prefill token 可以共享权重加载、提高算术强度，而不必按新增 token 数同比增加时间。这个机会存在到计算、attention 或通信开始主导为止，不能解释成零开销。

### 预算怎么选

先对模型、硬件、并行配置与有代表性的上下文长度做 profiling，观察每轮 token 数对 TBT 和 prefill 效率的影响，再选择满足延迟约束的工作区间。矩阵乘 tile 对齐和流水批次间运行时差异也会影响结果。

预算数的是本轮新位置，无法完整表达读取历史 KV 的成本。相同 token 数但上下文很不一样的两个批次，运行时间仍可能不同。在线预算调整是额外的调度设计，不能从分块机制本身直接推出。

## 关键性质与代价

| 收益 | 代价与条件 |
| --- | --- |
| 限制长 prefill 对一轮 decode 的影响 | 较小块会损失部分 prefill 效率，首 token 可能更晚 |
| 可以利用 decode 的算力余量 | 余量随 batch、上下文、模型和硬件变化 |
| 让流水批次的计算量更接近 | 等 token 数仍不保证等执行时间 |
| 保留完整因果上下文 | 后续块要重新读取历史 KV，增加跨 kernel 的访存 |
| 同实例即可组织两阶段 | 需要支持带历史 KV 的多 token attention kernel，并维护部分 prefill 状态 |

若一个 prompt 分为 $N$ 块，第一块 KV 被后续 $N-1$ 块读取，第二块被后续 $N-2$ 块读取，依次类推。这个计数说明重读机制；实际 HBM 字节数还受 cache、attention kernel 和块大小影响。分块没有把精确注意力的二次计算量变成线性。

## 常见误解

- **误解**：分块等于把上下文截短 → **实际**：块间通过 KV 保持依赖，注意力窗口是独立的模型选择。
- **误解**：chunk size 越小越好 → **实际**：需要平衡 TBT、TTFT、kernel 开销和计算效率。
- **误解**：stall-free 保证 decode 完全不变慢 → **实际**：它避免长 prefill 独占执行进度，混合迭代本身仍可能更长。
- **误解**：预算等于 batch size → **实际**：前者计 token，后者通常计请求；一个 prefill 请求可贡献许多 token。
- **误解**：分块与分页式 KV 管理是同一件事 → **实际**：分块决定何时计算，分页决定 KV 如何分配与寻址，两者可以组合。
- **误解**：分块一定优于 PD 分离 → **实际**：二者提供不同的调度与资源安排，比较需要统一硬件预算、网络和 SLO。

## 出现在哪些论文里

- [Sarathi-Serve](../topics/llm-serving/2024-sarathi-serve/README.md) —— 组合 chunked-prefills、decode 优先的 stall-free batching 和 token budget，并分析流水均匀性。§5.4 Figure 14 在 Yi-34B / 两张 A100、TP2、所测 prompt 长度下，512-token chunk 相对完整 prefill 的额外时间最高约 25%，2048-token chunk 则接近无额外开销。
- [DistServe](../topics/llm-serving/2024-distserve/README.md) —— 将同实例分块作为相关设计讨论；其引文指向 SARATHI 2023 前作。该对照帮助识别“控制干扰”与“分离资源”的不同自由度。

## 延伸阅读

- [Continuous Batching](continuous-batching.md) · [KV Cache](kv-cache.md) · [Prefill / Decode 分离](prefill-decode-disaggregation.md)。
- [SARATHI（2023）](https://arxiv.org/abs/2308.16369)：分块和 decode piggybacking 的前作。
- [SLO 容量](slo-capacity.md)：以满足延迟约束的可持续到达率评估调度，而非只比较峰值吞吐。
