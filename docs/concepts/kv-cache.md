---
concept: "KV Cache"
aliases: ["KV 缓存", "Key-Value Cache"]
tags: ["llm-serving", "inference", "memory"]
papers: ["topics/foundations/2017-attention-is-all-you-need", "topics/llm-serving/2026-cacheroute"]
---

# KV Cache

> **一句话定义**：自回归解码时把每一层每个已处理位置的 key 与 value 张量缓存下来，使生成第 $t$ 个 token 时只需为这一个新位置计算 $K$、$V$，而不必重算整个前缀。

## 为什么需要它

Transformer decoder 的因果 mask 保证位置 $i$ 只依赖 $\le i$ 的位置。由此得到一个关键性质：

> 位置 $j$ 的 $K_j$、$V_j$ 只依赖位置 $\le j$ 的输入，**新生成的 token 不会改变它们**。

若不缓存，生成长度为 $n$ 的序列需要做 $n$ 次前向，第 $t$ 次处理长度为 $t$ 的序列，总注意力计算是 $O(n^3 d)$ 量级。缓存之后每步只处理一个新位置，总量降到 $O(n^2 d)$，且每步的计算从「重算前缀」变成「读取前缀」。

## 它是怎么工作的

解码被划分成两个阶段，两者的资源特征完全不同：

| 阶段 | 处理内容 | 瓶颈 | 产物 |
| --- | --- | --- | --- |
| **Prefill** | 一次性处理整个 prompt（$n$ 个 token） | 算力（大矩阵乘） | 写入 $n$ 个位置的 KV |
| **Decode** | 每步一个新 token | 访存（读权重 + 读 KV Cache） | 追加 1 个位置的 KV |

每步 decode 的读取量随已生成长度线性增长，因此长上下文场景下 decode 阶段是**访存受限**的。

### 容量估算

$$\text{KV bytes} = 2 \times L \times H_{kv} \times d_{\text{head}} \times n \times \text{bytes\_per\_elem}$$

其中 $L$ 层数，$H_{kv}$ K/V 的头数，$n$ 序列长度，前面的 2 对应 K 和 V 各一份。这个量与 batch 内的每条序列成正比，因此**并发数直接被显存约束**。

### 压缩这一项的常见手段

- **MQA / GQA**：减小 $H_{kv}$（所有 query 头共享一组 K/V，或分组共享）。
- **量化**：把 KV 存成 fp8 / int8，减小 `bytes_per_elem`。
- **分页管理（PagedAttention）**：把 KV 切成固定大小的 block，按需分配，消除为最大长度预留造成的内部碎片。

## 关键性质与代价

| 收益 | 代价 |
| --- | --- |
| 每步 decode 从重算前缀降为读取前缀 | 显存占用与「并发 × 序列长度」成正比 |
| 使跨请求复用（prefix caching）成为可能 | decode 阶段由算力受限转为访存受限 |
| 前缀不变性使缓存内容可长期保留 | 缓存内容绑定在**具体某台机器**上，路由会影响它的价值 |

最后一行是从单机走向集群时最容易被忽略的一条：KV Cache 是**有位置的状态**。一旦请求被路由到另一台机器，这份缓存对它就不存在。

## 常见误解

- **误解**：KV Cache 缓存的是模型权重 → **实际**：缓存的是每个位置的激活值（$K$、$V$），随请求而异，随并发线性增长。
- **误解**：query 也需要缓存 → **实际**：$Q$ 只用于当前这一步的计算，用完即弃。
- **误解**：KV Cache 让长上下文变便宜 → **实际**：它让*重复计算*变便宜，显存占用与访存量仍随长度线性增长。
- **误解**：缓存命中率越高，系统吞吐越高 → **实际**：命中率是集群平均量，而 SLO 容量由尾延迟决定。见 [缓存亲和性与负载均衡的取舍](cache-affinity-vs-load-balance.md)。

## 出现在哪些论文里

- [Attention Is All You Need](../topics/foundations/2017-attention-is-all-you-need/README.md) —— 未讨论推理优化，但其 decoder 的因果 mask 与自回归性质给出了 KV 可缓存的全部依据。
- [vLLM / PagedAttention](../topics/llm-serving/2023-vllm-pagedattention/README.md) —— 把 KV Cache 从「实现细节」提升为「决定服务吞吐的一等资源」，并给出 800 KB/token 这类可直接复用的容量估算方法。
- [CacheRoute](../topics/llm-serving/2026-cacheroute/README.md) —— 把「KV Cache 是有位置的状态」这一性质当作路由问题来处理：请求落在哪台机器决定了缓存能否被用上。

## 延伸阅读

- [分页式 KV 管理](paged-kv-memory.md)：KV 的分页分配与共享。
- MQA / GQA：压缩 K/V 头数。
- [Prefix Caching](prefix-caching.md)：跨请求复用 KV 的形式。
