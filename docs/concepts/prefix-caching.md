---
concept: "Prefix Caching"
aliases: ["前缀缓存", "prefix KV reuse", "automatic prefix caching"]
tags: ["llm-serving", "inference", "caching"]
papers: ["topics/llm-serving/2026-cacheroute"]
---

# Prefix Caching

> **一句话定义**：当两个请求拥有相同的前缀 token 序列时，第二个请求直接复用第一个请求留下的 KV，跳过这段前缀的 prefill。

## 为什么需要它

真实负载中大量请求共享前缀：

- 系统提示词与工具定义（几乎所有请求共享）
- 多轮会话中此前的对话历史（同一会话内共享）
- 每租户/每业务的固定上下文（同一租户的所有请求共享）
- few-shot 示例块

这些 token 每次都做一遍 prefill 是纯粹的重复劳动。prefill 是算力受限的，直接决定 TTFT（time to first token），也就是 p99 尾延迟的主要来源。

## 它是怎么工作的

### 命中条件

复用要求**精确的前缀匹配**：从第 0 个 token 开始逐位置相同，且必须连续。中间任何一个 token 不同，从该位置往后的 KV 全部作废。

因此前缀的**排列顺序**是可优化的：把最稳定的内容（系统提示、租户上下文）放在最前，把易变内容（检索结果、当前问题）放在后面，能显著提高可复用的长度。

### 常见实现

- **Radix Tree / 前缀树索引**（SGLang 的 RadixAttention）：把已缓存前缀组织成基数树，新请求沿树匹配最长公共前缀。
- **Block 哈希**（vLLM 的 automatic prefix caching）：按固定大小 block 对 token 序列滚动哈希，用哈希值索引已有的 KV block，天然支持共享与写时复制。

### 三个前置条件

1. **前缀确实重复**：可复用段在 prompt 中占比足够大。
2. **回来得够快**：两次访问的间隔短于该前缀的有效驱逐时间。
3. **回到同一台机器**：KV 是有位置的状态，缓存只存在于处理过第一个请求的那台机器上。

第三条在单机语境下是隐含成立的，在集群里由**路由策略**决定，而缓存本身管不了它。

## 关键性质与代价

| 收益 | 代价 |
| --- | --- |
| 跳过重复 prefill，直接降低 TTFT | 占用 KV 显存，与可服务并发数竞争 |
| 命中越多，单请求工作量越小，单机吞吐越高 | 精确前缀匹配，易变内容会打断复用 |
| 实现完全在引擎内部，对上层透明 | 收益上限由「可复用段 / 整个 prompt」的比例封顶 |

**收益上限这一条最容易被高估**。举例：某负载 prompt 平均 1.2K token，其中全局模板在每台机器上本来就是热的（路由与否都命中），检索块在 67% 的相邻请求对之间就变了（几乎不可能命中），真正与路由 key 绑定且可复用的只有约 180 token（15%）。此时任何路由优化能争取的最大收益就是这 15%。

## 常见误解

- **误解**：prefix caching 打开就有收益 → **实际**：收益 = 可复用段占比 × 命中率。可复用段很小时，围绕它做的优化可能得不偿失。
- **误解**：集群越大，缓存总容量越大，命中率越高 → **实际**：在 cache-blind 均衡下，同一 key 回到同一台机器的平均间隔按 $R/\lambda_b$ 放大，**扩容会让 prefix 变冷**。
- **误解**：把可复用内容放在 prompt 任何位置都行 → **实际**：必须是前缀。稳定内容排在前面才可复用。
- **误解**：命中率是优化目标 → **实际**：命中率是中间量，目标是 SLO 下的容量。见 [SLO 容量与 p99 尾延迟](slo-capacity.md)。

## 出现在哪些论文里

- [vLLM / PagedAttention](../topics/llm-serving/2023-vllm-pagedattention/README.md) —— §4.4 的 shared prefix 是它最早的系统化实现：服务方预先为一组声明的前缀保留物理块，类比 OS 的共享库。需要手动声明是它的主要缺口。
- [SGLang / RadixAttention](../topics/llm-serving/2024-sglang-radixattention/README.md) —— 补上了那个缺口：把 KV Cache 当成**基数树上的 LRU 缓存**，请求结束后不丢弃、自动做最长前缀匹配，并用「最长匹配前缀优先」的调度把命中率推到理论最优的 96%。
- [CacheRoute](../topics/llm-serving/2026-cacheroute/README.md) —— 指出 prefix caching 的第三个前置条件（回到同一台机器）由路由决定，并把它规划成一个周期性的分配问题；同时给出可复用段太小时该优化净亏损的实测反例。

## 延伸阅读

- [SGLang RadixAttention](../topics/llm-serving/2024-sglang-radixattention/README.md)：基数树 + 叶子优先 LRU + cache-aware 调度的完整实现。
- Mooncake / MemServe：把 KV 池化或跨机迁移，绕开「必须回到同一台机器」这个约束。
- [KV Cache](kv-cache.md) · [分页式 KV 管理](paged-kv-memory.md) · [缓存亲和性与负载均衡的取舍](cache-affinity-vs-load-balance.md)
