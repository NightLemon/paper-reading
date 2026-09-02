# Concepts · 知识点索引

跨论文复用的概念。每篇独立成文，读完不依赖任何一篇具体论文；论文笔记通过链接引用它们，概念页也反向列出出现过它的论文。

新增概念时从仓库根目录的 `templates/concept.md` 复制，文件名用 kebab-case。

## 索引

| 概念 | 别名 | 相关主题 | 出现在 |
| --- | --- | --- | --- |
| [Self-Attention](self-attention.md) | 自注意力、intra-attention | Transformer 架构 | Attention Is All You Need |
| [Multi-Head Attention](multi-head-attention.md) | 多头注意力、MHA | Transformer 架构 | Attention Is All You Need |
| [Positional Encoding](positional-encoding.md) | 位置编码、PE | Transformer 架构 | Attention Is All You Need |
| [残差连接与 Layer Normalization](residual-and-layer-norm.md) | Post-LN、Pre-LN | 训练稳定性 | Attention Is All You Need |
| [学习率 Warmup](lr-warmup.md) | Noam schedule | 优化与调度 | Attention Is All You Need |
| [KV Cache](kv-cache.md) | KV 缓存 | 推理与显存 | Attention Is All You Need · vLLM · CacheRoute |
| [分页式 KV 管理](paged-kv-memory.md) | PagedAttention、block table | 显存管理 | vLLM |
| [Continuous Batching](continuous-batching.md) | 连续批处理、iteration-level scheduling | 调度 | vLLM |
| [Prefix Caching](prefix-caching.md) | 前缀缓存、prefix KV reuse | LLM Serving | vLLM · CacheRoute |
| [缓存亲和性与负载均衡的取舍](cache-affinity-vs-load-balance.md) | locality-load tradeoff | 路由与调度 | CacheRoute |
| [SLO 容量与 p99 尾延迟](slo-capacity.md) | SLO capacity、goodput | 性能评测 | CacheRoute |
| [IO-aware 算法设计](io-aware-kernel-design.md) | IO-Awareness、tiling | GPU 性能 | 🚧 待接单（FlashAttention） |
| [张量并行](tensor-parallelism.md) | Tensor Parallelism、TP | 分布式训练 | 🚧 待接单（Megatron-LM） |
| [共识与复制状态机](consensus-and-replication.md) | Consensus、quorum | 分布式系统 | 🚧 待接单（Raft） |

[← 返回总索引](../index.md)
