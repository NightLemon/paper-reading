# Concepts · 知识点索引

跨论文复用的概念。每篇独立成文，读完不依赖任何一篇具体论文；论文笔记通过链接引用它们，概念页也反向列出出现过它的论文。

新增概念时从仓库根目录的 `templates/concept.md` 复制，文件名用 kebab-case。

## 索引

| 概念 | 别名 | 相关主题 | 出现在 |
| --- | --- | --- | --- |
| [Self-Attention](self-attention.md) | 自注意力、intra-attention | Transformer 架构 | Attention Is All You Need · RoFormer / RoPE · Speculative Decoding |
| [Multi-Head Attention](multi-head-attention.md) | 多头注意力、MHA | Transformer 架构 | Attention Is All You Need · GQA |
| [Positional Encoding](positional-encoding.md) | 位置编码、PE | Transformer 架构 | Attention Is All You Need · RoFormer / RoPE |
| [残差连接与 Layer Normalization](residual-and-layer-norm.md) | Post-LN、Pre-LN | 训练稳定性 | Attention Is All You Need |
| [学习率 Warmup](lr-warmup.md) | Noam schedule | 优化与调度 | Attention Is All You Need |
| [KV Cache](kv-cache.md) | KV 缓存 | 推理与显存 | Attention Is All You Need · vLLM · CacheRoute · Mooncake · Sarathi-Serve · GQA · RoFormer / RoPE · Speculative Decoding |
| [分页式 KV 管理](paged-kv-memory.md) | PagedAttention、block table | 显存管理 | vLLM |
| [Continuous Batching](continuous-batching.md) | 连续批处理、iteration-level scheduling | 调度 | Orca · vLLM · Sarathi-Serve · Speculative Decoding |
| [Prefix Caching](prefix-caching.md) | 前缀缓存、prefix KV reuse | LLM Serving | vLLM · SGLang · CacheRoute · Mooncake |
| [约束解码](constrained-decoding.md) | Constrained Decoding、结构化输出 | 解码与工程集成 | SGLang |
| [Prefill / Decode 分离](prefill-decode-disaggregation.md) | PD Disaggregation、阶段分离 | 服务架构 | DistServe · Mooncake · Sarathi-Serve |
| [KVCache 池化与分层存储](disaggregated-kv-store.md) | Disaggregated KV Store、KVCache Pool | 服务架构与存储 | Mooncake · CacheRoute |
| [缓存亲和性与负载均衡的取舍](cache-affinity-vs-load-balance.md) | locality-load tradeoff | 路由与调度 | CacheRoute · Mooncake |
| [SLO 容量与 p99 尾延迟](slo-capacity.md) | SLO capacity、goodput | 性能评测 | CacheRoute · Sarathi-Serve |
| [IO-aware 算法设计](io-aware-kernel-design.md) | IO-Awareness、tiling | GPU 性能 | FlashAttention |
| [张量并行](tensor-parallelism.md) | Tensor Parallelism、TP | 分布式训练 | Megatron-LM · Sarathi-Serve · GQA · ZeRO · GPipe / PipeDream |
| [共识与复制状态机](consensus-and-replication.md) | Consensus、quorum | 分布式系统 | Raft |
| [Chunked Prefill](chunked-prefill.md) | 分块预填充、chunked-prefills、prefill chunking | LLM Serving | Sarathi-Serve · DistServe |
| [Grouped-Query Attention](grouped-query-attention.md) | GQA、分组查询注意力、MQA、Multi-Query Attention | Foundations | GQA |
| [Rotary Position Embedding](rotary-position-embedding.md) | RoPE、旋转位置编码 | Foundations | RoFormer / RoPE |
| [Speculative Decoding](speculative-decoding.md) | 投机解码、speculative sampling、draft-and-verify | LLM Serving | Speculative Decoding |
| [ZeRO 状态分片](zero-state-sharding.md) | Zero Redundancy Optimizer、optimizer state sharding、ZeRO-1/2/3 | LLM Training | ZeRO · GPipe / PipeDream |
| [激活检查点](activation-checkpointing.md) | activation checkpointing、rematerialization | 训练显存与重算 | ZeRO · GPipe / PipeDream |
| [流水并行](pipeline-parallelism.md) | Pipeline Parallelism、PP、microbatch pipeline、1F1B、weight stashing | LLM Training | GPipe / PipeDream |

[← 返回总索引](../index.md)
