# 待读队列 Reading List

这是一条**面向新人的学习路径**，按依赖关系分阶段排列。每篇都标注了发表场合、影响力层级和推荐理由。

**用法**：把想读的那行 `[ ]` 改成 `[x]`，然后说「精读 XXX」，我会拉原文、建目录、写完整报告。也可以直接说「按路径继续」，我从当前阶段的第一篇未读论文开始。

**影响力标注**：
`⭐⭐⭐` 领域奠基，不读会有知识断层 · `⭐⭐` 高影响力，业界广泛引用/落地 · `⭐` 值得一读，视兴趣取舍

---

## 阶段 0 · 已完成

| 论文 | 场合 | 笔记 |
| --- | --- | --- |
| Attention Is All You Need | NeurIPS'17 | [笔记](topics/foundations/2017-attention-is-all-you-need/README.md) |
| vLLM / PagedAttention | SOSP'23 | [笔记](topics/llm-serving/2023-vllm-pagedattention/README.md) |
| CacheRoute | arXiv'26 | [笔记](topics/llm-serving/2026-cacheroute/README.md) |

到这里已经有了完整的一条链：Transformer 架构 → KV Cache → 单机分页管理 → 跨机路由。

---

## 阶段 1 · 推理系统主线（当前阶段，建议按顺序）

读完这一阶段，你能看懂几乎所有 LLM serving 相关的技术讨论与招聘要求。

- [x] ⭐⭐⭐ **Orca: A Distributed Serving System for Transformer-Based Generative Models** — OSDI'22 · 🚧 **已拆成 GitHub Issue，待接单** · [骨架页](topics/llm-serving/2022-orca/README.md)
      *为什么现在读*：continuous batching（iteration-level scheduling）的出处。vLLM 论文全程把它当既有背景，[Continuous Batching 概念页](concepts/continuous-batching.md)目前也是基于二手描述写的。它和 PagedAttention 是当今每个推理引擎的两块基石。
      *难度*：中。无 arXiv 版本，只有 OSDI 页面的 PDF。

- [x] ⭐⭐⭐ **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** — NeurIPS'22 · [arXiv:2205.14135](https://arxiv.org/abs/2205.14135) · 🚧 **已拆成 GitHub Issue，待接单** · [骨架页](topics/foundations/2022-flashattention/README.md)
      *为什么现在读*：Transformer 笔记 §3.4 指出原论文 Table 1 缺了「显存」这一维，FlashAttention 正是补这一维的。它同时是训练和推理 prefill 阶段的标配，vLLM 的 prefill 直接调用它。
      *难度*：中偏高，需要一点 GPU 存储层次（SRAM / HBM）的概念。读完能掌握「IO-aware 算法设计」这个思路。

- [ ] ⭐⭐ **SGLang: Efficient Execution of Structured Language Model Programs（RadixAttention）** — NeurIPS'24 · [arXiv:2312.07104](https://arxiv.org/abs/2312.07104)
      *为什么现在读*：补上 vLLM §4.4 那个明确的缺口——共享 prefix 需要**手动声明**。RadixAttention 用基数树做**自动**前缀复用。读完 [Prefix Caching](concepts/prefix-caching.md) 就从原理补到了工程实现。

- [ ] ⭐⭐ **DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving** — OSDI'24 · [arXiv:2401.09670](https://arxiv.org/abs/2401.09670)
      *为什么现在读*：vLLM §2.2 指出 prefill 是 compute-bound、decode 是 memory-bound，却把两者放在同一批机器上跑。DistServe 直接把它们拆到不同实例。它也是 goodput 这个指标最重要的推动者，与 [SLO 容量](concepts/slo-capacity.md)直接相关。

- [ ] ⭐⭐ **Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving** — FAST'25 · [arXiv:2407.00079](https://arxiv.org/abs/2407.00079)
      *为什么现在读*：它是「KV 是本地状态」这个假设的反面——把 KV Cache 做成独立的分布式存储层。与 CacheRoute 的「靠入口路由保住本地 KV」构成两条相反的技术路线，对照读收获最大。

- [ ] ⭐ **Sarathi-Serve: Taming Throughput-Latency Tradeoff with Chunked Prefills** — OSDI'24 · [arXiv:2403.02310](https://arxiv.org/abs/2403.02310)
      *为什么读*：解决 continuous batching 里「长 prompt 的 prefill 打断 decode 造成延迟毛刺」这个具体问题。工程味浓，适合在读完 Orca 之后作为补充。

---

## 阶段 2 · 模型侧的效率优化

这一阶段的论文都直接影响推理成本，概念简单、篇幅短，适合穿插在阶段 1 之间读。

- [ ] ⭐⭐ **GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints** — EMNLP'23 · [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
      *为什么读*：[Multi-Head Attention](concepts/multi-head-attention.md) 提到 KV Cache 大小正比于 $h \cdot d_k$，GQA/MQA 就是压这一项的标准做法。今天几乎所有开源模型都用 GQA。篇幅很短。

- [ ] ⭐⭐ **RoFormer: Enhanced Transformer with Rotary Position Embedding（RoPE）** · [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
      *为什么读*：Transformer 笔记 Q3 指出正弦编码「可外推」这个论断从未被验证，RoPE 是沿这条线的主流答案，今天几乎所有主流模型都在用。

- [ ] ⭐ **Fast Inference from Transformers via Speculative Decoding** — ICML'23 · [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)
      *为什么读*：直接攻 Transformer 论文 §7 那条至今未解的展望——「making generation less sequential」。用小模型草稿 + 大模型验证打破自回归的顺序依赖。

---

## 阶段 3 · 训练系统与并行策略

推理看熟之后再来。这一阶段的论文彼此依赖强，建议连着读完。

- [x] ⭐⭐⭐ **Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism** · [arXiv:1909.08053](https://arxiv.org/abs/1909.08053) · 🚧 **已拆成 GitHub Issue，待接单** · [骨架页](topics/llm-training/2019-megatron-lm/README.md)
      *为什么读*：张量并行的出处。vLLM §4.6 直接说「支持 Megatron-LM 式的张量并行」，CacheRoute 的一个 destination 就是一个 TP2 组。不读这篇，「TP2」只是个符号。

- [ ] ⭐⭐⭐ **ZeRO: Memory Optimizations Toward Training Trillion Parameter Models** — SC'20 · [arXiv:1910.02054](https://arxiv.org/abs/1910.02054)
      *为什么读*：训练侧的显存优化范式，与 vLLM 在推理侧做的事精神一致——都是「显存是瓶颈，靠切分与按需分配解决」。DeepSpeed 的核心。

- [ ] ⭐⭐ **GPipe / PipeDream** — 流水并行的两种气泡处理思路 · [GPipe arXiv:1811.06965](https://arxiv.org/abs/1811.06965)
      *为什么读*：建议合并成一篇对比笔记。三种并行（数据 / 张量 / 流水）凑齐后，就能读懂任何一篇大模型训练报告里的并行配置。

- [ ] ⭐ **序列并行 / Ring Attention** — 长上下文训练的并行维度补充。

---

## 阶段 4 · 分布式系统经典

计算机系统的通识，与 LLM 无关但决定你理解系统论文的深度。可以长期慢读。

- [x] ⭐⭐⭐ **In Search of an Understandable Consensus Algorithm（Raft）** — ATC'14 · 🚧 **已拆成 GitHub Issue，待接单** · [骨架页](topics/distributed-systems/2014-raft/README.md)
      *为什么读*：共识算法的最佳入门，论文本身以「可理解性」为设计目标，对新人极友好。

- [ ] ⭐⭐⭐ **MapReduce: Simplified Data Processing on Large Clusters** — OSDI'04
      *为什么读*：分布式计算的思维起点。篇幅短、思想清晰，是理解「把复杂度藏进框架」这一范式的最佳样本。

- [ ] ⭐⭐ **The Google File System** — SOSP'03 · **Bigtable** — OSDI'06
      *为什么读*：存储层的两块基石，与 Mooncake 这类 KV 存储层设计一脉相承。

- [ ] ⭐⭐ **Dynamo: Amazon's Highly Available Key-value Store** — SOSP'07
      *为什么读*：一致性哈希在工业系统中的经典应用，是[缓存亲和性与负载均衡](concepts/cache-affinity-vs-load-balance.md)的理论背景。

- [ ] ⭐ **Spanner** — OSDI'12 · **Paxos Made Simple / Paxos Made Live**

---

## 阶段 5 · 网络与 RDMA

与训练 / 推理的通信性能直接相关，建议在阶段 3 之后读。

- [ ] ⭐⭐ **FaRM: Fast Remote Memory** — NSDI'14
- [ ] ⭐⭐ **Design Guidelines for High Performance RDMA Systems** — ATC'16
      *为什么读*：实操指南性质，读完能理解 Mooncake 这类跨机 KV 传输的性能边界从哪来。
- [ ] ⭐ **NCCL 与集合通信的拓扑感知优化** — 与训练并行策略强相关。

---

## 阶段 6 · Agent 与 RAG 应用架构

这一阶段迭代快、共识少，建议等前面阶段稳固后再进入，且以综述性阅读为主。

- [ ] ⭐⭐ **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks** — NeurIPS'20 · [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
- [ ] ⭐⭐ **ReAct: Synergizing Reasoning and Acting in Language Models** — ICLR'23 · [arXiv:2210.03629](https://arxiv.org/abs/2210.03629)
- [ ] ⭐ **MemGPT / Toolformer** — Agent 记忆与工具调用，适合做成主题综述而非单篇精读。

---

## 阶段 7 · 经典奠基（补充阅读）

- [ ] ⭐⭐⭐ **BERT: Pre-training of Deep Bidirectional Transformers** · [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
- [ ] ⭐⭐⭐ **Language Models are Few-Shot Learners（GPT-3）** · [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)
      *为什么读*：Transformer 的 encoder 与 decoder 两条分支各自的代表作，读完 LLM 的来龙去脉就闭环了。
- [ ] ⭐⭐ **Scaling Laws for Neural Language Models** · [arXiv:2001.08361](https://arxiv.org/abs/2001.08361) · **Chinchilla** · [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
      *为什么读*：解释「为什么模型是这个尺寸」，是理解所有模型规模决策的依据。

---

## 已进入精读

| 论文 | 主题 | 笔记 | 完成日期 |
| --- | --- | --- | --- |
| Attention Is All You Need | Foundations | [笔记](topics/foundations/2017-attention-is-all-you-need/README.md) | 2026-09-02 |
| vLLM / PagedAttention | LLM Serving | [笔记](topics/llm-serving/2023-vllm-pagedattention/README.md) | 2026-09-02 |
| CacheRoute | LLM Serving | [笔记](topics/llm-serving/2026-cacheroute/README.md) | 2026-09-02 |
