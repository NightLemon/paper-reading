# 待读队列 Reading List

这是一条**面向新人的学习路径**，按依赖关系分阶段排列。每篇都标注了发表场合、影响力层级和推荐理由。

**用法**：把想读的那行 `[ ]` 改成 `[x]`，然后说「精读 XXX」，我会拉原文、建目录、写完整报告。也可以直接说「按路径继续」，我从当前阶段的第一篇未读论文开始。

**当前进度（2026-09-08）**：阶段 1–2 已全部精读；阶段 3 已补入 **ZeRO** 与 **GPipe / PipeDream 对比精读**，接下来是 **序列并行 / Ring Attention**。

**影响力标注**：
`⭐⭐⭐` 领域奠基，不读会有知识断层 · `⭐⭐` 高影响力，业界广泛引用/落地 · `⭐` 值得一读，视兴趣取舍

---

## 阶段 0 · 已完成

| 论文 | 场合 | 笔记 |
| --- | --- | --- |
| Attention Is All You Need | NeurIPS'17 | [笔记](topics/foundations/2017-attention-is-all-you-need/README.md) |
| vLLM / PagedAttention | SOSP'23 | [笔记](topics/llm-serving/2023-vllm-pagedattention/README.md) |
| SGLang / RadixAttention | NeurIPS'24 | [笔记](topics/llm-serving/2024-sglang-radixattention/README.md) |
| DistServe | OSDI'24 | [笔记](topics/llm-serving/2024-distserve/README.md) |
| CacheRoute | arXiv'26 | [笔记](topics/llm-serving/2026-cacheroute/README.md) |

到这里已经有了完整的一条链：Transformer 架构 → KV Cache → 单机分页管理 → 跨机路由。

---

## 阶段 1 · 推理系统主线（已完成）

读完这一阶段，你能看懂几乎所有 LLM serving 相关的技术讨论与招聘要求。

- [x] ⭐⭐⭐ **Orca: A Distributed Serving System for Transformer-Based Generative Models** — OSDI'22 · ✅ **已精读** · [笔记](topics/llm-serving/2022-orca/README.md)

- [x] ⭐⭐⭐ **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** — NeurIPS'22 · [arXiv:2205.14135](https://arxiv.org/abs/2205.14135) · ✅ **已精读** · [笔记](topics/foundations/2022-flashattention/README.md)

- [x] ⭐⭐ **SGLang: Efficient Execution of Structured Language Model Programs（RadixAttention）** — NeurIPS'24 · [arXiv:2312.07104](https://arxiv.org/abs/2312.07104) · ✅ **已精读** · [笔记](topics/llm-serving/2024-sglang-radixattention/README.md)

- [x] ⭐⭐ **DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving** — OSDI'24 · [arXiv:2401.09670](https://arxiv.org/abs/2401.09670) · ✅ **已精读** · [笔记](topics/llm-serving/2024-distserve/README.md)

- [x] ⭐⭐ **Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving** — FAST'25 · [arXiv:2407.00079](https://arxiv.org/abs/2407.00079) · ✅ **已精读** · [笔记](topics/llm-serving/2025-mooncake/README.md)

- [x] ⭐ **Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve** — OSDI'24 · [arXiv:2403.02310](https://arxiv.org/abs/2403.02310) · ✅ **已精读** · [笔记](topics/llm-serving/2024-sarathi-serve/README.md)
      *为什么读*：解决 continuous batching 里「长 prompt 的 prefill 打断 decode 造成延迟毛刺」这个具体问题。工程味浓，适合在读完 Orca 之后作为补充。

---

## 阶段 2 · 模型侧的效率优化

✅ **已完成**。这一阶段补充 KV 共享、旋转位置编码与投机解码；各篇分别解释成本、数学性质和实验适用范围。

- [x] ⭐⭐ **GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints** — EMNLP'23 · [arXiv:2305.13245](https://arxiv.org/abs/2305.13245) · ✅ **已精读** · [笔记](topics/foundations/2023-gqa/README.md)
      *为什么读*：在保持 query 头数和每头维度时，GQA/MQA 通过共享 K/V 压缩 [KV Cache](concepts/kv-cache.md)。精读同时解释 MHA checkpoint 转换、uptraining 与组数的质量–速度取舍。

- [x] ⭐⭐ **RoFormer: Enhanced Transformer with Rotary Position Embedding（RoPE）** · [arXiv:2104.09864](https://arxiv.org/abs/2104.09864) · ✅ **已精读** · [笔记](topics/foundations/2021-roformer/README.md)
      *为什么读*：承接 Transformer 笔记的位置编码问题，从二维旋转推导相对位置关系，并区分位置函数可计算、距离衰减与超训练长度的实际质量。

- [x] ⭐ **Fast Inference from Transformers via Speculative Decoding** — ICML'23 · [arXiv:2211.17192](https://arxiv.org/abs/2211.17192) · ✅ **已精读** · [笔记](topics/llm-serving/2023-speculative-decoding/README.md)
      *为什么读*：承接 Transformer §7 的“making generation less sequential”。用小模型草稿与并行验证减少目标模型串行调用，精读接受概率、残差修正和保持目标分布的条件。

---

## 阶段 3 · 训练系统与并行策略

推理看熟之后再来。这一阶段的论文彼此依赖强，建议连着读完。

- [x] ⭐⭐⭐ **Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism** · [arXiv:1909.08053](https://arxiv.org/abs/1909.08053) · ✅ **已精读** · [笔记](topics/llm-training/2019-megatron-lm/README.md)

- [x] ⭐⭐⭐ **ZeRO: Memory Optimizations Toward Training Trillion Parameter Models** — SC'20 · [arXiv:1910.02054](https://arxiv.org/abs/1910.02054) · ✅ **已精读** · [笔记](topics/llm-training/2020-zero/README.md)
      *为什么读*：训练侧的显存优化范式，与 vLLM 在推理侧做的事精神一致——都是「显存是瓶颈，靠切分与按需分配解决」。DeepSpeed 的核心。

- [x] ⭐⭐ **GPipe / PipeDream** — 流水调度与更新语义的两种取舍 · [GPipe arXiv:1811.06965](https://arxiv.org/abs/1811.06965) · [PipeDream SOSP'19](https://doi.org/10.1145/3341301.3359646) · ✅ **已精读** · [笔记](topics/llm-training/2019-gpipe-pipedream/README.md)
      *为什么读*：已合并精读，分别按两篇原文章节分析同步微批、1F1B、weight stashing 和达到目标质量的时间；与 Megatron-LM、ZeRO 串起多维并行。

- [ ] ⭐ **序列并行 / Ring Attention** — 长上下文训练的并行维度补充。

---

## 阶段 4 · 分布式系统经典

计算机系统的通识，与 LLM 无关但决定你理解系统论文的深度。可以长期慢读。

- [x] ⭐⭐⭐ **In Search of an Understandable Consensus Algorithm（Raft）** — ATC'14 · ✅ **已精读** · [笔记](topics/distributed-systems/2014-raft/README.md)

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
| In Search of an Understandable Consensus Algorithm（Raft） | Distributed Systems | [笔记](topics/distributed-systems/2014-raft/README.md) | 2026-09-02 |
| Attention Is All You Need | Foundations | [笔记](topics/foundations/2017-attention-is-all-you-need/README.md) | 2026-09-02 |
| Megatron-LM | LLM Training | [笔记](topics/llm-training/2019-megatron-lm/README.md) | 2026-09-02 |
| Orca | LLM Serving | [笔记](topics/llm-serving/2022-orca/README.md) | 2026-09-02 |
| FlashAttention | Foundations | [笔记](topics/foundations/2022-flashattention/README.md) | 2026-09-02 |
| vLLM / PagedAttention | LLM Serving | [笔记](topics/llm-serving/2023-vllm-pagedattention/README.md) | 2026-09-02 |
| SGLang / RadixAttention | LLM Serving | [笔记](topics/llm-serving/2024-sglang-radixattention/README.md) | 2026-09-02 |
| DistServe | LLM Serving | [笔记](topics/llm-serving/2024-distserve/README.md) | 2026-09-02 |
| Mooncake | LLM Serving | [笔记](topics/llm-serving/2025-mooncake/README.md) | 2026-09-02 |
| CacheRoute | LLM Serving | [笔记](topics/llm-serving/2026-cacheroute/README.md) | 2026-09-02 |
| Sarathi-Serve | LLM Serving | [笔记](topics/llm-serving/2024-sarathi-serve/README.md) | 2026-09-08 |
| GQA | Foundations | [笔记](topics/foundations/2023-gqa/README.md) | 2026-09-08 |
| RoFormer / RoPE | Foundations | [笔记](topics/foundations/2021-roformer/README.md) | 2026-09-08 |
| Speculative Decoding | LLM Serving | [笔记](topics/llm-serving/2023-speculative-decoding/README.md) | 2026-09-08 |
| ZeRO | LLM Training | [笔记](topics/llm-training/2020-zero/README.md) | 2026-09-08 |
| GPipe / PipeDream | LLM Training | [笔记](topics/llm-training/2019-gpipe-pipedream/README.md) | 2026-09-08 |
