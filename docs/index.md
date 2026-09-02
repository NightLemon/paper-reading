# Paper Reading

计算机与软件系统工程方向的论文精读笔记与知识点沉淀。

每篇论文一个目录，包含**原文**（PDF + HTML）与一份**六段式报告**：元信息 → 摘要速览 → 细读 → 关键问题解析 → 可迁移知识点 → 批判与开放问题。跨论文复用的概念抽到 [`concepts/`](concepts/README.md) 单独成篇，与论文笔记双向链接。

---

## 论文索引

| 论文 | 主题 | 年份 | 来源 | 状态 | 评分 |
| --- | --- | --- | --- | --- | --- |
| [Attention Is All You Need](topics/foundations/2017-attention-is-all-you-need/README.md) | Foundations | 2017 | NeurIPS'17 | ✅ 已完成 | ⭐⭐⭐⭐⭐ |
| [Efficient Memory Management for LLM Serving with PagedAttention（vLLM）](topics/llm-serving/2023-vllm-pagedattention/README.md) | LLM Serving | 2023 | SOSP'23 | ✅ 已完成 | ⭐⭐⭐⭐⭐ |
| [SGLang: Efficient Execution of Structured Language Model Programs（RadixAttention）](topics/llm-serving/2024-sglang-radixattention/README.md) | LLM Serving | 2024 | NeurIPS'24 | ✅ 已完成 | ⭐⭐⭐⭐⭐ |
| [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving](topics/llm-serving/2024-distserve/README.md) | LLM Serving | 2024 | OSDI'24 | ✅ 已完成 | ⭐⭐⭐⭐⭐ |
| [Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving](topics/llm-serving/2025-mooncake/README.md) | LLM Serving | 2025 | FAST'25 | ✅ 已完成 | ⭐⭐⭐⭐ |
| [CacheRoute: Planned Prefix-Affinity Routing for Large-Scale LLM Serving](topics/llm-serving/2026-cacheroute/README.md) | LLM Serving | 2026 | arXiv | ✅ 已完成 | ⭐⭐⭐⭐ |
| [Orca: A Distributed Serving System for Transformer-Based Generative Models](topics/llm-serving/2022-orca/README.md) | LLM Serving | 2022 | OSDI'22 | 🚧 已派工单 | — |
| [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](topics/foundations/2022-flashattention/README.md) | Foundations | 2022 | NeurIPS'22 | 🚧 已派工单 | — |
| [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](topics/llm-training/2019-megatron-lm/README.md) | LLM Training | 2019 | arXiv | 🚧 已派工单 | — |
| [In Search of an Understandable Consensus Algorithm（Raft）](topics/distributed-systems/2014-raft/README.md) | Distributed Systems | 2014 | ATC'14 | 🚧 已派工单 | — |

状态图例：📥 待读 · 🚧 已拆成 GitHub Issue 待接单 · 📖 精读中 · ✅ 已完成

---

## 主题分区

| 主题 | 说明 | 篇数 |
| --- | --- | --- |
| [foundations](topics/foundations/README.md) | 奠基性经典论文 | 2 |
| [llm-serving](topics/llm-serving/README.md) | LLM 推理服务、调度、KV Cache、请求路由 | 6 |
| [llm-training](topics/llm-training/README.md) | 训练系统与并行策略 | 1 |
| [distributed-systems](topics/distributed-systems/README.md) | 分布式系统与共识 | 1 |
| [networking](topics/networking/README.md) | 网络、RDMA、集合通信 | 0 |
| [llm-applications](topics/llm-applications/README.md) | Agent、RAG 等应用架构 | 0 |

---

## 其他入口

- [待读队列 reading-list.md](reading-list.md) —— 候选论文与推荐理由，勾选后进入精读
- [概念索引 concepts/](concepts/README.md) —— 跨论文的知识点
- 笔记模板在仓库根目录 `templates/`（仅供撰写使用，不发布到站点）

---

## 目录约定

```
docs/topics/<主题>/<年份>-<论文短名>/
├── README.md     # 六段式报告
├── paper.pdf     # 原文 PDF
└── paper.html    # 原文 HTML（arXiv LaTeXML 版优先）
```

- 论文目录名用 `<发表年份>-<kebab-case 短名>`，例如 `2026-cacheroute`。
- 概念文件名用 kebab-case，例如 `concepts/prefix-caching.md`。
- 论文笔记与概念笔记的元数据写在 YAML frontmatter 里，便于日后脚本化生成索引。

## 加一篇新论文的流程

1. 拿到原文：**优先 arXiv 的 HTML 版**（LaTeXML 生成，章节/公式/图注结构完整），配合 PDF 一起放入论文目录。没有 HTML 版的（多数 OSDI/SOSP/NSDI 论文）只放 PDF 并在笔记里记录会议页面链接。
2. 从仓库根目录的 `templates/paper-note.md` 复制出 `README.md`，填 frontmatter。
3. 写完报告后，把新抽出的概念补进 `docs/concepts/`，并双向链接。
4. 更新本页的论文索引表与主题分区篇数，以及仓库根目录 `mkdocs.yml` 的 `nav`。
