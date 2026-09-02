---
title: "SGLang: Efficient Execution of Structured Language Model Programs"
authors: ["Lianmin Zheng", "Liangsheng Yin", "Zhiqiang Xie", "Chuyue Sun", "Jeff Huang", "Cody Hao Yu", "Shiyi Cao", "Christos Kozyrakis", "Ion Stoica", "Joseph E. Gonzalez", "Clark Barrett", "Ying Sheng"]
affiliation: "Stanford University / UC Berkeley / Shanghai Jiao Tong University / Texas A&M University"
venue: "NeurIPS 2024"
year: 2024
arxiv: "2312.07104"
url: "https://arxiv.org/abs/2312.07104"
topic: "llm-serving"
tags: ["radixattention", "prefix-caching", "kv-cache", "constrained-decoding", "dsl", "neurips"]
concepts: ["prefix-caching", "kv-cache", "constrained-decoding", "paged-kv-memory"]
status: "done"
rating: 5
read_date: "2026-09-02"
---

# SGLang: Efficient Execution of Structured Language Model Programs（RadixAttention）

> **一句话结论**：把 KV Cache 当成一棵**基数树上的 LRU 缓存**来管，请求结束后不丢弃、按最长公共前缀自动复用，再配一个「最长匹配前缀优先」的调度器把命中率推到接近理论最优——在多调用的 LM program 负载上取得最高 $6.4\times$ 吞吐提升，而在完全没有复用机会时的额外开销低于 0.3%。

---

## 1. 元信息

| 项目 | 内容 |
| --- | --- |
| 作者 / 机构 | Lianmin Zheng、Liangsheng Yin、Zhiqiang Xie、Chuyue Sun、Jeff Huang（共同一作）等 12 人 · Stanford / UC Berkeley / 上海交大 / Texas A&M |
| 发表 | **NeurIPS 2024**（arXiv:2312.07104） |
| 原文 | [PDF](paper.pdf) · [HTML](paper.html) · [arXiv](https://arxiv.org/abs/2312.07104) |
| 代码 | [sgl-project/sglang](https://github.com/sgl-project/sglang)（现已是与 vLLM 并列的主流推理引擎） |
| 关键词 | RadixAttention · LM program · 压缩有限状态机 · cache-aware scheduling |
| 前置知识 | [KV Cache](../../../concepts/kv-cache.md)、[Prefix Caching](../../../concepts/prefix-caching.md)、[分页式 KV 管理](../../../concepts/paged-kv-memory.md)、基数树 / LRU |
| 实验条件 | Llama-2 7B/70B、Mixtral-8x7B、LLaVA 系列，fp16，AWS EC2 G5（A10G 24GB）为主，部分 A100 80GB |

**这篇在本仓库的位置**：vLLM 笔记 §4.4 指出它的 shared prefix 需要服务方**预先声明**哪些前缀会被共享，这是设计上一个明显的未完成部分。SGLang 补的正是这个缺口——**自动、跨请求、跨调用**的前缀复用。读完这篇，[Prefix Caching](../../../concepts/prefix-caching.md) 就从「原理」补到了「工程实现」。

---

## 2. 摘要速览（5 分钟版）

### 2.1 要解决的问题

LLM 的使用形态正在从「单轮对话」转向**LM program**（Language Model Program）——用程序去调度和控制模型的生成过程。论文给出 LM program 的两条共同性质：

1. 包含**多次 LLM 调用**，中间夹杂控制流；
2. 接收**结构化输入**、产生**结构化输出**。

few-shot、self-consistency、tree-of-thought、ReAct agent、RAG pipeline 全都落在这个范畴里。由此产生两个问题：

- **写起来麻烦**：大量字符串拼接、脆弱的输出解析、手写并行控制。
- **跑起来低效**：当时的推理引擎（vLLM、TGI、TensorRT-LLM）**在不了解工作负载的前提下**做优化，这让它们通用且健壮，但对任何具体负载都留下了明显的浪费。最突出的两处：
  - **KV Cache 复用**：一个请求处理完，它的 KV Cache 就被丢弃。而 LM program 执行时，不同调用之间大量共享前缀。
  - **约束解码**：JSON 之类的结构化输出，很多位置其实只有一个合法 token，但已有系统仍然一次只解一个。

### 2.2 核心方法

SGLang = **前端 DSL** + **后端 runtime**，两部分可以协同也可以独立使用。

**前端**：嵌入 Python 的领域特定语言。生成原语 `gen` / `select` / `extend`（`+=`），并行原语 `fork` / `join`，多模态原语 `image` / `video`。解释器把 prompt 当成**异步流**，原语提交后不阻塞，取结果时才同步——这提供了程序内并行。

**后端三项优化**：

| 技术 | 解决什么 |
| --- | --- |
| **RadixAttention** | 请求结束后**不丢弃** KV Cache，而是存进一棵基数树，做成 LRU 缓存；配 cache-aware 调度提高命中率 |
| **压缩有限状态机** | 把正则约束 FSM 中相邻的**单一转移边**压成一条边，一次前向解出多个 token |
| **API 推测执行** | 对黑盒 API 模型，忽略 stop 条件多生成一段，供后续原语匹配复用，省一次调用的输入费用 |

### 2.3 主要结果

| 指标 | 结果 |
| --- | --- |
| 吞吐（Llama-7B，各类 LM program 负载） | 最高 $\mathbf{6.4\times}$（对比 Guidance / vLLM v0.2.5 / LMQL） |
| 延迟 | 最高降低 $3.7\times$ |
| 缓存命中率 | 各基准上 50%–99%；cache-aware 调度平均达到**理论最优命中率的 96%** |
| 无复用机会时的开销 | ShareGPT 上 100 个请求耗时 74.3 s，其中维护基数树只用 **0.2 s（< 0.3%）** |
| 压缩 FSM | JSON 解码吞吐 $+1.6\times$ |
| 多模态 | LLaVA-v1.5-7B：0.18 → **1.15 image/s**；LLaVA-NeXT-34B 视频：0.02 → **0.10 frame/s** |
| 生产环境（Chatbot Arena，单 worker，一个月） | LLaVA-NeXT-34B 命中率 **52.4%**，Vicuna-33B **74.1%**；后者首 token 延迟平均降低 $1.7\times$ |

### 2.4 我的评价

值得精读，理由和 vLLM 不同。vLLM 的贡献是**把一个成熟的 OS 方案精确映射到新问题上**；SGLang 的贡献是**把「缓存」这个抽象本身用对了**。

具体说：vLLM 已经有了分页和块共享的机制，但它把 KV 当作「请求的私有资源」——请求结束就释放。SGLang 意识到 KV 其实是**可以跨请求复用的缓存内容**，于是把缓存该有的三件套补齐了——**索引结构**（基数树）、**淘汰策略**（叶子优先的 LRU）、**准入/调度**（最长匹配前缀优先）。一旦把它当缓存看，剩下的设计几乎是被推着走的。

三个我认为最值得记住的点：

1. **「命中率 → 更大 batch → 更高吞吐 + 更低延迟」这条链被实测验证了**（§6.3 消融，通过运行时部分禁用已匹配 token 来扫描命中率）。这与 [CacheRoute](../2026-cacheroute/README.md) 里「命中率与容量非单调」并不矛盾——差别在于**单机内提高命中率不引入负载倾斜**，而跨机路由会。
2. **开销 < 0.3%，所以可以默认开启**。这个数字比 $6.4\times$ 更重要：它决定了这个特性能不能进默认配置。
3. **附录 A.4 的 meta-tree router 与 [CacheRoute](../2026-cacheroute/README.md) 直接对话**，而且论文点名 Preble 是基于早期 SGLang 的并行工作——CacheRoute 最强的基线就是 Preble。这条线索把本仓库的三篇 serving 论文串起来了。

---

## 3. 细读

### 3.1 Introduction

论文先立一个概念：**LM program**。它的两条性质（多次调用 + 结构化输入输出）是全文所有设计的出发点。

对已有系统的批评写得很克制且准确：

> State-of-the-art inference engines have been optimized to reduce latency and improve throughput **without direct knowledge of the workload**. This makes these systems general and robust but also results in significant inefficiencies for any given workload.

也就是说，问题不在于 vLLM 做错了什么，而在于**通用性本身有代价**——它不知道下一个请求会不会和上一个共享前缀，于是只能丢弃。

系统由前端语言 + 后端 runtime 两部分组成（Figure 1），三项技术分别对应三类浪费。

> **批注**：「通用系统对任何具体负载都留下浪费」这个论证方式很值得学。它把「引入领域知识」而不是「更聪明的通用算法」立为贡献来源，因此后面的前后端协同设计（frontend hint）就顺理成章。这也解释了为什么 SGLang 一定要自带一个前端 DSL——没有前端，runtime 就拿不到「这几个请求会 fork 出同一个前缀」这种信息。

### 3.2 Programming Model

**语言原语**：

| 原语 | 作用 |
| --- | --- |
| `gen(name, regex=...)` | 调用模型生成，结果存入变量；`regex` 参数施加正则约束 |
| `select(name, choices)` | 让模型在选项中挑概率最高的 |
| `+=` / `extend` | 向 prompt 追加字符串 |
| `s["name"]` | 取出某次生成的结果 |
| `fork` / `join` | 创建 / 合并 prompt 状态的并行分支 |
| `image` / `video` | 多模态输入 |

**执行模式**：

- **解释器模式**（默认）：prompt 被当作异步流，`extend` / `gen` / `select` 提交后立即返回，Python 代码继续跑——论文的类比是「像异步启动 CUDA kernel 一样」。每个 prompt 由后台线程里的 stream executor 管理，从而获得**程序内并行**。取结果时阻塞，保证同步正确。
- **编译器模式**：程序可被 trace 成计算图交给 graph executor，支持更多静态优化（附录 D）。

**与同类系统的对比**（Table 1）：

| System | Syntax | Language Primitives | Runtime Backends |
| --- | --- | --- | --- |
| LMQL | Custom | extend, gen, select | HF Transformers, llama.cpp, OpenAI |
| Guidance | Python | extend, gen, select, image | HF Transformers, llama.cpp, OpenAI |
| **SGLang** | Python | extend, gen, select, image, video, **fork, join** | **SGLang Runtime (SRT)**, OpenAI |

论文把编程系统分成高层（LangChain、DSPy，会改写/自动生成 prompt）与低层（LMQL、Guidance、SGLang，直接操纵 prompt）。SGLang 属于低层，区别在于**自带协同设计的 runtime**。高层系统可以编译到低层系统——论文在评测里把 SGLang 作为 DSPy 的后端。

一个可量化的易用性数字：Figure 2 那个 branch-solve-merge 的例子，用 OpenAI 风格 API 写等价程序需要 $2.1\times$ 的代码行数。

### 3.3 Efficient KV Cache Reuse with RadixAttention

这一节是全文核心。

#### 问题

KV Cache 的计算**只依赖前缀 token**，因此拥有相同 prompt 前缀的请求可以复用。但已有系统「在一个生成请求结束后就丢弃 KV Cache」，于是跨调用的复用机会全部浪费。论文指出已有工作探索过部分复用场景，但**需要手动配置，且无法处理动态树状结构**。

#### 基数树作为缓存索引

基数树（radix tree）是 trie 的空间高效变体——**边可以标注变长的元素序列**，而不只是单个元素。SGLang 用它维护「token 序列 → 对应 KV Cache 张量」的映射。KV 张量本身以**非连续的分页布局**存储，**每页大小等于一个 token**。

> **批注**：「每页一个 token」是一个值得注意的选择。vLLM 默认 block size = 16，理由是平衡 GPU 并行度与内部碎片（见 [分页式 KV 管理](../../../concepts/paged-kv-memory.md)）。SGLang 取 1，是因为它的首要目标变成了**最大化前缀匹配的粒度**——块越大，「前缀相同但不足一整块」的部分就越难共享。这是同一个取舍在不同优化目标下给出的相反答案。

#### 淘汰：叶子优先的 LRU

> we introduce a simple LRU eviction policy that evicts the **least recently used leaf** first. By evicting leaves first, we enable the re-use of their common ancestors until those ancestors become leaves and are also evicted.

这条规则的含义：**越靠近根的节点（越短的公共前缀）被越多请求共享，因此活得越久**。淘汰从叶子开始逐层往上剥，共享度高的前缀自然被保留。

配合 continuous batching，还需要**引用计数**：每个节点记录有多少个运行中的请求正在使用它，refcount 为 0 才可淘汰。

#### 内存池不预留

> we do **not** preallocate a fixed-size memory pool as a cache. Instead, we let the cached tokens and the currently running requests share the same memory pool. ... When enough waiting requests run, the system will evict all cached tokens in favor of a larger batch size.

也就是说，**缓存与运行中请求争抢同一块显存，且运行中请求优先**。当等待队列足够长时，系统会把所有缓存内容清空，换取更大的 batch。

> **批注**：这是一条很重要的设计判断——**缓存是机会性的，吞吐是刚性的**。它保证了「开启缓存最坏情况下不劣于不开缓存」，这也是 §6.3 那个 0.3% 开销数字能成立的前提。任何要把某个特性做成默认开启的设计，都需要类似的「最坏情况不劣化」保证。

#### 前后端协同：Frontend Hint

执行 `fork` 原语时，前端**先把公共前缀单独发给 runtime 作为 hint**，确保它被正确插入树中，然后再发各分支剩下的部分。这样 runtime 不必猜测分支之间的公共部分。树结构存在 CPU 上，维护开销可忽略。

#### Cache-aware 调度

命中率定义：

$$\text{cache hit rate} = \frac{\text{已缓存的 prompt token 数}}{\text{prompt token 总数}}$$

关键观察是**执行顺序显著影响命中率**——调度器若在不相关的请求之间频繁切换，会导致缓存抖动。因此 SGLang 放弃 FCFS，改为**按已匹配前缀长度排序，最长匹配前缀优先**。

**Theorem 3.1**：

> 对一批请求，**按深度优先搜索（DFS）顺序**遍历这批请求构成的基数树，可以取得**最优命中率**，前提是**缓存容量 ≥ 最大请求长度**。而「最长公共前缀优先」的顺序等价于 DFS 顺序。

论文同时给了两条限定：

- 脚注承认：实践中的计算与定理证明所描述的不同，因为**输出 token 数不可预测**会导致 KV Cache 被重算。
- 在线场景下 DFS 顺序会被打断，调度只能**近似** DFS 行为。
- 贪心的 cache-aware 调度**会导致饿死**（starvation），与公平调度的结合被列为 future work。

#### 分布式

- **张量并行**：每张 GPU 维护自己那一份分片的 KV Cache，**不需要额外同步**，因为树操作在各卡上是一样的。
- **数据并行**（附录 A.4）：每个 worker 维护自己的子树，**router 维护一棵 meta-tree**——一棵记录所有子树及其所在设备的 trie。新批请求到达 router 时先在 meta-tree 上做前缀匹配，再按「与特定 worker 及同组其他请求的共享前缀长度」衡量的亲和性来分派。worker 侧发生淘汰时把事件提交到队列，router 在低负载时段处理，因此这是一个**弱一致的分布式缓存**。4 worker + MMLU 上观测到线性扩展与最优命中率。论文明确指出**数据局部性与并行效率之间存在取舍**，把更好的调度策略列为 future work，并点名 **Preble** 是基于早期 SGLang 的并行工作。

> **批注**：附录 A.4 这半页就是 [CacheRoute](../2026-cacheroute/README.md) 整篇论文的问题陈述。SGLang 的 router 是**反应式**的（每批请求到达时在 meta-tree 上匹配），CacheRoute 换成了**周期性规划**的。而 CacheRoute 最强的基线 Preble，论文这里说了是基于早期 SGLang 做的数据并行调度。三篇论文在这一点上首尾相接。

### 3.4 Efficient Constrained Decoding with Compressed Finite State Machine

**已有做法**：把正则表达式转成有限状态机（FSM），解码时维护当前状态、取出下一状态允许的 token 集合、把非法 token 的概率置零，**一次解一个 token**。

**问题**：很多位置其实只有一个合法 token。论文举的例子是 JSON 里的常量串 `{"summary": "`——它在正常解码中跨越多个 token，需要多轮前向，尽管每一步都只有一个合法选择。

**做法**：分析 FSM，把**相邻的单一转移边压缩成一条边**。压缩后的边上可以一次前向解出多个 token。方法对所有正则表达式通用。

**难点**（附录 B）：约束是用**字符/字符串**表达的，而模型处理的是 **token**，两者之间的映射复杂且不是一一对应的。

### 3.5 Efficient Endpoint Calling with API Speculative Execution

针对只能调黑盒 API 的模型（如 GPT-4）。

例子：`s += context + "name:" + gen("name", stop="\n") + "job:" + gen("job", stop="\n")`。朴素做法是两次 API 调用，`context` 的输入 token 费用要付两遍。

**做法**：在第一次调用时开启推测执行，**忽略 stop 条件**让它多生成几个 token。解释器保留这些多出来的输出，与后续原语做匹配和复用。在 prompt 工程得当的情况下，模型能高准确率地匹配模板，省下一次 API 调用的延迟与输入费用。

### 3.6 Evaluation

#### 6.1 设置

- **实现**：PyTorch + 来自 **FlashInfer** 与 **Triton** 的自定义 CUDA kernel。
- **模型**：Llama-2（7B–70B）、Mixtral-8x7B（MoE）、LLaVA-v1.5-7B（图像）、LLaVA-NeXT-34B（视频）、GPT-3.5（API）。开源模型用 fp16。
- **硬件**：主要是 AWS EC2 G5（**A10G，24GB**）；7B 单卡，更大的模型用张量并行；部分实验在 A100 80GB。
- **基线**：Guidance v0.1.8 + llama.cpp；**vLLM v0.2.5** + 默认 API server；LMQL v0.7.3 + HF Transformers。
  - 论文脚注说明：**RadixAttention 已作为可选实验特性被部分集成进较新版本的 vLLM，因此这里用的是更早的版本做对比。**
- **负载**：5-shot MMLU、20-shot HellaSwag、ReAct agent、generative agents、tree-of-thought（GSM-8K）、skeleton-of-thought、LLM judge（branch-solve-merge）、JSON 解码、多轮对话（4 轮，每轮输入 256–512 token；short 输出 4–8 token，long 输出 256–512 token）、DSPy RAG。
- **指标**：**吞吐**用「每秒执行的 program 实例数（programs per second, p/s）」，跑足够大的批量取最大值；**延迟**在**不做批处理、一次只跑一个 program** 的条件下取平均。

#### 6.2 端到端性能

吞吐最高提升 $6.4\times$，延迟最高降低 $3.7\times$。论文逐个基准解释了收益来源：

| 基准 | 收益来源 |
| --- | --- |
| MMLU | 复用 5-shot 示例的 KV；既省显存（更大 batch → 更高吞吐）又省 prefill（更低首 token 延迟） |
| HellaSwag | **两级共享**：few-shot 示例 + 多个选项的公共问题前缀 |
| ReAct / generative agents | 复用 agent 模板与此前调用的 KV |
| tree-of-thought / skeleton-of-thought | 程序内并行 + 尽可能复用 KV |
| JSON 解码 | 压缩 FSM 一次解多个 token |
| 多轮对话 | 复用对话历史的 KV |
| DSPy RAG | 复用公共上下文示例 |

**命中率**：各基准上 50%–99%；cache-aware 调度**平均达到理论最优命中率的 96%**。

**一个论文自己点出的负面情形**：多轮对话的**长输出**版本几乎没有加速——因为不同会话之间共享很少，且**解码时间占主导**，而 KV 复用主要省的是 prefill。

LMQL 和 Guidance 在后五个基准的部分格子被排除，原因是「性能太慢与功能缺失」——LMQL 是 token 级处理慢加后端未优化，Guidance 缺少批处理与并行支持。

**更大模型**：Mixtral-8x7B 与 Llama-70B 用张量并行跑同一组基准，加速趋势与小模型相似。Guidance 与 LMQL 因缺少高效的张量并行实现被略去。

**多模态**：对输入图像**计算哈希作为基数树的 key**，从而复用同一张图片的图像 token 的 KV。基线是模型作者的原始 HF Transformers 实现。

| Model | 基线 | SGLang |
| --- | --- | --- |
| LLaVA-v1.5-7B（image） | 0.18 image/s | **1.15 image/s** |
| LLaVA-NeXT-34B（video） | 0.02 frame/s | **0.10 frame/s** |

**生产部署**：SGLang 已部署在 Chatbot Arena 服务开源模型。由于部分模型流量低，每个模型只有一个 worker。一个月后观测到 **LLaVA-NeXT-34B 命中率 52.4%、Vicuna-33B 74.1%**；命中来自公共系统消息、频繁复用的示例图片、以及多轮对话历史。Vicuna-33B 的首 token 延迟平均降低 $1.7\times$。

**API 模型**：用 GPT-3.5 从维基页面抽取三个字段。few-shot 提示下 API 推测执行准确率很高，**输入 token 成本降低约三倍**（因为要抽三个字段）。

#### 6.3 消融

**命中率 vs 性能**（tree-of-thought 基准，通过在运行时**部分禁用已匹配的 token** 来扫描命中率）：命中率越高 → batch 越大 → 吞吐越高、延迟越低。

**RadixAttention 各组件**（Figure 8c）：

| 配置 | 含义 |
| --- | --- |
| No Cache | 完全不用缓存 |
| No Tree-Structure | 用简单的**表式**缓存替代树式 |
| FCFS Schedule | 用先来先服务替代 cache-aware 调度 |
| Random Schedule | 随机顺序调度 |
| No Frontend Parallelism | 关闭解释器里的并行 |
| No Frontend Hint | 关闭 fork 时前端发送的前缀提示 |
| Full optimizations | 全开 |

结论是**每一项都是达到最佳性能所必需的**。关掉前端并行与前端提示同样导致 runtime 性能次优——论文用这一点论证前端语言与 runtime **协同设计**的必要性。

**开销**：在 ShareGPT 这个**没有任何复用机会**的基准上，100 个请求耗时 74.3 秒，其中管理 RadixAttention 数据结构只用了 **0.2 秒，占比 < 0.3%**。原因是树操作的复杂度是线性且常数很小。因此可以**默认开启**。

**压缩 FSM**：JSON 解码吞吐提升 $1.6\times$。另外，状态机需要**预处理一次并在一批请求间复用**；若每个请求都重做预处理，吞吐会**低 $2.4\times$**。

### 3.7 Related Work

论文对自己的定位写得很具体：

> RadixAttention **first** proposes treating the KV cache as a **tree-based LRU cache**. It is the first solution that supports **multi-level sharing, cache-aware scheduling, frontend-runtime co-scheduling, and distributed cases**.

对同期工作的区分：

| 工作 | 区别 |
| --- | --- |
| vLLM、ChunkedAttention | 探索了简单复用场景（如系统提示共享），不覆盖多级树状共享与 LRU 缓存 |
| PromptCache | 提出超越前缀的模块化复用，但**准确率最多下降 43%** |
| HydraGen、FlashInfer、ChunkedAttention | 聚焦 CUDA kernel 优化，没有 LRU 缓存的概念 |
| APIServe、LLM-SQL | 针对特定应用的 KV 复用，没有基数树与 cache-aware 调度 |

### 3.8 Future Directions and Conclusion

论文列出的未完成方向：

- 支持更多输出模态；
- 让 RadixAttention 跨**多级存储层次**（DRAM、磁盘）工作；
- 在 RadixAttention 中支持**模糊语义匹配**；
- 在 SGLang 之上提供更高层的原语；
- **修复 cache-aware 调度的饿死问题**；
- 增强编译器做更高级的静态优化（调度、内存规划）。

> **批注**：第二条（多级存储层次）后来由 Mooncake 这类工作接手；第五条（饿死）至今仍是 cache-aware 调度类系统的通病，[CacheRoute](../2026-cacheroute/README.md) 用「周期内固定的路由表 + 未准入流量走 flat-LB」在某种程度上绕开了它——未准入的冷尾流量始终有一条不受亲和性影响的通路。

---

## 4. 关键问题解析

### Q1: RadixAttention 相比 vLLM 的前缀共享，具体多了什么？

**A**: vLLM（[笔记](../2023-vllm-pagedattention/README.md) §4.4）已经有了块级共享 + 引用计数 + copy-on-write 的**机制**，但缺的是把它变成**缓存**所需要的三件事。

| 维度 | vLLM（论文版本） | SGLang RadixAttention |
| --- | --- | --- |
| **生命周期** | 请求结束即释放 KV | 请求结束后**保留**，进入缓存 |
| **共享的发现方式** | 服务方**预先声明**一组共享前缀，为其保留物理块 | **自动**：新请求在基数树上做最长前缀匹配 |
| **索引结构** | 无（按声明的前缀查） | 基数树，支持**多级共享**与动态树状结构 |
| **淘汰策略** | 无（缓存不存在，谈不上淘汰） | 叶子优先的 LRU + 引用计数 |
| **调度感知** | FCFS | 最长匹配前缀优先（cache-aware） |
| **共享粒度** | block（默认 16 token） | page = **1 token** |

关键的一句区别：vLLM 把 KV 当作**请求的私有资源**，SGLang 把它当作**服务的共享缓存**。前者的正确操作是「用完释放」，后者的正确操作是「用完留着，按策略淘汰」。

论文脚注也说明了后续演进：**RadixAttention 已作为可选实验特性被部分集成进较新版本的 vLLM**。这两条路线在工程上已经合流。

### Q2: 为什么用基数树而不是哈希表？「多级共享」是什么意思？

**A**: 哈希表能回答「这个完整前缀在不在缓存里」，回答不了「**最长的公共前缀有多长**」。

**多级共享**指的是共享结构本身是**树状且动态**的。论文的 HellaSwag 例子最清楚：

```
[20-shot 示例]                  ← 所有请求共享（第一级）
   └─ [某道题的题干]            ← 同一道题的所有选项共享（第二级）
        ├─ 选项 A
        ├─ 选项 B
        └─ 选项 C
```

哈希表要表达这个结构，得为每一级各存一个条目，且**无法在淘汰时知道谁是谁的祖先**——而这恰恰是叶子优先 LRU 成立的前提。

**为什么是基数树而不是普通 trie**：trie 的每条边只标一个元素，一条长前缀会变成一条长链，节点数等于 token 数。基数树的边可以标**变长序列**，一条没有分叉的长前缀塌缩成一条边——这既省内存，也让「分裂节点」这个操作恰好对应「出现了新的共享点」。论文 Figure 3 的第 (4) 步和第 (7) 步演示的正是这个分裂。

**实测证据**：消融里的 "No Tree-Structure"（改用表式缓存）确实劣于完整方案。

### Q3: 为什么 LRU 要「先淘汰叶子」？

**A**: 因为**在这棵树上，节点的深度与它的共享程度负相关**。

- 靠近根的节点 = 短前缀 = 被很多请求共享（系统提示、few-shot 示例）。
- 叶子节点 = 完整的某次对话 = 通常只被一个会话使用。

普通 LRU 只看「最近是否被访问」，会误伤高价值的短前缀：一个系统提示可能上一次被"访问"是在很久以前（因为之后的请求都是从它的子节点继续的），但它的价值极高。

**叶子优先** 把这个问题消掉了：

> By evicting leaves first, we enable the re-use of their common ancestors **until those ancestors become leaves and are also evicted**.

祖先节点只有在它的所有子节点都被淘汰之后（也就是它自己变成叶子之后）才可能被淘汰。这等于**给共享度高的前缀免费加了一层保护**，而不需要显式统计引用次数。

另一半是**引用计数**：continuous batching 下不能淘汰正在被运行中请求使用的节点，refcount 为 0 才可淘汰。注意这个 refcount 与 vLLM 的块 refcount 语义不同——vLLM 的计数是「多少个序列共享这个块」，SGLang 的是「多少个**运行中**的请求正在用这个节点」。

### Q4: Theorem 3.1 说了什么？它的前提有多强？

**A**:

**定理**：对一批请求，按 **DFS 顺序**遍历这批请求构成的基数树可取得**最优命中率**，前提是**缓存容量 $\ge$ 最大请求长度**；而「最长公共前缀优先」等价于 DFS 顺序。

**直觉**：DFS 保证「进入一棵子树后，把它走完再离开」。一条前缀被加载进缓存后，所有需要它的请求会连续执行完，之后这条前缀才不再被需要。任何打断这个连续性的顺序，都会让同一条前缀被加载多次。

**前提有多强**——三条限定，论文都写了：

1. **缓存容量 $\ge$ 最大请求长度**。这条在真实负载下很容易不成立：ShareGPT 那种 1024+ token 的 prompt，几十个并发就把缓存撑满了。而 SGLang 的内存池设计（缓存与运行中请求争抢、运行中优先）意味着**有效缓存容量是动态的、在高负载下会被压到接近零**。
2. **脚注承认**：实践中的计算与证明所描述的不同，因为**输出 token 数不可预测**会导致 KV Cache 被重算。也就是说，定理描述的是一个「所有请求长度已知」的离线模型，而真实的解码长度到结束才知道。
3. **在线场景 DFS 顺序会被打断**，调度只能在「完整基数树的增量部分」上近似 DFS。

**这个定理的实际价值**：它把「最长匹配前缀优先」这条启发式从「看起来合理」提升到「在理想条件下最优」，为工程选择提供了理论锚点。实测数据（平均达到最优命中率的 96%）说明近似质量确实不错。但**它不是一个可以用来做容量规划的定理**——参见 [CacheRoute](../2026-cacheroute/README.md) Q8，那篇论文用实测证明了解析式的缓存驻留预测在真实引擎上误差极大。

### Q5: cache-aware 调度为什么会饿死请求？

**A**: 因为调度优先级是**匹配前缀长度**，而这个量与**等待时间无关**。

考虑一个热门系统提示带来的请求流：它们的匹配前缀总是很长，因此总是排在前面。而一个前缀独特的请求（新租户、新对话）匹配长度为 0，只要热门流不断，它就永远排在队尾。这不是概率上的不幸，而是**优先级函数的结构性后果**。

论文直接承认了：

> While greedy cache-aware scheduling can achieve high throughput, it can lead to starvation. We leave its integration with other fair scheduling methods as future work.

**与 FCFS 的取舍**：FCFS 保证有界等待时间但命中率差（消融里 FCFS 明显劣于 cache-aware）；cache-aware 命中率高但无公平性保证。常见的折中是**加入等待时间因子**（例如按 `匹配长度 - α × 等待时间` 排序），或者给每个租户配额——但这些都会牺牲一部分命中率。

论文自己也留了一句余地：「在更延迟敏感的场景下，我们或许仍能容忍有限的批次重排序来改善缓存复用」——即把重排序的窗口限制住，用命中率换有界延迟。

**对照**：[CacheRoute](../2026-cacheroute/README.md) 的设计在这一点上更稳——未准入的冷尾流量始终走 power-of-two-choices，**不受亲和性影响**，因此不存在这种结构性饿死。代价是冷尾流量本来也拿不到缓存收益。

### Q6: 缓存与运行中请求共享同一内存池，意味着什么？

**A**: 意味着**缓存是机会性的，吞吐是刚性的**。

论文明确写了不预分配固定大小的缓存池：

> we let the cached tokens and the currently running requests share the same memory pool ... When enough waiting requests run, the system will **evict all cached tokens** in favor of a larger batch size.

三个后果：

1. **最坏情况不劣化**。高负载下缓存被清空，系统退化成一个没有前缀复用的普通引擎，而不是「因为缓存占着显存导致 batch 变小」。这是 §6.3 那个「开销 < 0.3%，可以默认开启」的前提——如果缓存会挤占 batch，就不敢默认开。
2. **命中率随负载反向变化**。负载越高，可用缓存越少，命中率越低。这与「负载越高越需要省算力」的直觉是相反的，是这套设计的固有性质。
3. **Theorem 3.1 的前提更难满足**（见 Q4）——有效缓存容量在高负载下会被压到远小于「最大请求长度」。

**与 vLLM 的对比**：vLLM 的 block engine 预先分配一整块显存切成物理块，块要么属于某个运行中序列、要么空闲，**没有「缓存」这个第三态**。SGLang 引入了这个第三态，并规定它的优先级最低。

### Q7: 压缩 FSM 一次解多个 token，这和 speculative decoding 是一回事吗？

**A**: **不是**，两者的信息来源完全不同。

| | 压缩 FSM | Speculative Decoding |
| --- | --- | --- |
| 多出来的 token 从哪来 | **约束本身**——FSM 上只有一条合法路径 | **草稿模型**的预测 |
| 是否需要验证 | 不需要，合法路径唯一 | 需要，用大模型验证草稿 |
| 会不会白做功 | 不会 | 会（草稿被拒绝时） |
| 适用条件 | 输出受正则/语法约束 | 任意输出 |

压缩 FSM 的逻辑是：如果 FSM 当前状态出发只有一条**单一转移边**链，那么这条链上的所有字符是**确定的**——模型的预测在这里没有任何自由度，因此根本不需要跑前向。把这些边压成一条，就能在一次前向里跨过整段。

这是「**用约束消除计算**」，而 speculative decoding 是「**用便宜的猜测换昂贵的验证**」。两者正交，可以叠加。

**实现上的难点**（附录 B）：正则表达式作用在**字符**上，模型输出的是 **token**，两者的映射不是一一对应的。同一段字符串可能有多种 token 化方式，这使得「哪些 token 是合法的」这个判断本身就不平凡。

**一个容易忽略的实测结论**：状态机必须**预处理一次并在一批请求间复用**；每个请求各自重做预处理会让吞吐**低 $2.4\times$**——这个数字比压缩本身带来的 $1.6\times$ 还大。也就是说，**预处理的摊销比压缩优化更重要**。

### Q8: 分布式 RadixAttention 的 meta-tree router 和 CacheRoute 是什么关系？

**A**: 是同一个问题的**反应式**与**计划式**两种解法。附录 A.4 那半页就是 CacheRoute 的问题陈述。

| | SGLang 附录 A.4 | [CacheRoute](../2026-cacheroute/README.md) |
| --- | --- | --- |
| 决策时机 | **每批请求到达时**在 meta-tree 上匹配 | **每个控制周期一次**，周期内路由表固定 |
| 决策依据 | 与各 worker 的共享前缀长度（亲和性） | 全局的每 key 速率分布 $\{\lambda_b\}$ |
| 一致性 | 弱一致（淘汰事件排队，低负载时才处理） | 计划本身就是过时的，用实测代价决定何时换表 |
| 负载均衡 | 论文承认存在「数据局部性 vs 并行效率」的取舍，列为 future work | 用 LPT 放置显式求解 |
| 验证规模 | 4 worker + MMLU，观测到线性扩展 | 30 个 TP2 destination（60× H100），5 组配对种子 |

论文自己点名了中间那一环：

> concurrent work from **Preble** studies data-parallel scheduling based on an early version of SGLang.

而 Preble 正是 CacheRoute 最强的基线（$76 \pm 11$ QPS，CacheRoute 是它的 $2.3\times$）。所以这条演进链是完整的：

**SGLang 附录 A.4（反应式 meta-tree）→ Preble（数据并行的反应式调度）→ CacheRoute（周期性规划）**

读完这三篇可以得到一个一般性的判断：**当决策需要全局信息（速率分布）而不只是当前状态时，反应式策略拿不到做全局优化所需的输入**。这与 CacheRoute Q1 的结论一致。

---

## 5. 可迁移的知识点

- [Prefix Caching](../../../concepts/prefix-caching.md) —— 本文把它从「手动声明的共享前缀」推进到「基于基数树的自动多级复用」，并补齐了缓存该有的索引/淘汰/调度三件套。
- [KV Cache](../../../concepts/kv-cache.md) —— 本文重新定位了它的所有权：从「请求的私有资源」变成「服务的共享缓存」。
- [约束解码](../../../concepts/constrained-decoding.md) —— 本文的压缩 FSM 是这一方向上最有影响力的效率优化。
- [分页式 KV 管理](../../../concepts/paged-kv-memory.md) —— 本文采用 page = 1 token，与 vLLM 的 block = 16 形成对照，体现同一取舍在不同优化目标下的相反答案。

---

## 6. 批判与开放问题

### 6.1 局限性

**作者自己承认的**：

- **cache-aware 调度会导致饿死**，与公平调度的结合列为 future work（§3、§8）。
- Theorem 3.1 的前提（缓存容量 $\ge$ 最大请求长度）与实践有 gap，脚注明确承认**输出长度不可预测会导致 KV 重算**。
- 在线场景下 DFS 顺序被打断，只能近似。
- 多轮对话的**长输出**场景几乎没有加速——共享少且解码时间占主导。
- 数据并行下存在「数据局部性 vs 并行效率」的取舍，未解决。
- RadixAttention 尚未跨多级存储层次（DRAM/磁盘）；不支持模糊语义匹配。

**论文没提但存在的**：

- **命中率随负载反向变化**。缓存与运行中请求共享内存池且后者优先，意味着负载越高有效缓存越小。论文报告的 50%–99% 命中率是在特定负载强度下测的，**没有给出「命中率 vs offered load」的曲线**——而这正是运维最需要的那张图。
- **没有报告任何百分位延迟**。§6.1 明确说延迟是「一次只跑一个 program、不做批处理」的平均值。这个测法排除了排队，因此测出的是**理想条件下的延迟**，与生产环境的尾延迟没有对应关系。饿死问题本应在 p99 上有明确体现，但指标体系里根本没有这一项。
- **淘汰粒度与 vLLM 的 all-or-nothing 不同，但没有讨论交互**。SGLang 按树节点淘汰，vLLM 按序列全有或全无换出。SGLang 声称与 paged attention 兼容，但两套淘汰逻辑同时存在时的行为没有分析。

### 6.2 我的质疑

- **$6.4\times$ 这个数字的基线选择需要限定。** 论文自己在脚注里说明：RadixAttention 已被部分集成进较新版本的 vLLM，因此对比用的是**更早的 v0.2.5**。这个披露是诚实的，但**摘要与结论里的 $6.4\times$ 没有带这个限定**。准确的读法是「相对于当时不具备自动前缀复用的推理引擎」，而不是「相对于同期最佳」。
- **「programs per second」这个吞吐指标对 SGLang 有结构性优势。** 一个 program 包含多次 LLM 调用，而 SGLang 的前端并行能让这些调用在一个 program 内部并发。用 p/s 计量时，前端并行的收益与 runtime 优化的收益被合并进同一个数字，**无法从端到端结果里分离出 RadixAttention 单独的贡献**。§6.3 的消融部分缓解了这个问题（"No Frontend Parallelism" 那一行），但主结果表里两者是混在一起的。对于「LM program」这个 workload，p/s 确实是自然的指标；问题在于它让**跨系统比较**变得难以解释。
- **多模态那个 $6\times$ 的基线是研究代码。** 基线是「模型作者的原始 HF Transformers 实现」，那不是一个服务系统——它没有 continuous batching、没有分页、没有任何 serving 优化。相对它取得 $6\times$ 说明不了 RadixAttention 的贡献有多大，只说明「用推理引擎跑比用研究代码跑快」。这一格的数字与其他格不可比。
- **生产数据的说服力被采样条件限制了。** Chatbot Arena 的数据只有两个模型、**每个模型单 worker**、且论文明说「部分模型流量低」。单 worker 意味着**不存在跨机路由问题**，这恰好回避了附录 A.4 里承认的最难的部分。同时没有给出 QPS、并发数、缓存容量，因此 52.4% / 74.1% 这两个命中率无法被解释——它们可能来自高复用的负载，也可能来自低负载下缓存从未被清空。
- **Theorem 3.1 的表述容易被过度引用。** 「可以取得最优命中率」这个说法在缓存容量前提之外，还隐含了「一批请求的集合是已知的」。论文对此的处理是把限定条件都写出来了，态度是端正的；但这个定理在二手转述中很容易变成「最长前缀优先是最优调度」，而去掉前提后这个陈述是错的。

### 6.3 后续可读

- **Preble** —— 基于早期 SGLang 的数据并行前缀调度，是本文附录 A.4 与 CacheRoute 之间缺失的那一环。
- **[CacheRoute](../2026-cacheroute/README.md)** —— 本仓库已有笔记，把附录 A.4 的反应式 router 换成周期性规划。
- **Mooncake** —— 接手本文 future work 的第二条：让 KV 跨多级存储层次。
- **Outlines / XGrammar** —— 约束解码方向上的后续工作，把压缩 FSM 的思路推广到上下文无关文法。
- **FlashInfer** —— 本文使用的 kernel 库，处理带前缀共享的注意力算子。
- **[vLLM / PagedAttention](../2023-vllm-pagedattention/README.md)** —— 本仓库已有笔记，本文的直接对照对象。
