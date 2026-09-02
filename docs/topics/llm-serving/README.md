# LLM Serving · 推理服务与调度

覆盖推理引擎、批处理与调度、KV Cache 管理、prefix 复用、请求路由、PD 分离等方向。核心矛盾通常是**吞吐、尾延迟、显存**三者之间的取舍。

| 论文 | 年份 | 来源 | 状态 | 一句话 |
| --- | --- | --- | --- | --- |
| [Efficient Memory Management for LLM Serving with PagedAttention（vLLM）](2023-vllm-pagedattention/README.md) | 2023 | SOSP'23 | ✅ 已完成 | 把操作系统的虚拟内存分页搬到 KV Cache 上，显存有效利用率从 20% 提到接近 100% |
| [SGLang: Efficient Execution of Structured Language Model Programs（RadixAttention）](2024-sglang-radixattention/README.md) | 2024 | NeurIPS'24 | ✅ 已完成 | 把 KV Cache 当成基数树上的 LRU 缓存，自动多级前缀复用 + 最长匹配前缀优先调度 |
| [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving](2024-distserve/README.md) | 2024 | OSDI'24 | ✅ 已完成 | 把 prefill 与 decode 拆到不同 GPU 实例，消除干扰并各自按 TTFT/TPOT 选并行策略 |
| [Orca: A Distributed Serving System for Transformer-Based Generative Models](2022-orca/README.md) | 2022 | OSDI'22 | 🚧 已派工单 | 待精读 |
| [CacheRoute: Planned Prefix-Affinity Routing for Large-Scale LLM Serving](2026-cacheroute/README.md) | 2026 | arXiv | ✅ 已完成 | 用周期性路由计划把 prefix 亲和性与预期负载一起规划，SLO 容量达到最强基线的 2.3 倍 |

[← 返回总索引](../../index.md)
