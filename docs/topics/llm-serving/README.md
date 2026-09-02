# LLM Serving · 推理服务与调度

覆盖推理引擎、批处理与调度、KV Cache 管理、prefix 复用、请求路由、PD 分离等方向。核心矛盾通常是**吞吐、尾延迟、显存**三者之间的取舍。

| 论文 | 年份 | 来源 | 状态 | 一句话 |
| --- | --- | --- | --- | --- |
| [Efficient Memory Management for LLM Serving with PagedAttention（vLLM）](2023-vllm-pagedattention/README.md) | 2023 | SOSP'23 | ✅ 已完成 | 把操作系统的虚拟内存分页搬到 KV Cache 上，显存有效利用率从 20% 提到接近 100% |
| [Orca: A Distributed Serving System for Transformer-Based Generative Models](2022-orca/README.md) | 2022 | OSDI'22 | 🚧 已派工单 | 待精读 |
| [CacheRoute: Planned Prefix-Affinity Routing for Large-Scale LLM Serving](2026-cacheroute/README.md) | 2026 | arXiv | ✅ 已完成 | 用周期性路由计划把 prefix 亲和性与预期负载一起规划，SLO 容量达到最强基线的 2.3 倍 |

[← 返回总索引](../../index.md)
