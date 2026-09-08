# LLM Training · 训练系统与并行策略

数据/张量/流水/序列并行、显存优化、通信压缩、算子融合与 IO 感知 kernel。

| 论文 | 年份 | 来源 | 状态 | 一句话 |
| --- | --- | --- | --- | --- |
| [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](2019-megatron-lm/README.md) | 2019 | arXiv | ✅ 已完成 | 「第一个 GEMM 按列切、第二个按行切」，把每层通信压到前向 2 次 + 反向 2 次 all-reduce |
| [GPipe / PipeDream：流水并行的调度与更新语义（对比精读）](2019-gpipe-pipedream/README.md) | 2019 | NeurIPS'19 / SOSP'19 | ✅ 已完成 | 对照按 minibatch 排空后同步更新与跨 minibatch 流水，解释气泡、激活占用和权重版本语义 |
| [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](2020-zero/README.md) | 2020 | SC'20 | ✅ 已完成 | 在数据并行组内逐级分片训练状态，再处理激活、缓冲和碎片，扩大可运行模型与 batch |

候选见 [待读队列 · 阶段 3](../../reading-list.md#阶段-3--训练系统与并行策略)。

[← 返回总索引](../../index.md)
