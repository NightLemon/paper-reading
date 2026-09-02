# Foundations · 奠基性经典论文

奠定了某一整条技术路线的论文。读它们的目的不是追新，而是补齐后续所有工作的公共前提。

| 论文 | 年份 | 来源 | 状态 | 一句话 |
| --- | --- | --- | --- | --- |
| [Attention Is All You Need](2017-attention-is-all-you-need/README.md) | 2017 | NeurIPS'17 | ✅ 已完成 | 用纯注意力替代循环与卷积，把关联任意两个位置的顺序操作数降到 $O(1)$ |
| [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](2022-flashattention/README.md) | 2022 | NeurIPS'22 | ✅ 已完成 | 用分块 online softmax 让 $N \times N$ 中间矩阵不被物化，HBM 访问量降一个量级且结果逐位精确 |

[← 返回总索引](../../index.md)
