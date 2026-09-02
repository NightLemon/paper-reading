---
concept: "Prefill / Decode 分离"
aliases: ["PD Disaggregation", "prefill-decoding disaggregation", "PD 分离", "阶段分离"]
tags: ["llm-serving", "architecture", "scheduling", "slo"]
papers: ["topics/llm-serving/2024-distserve", "topics/llm-serving/2025-mooncake"]
---

# Prefill / Decode 分离

> **一句话定义**：把 LLM 推理的两个阶段放到**不同的 GPU 实例**上执行——prefill 实例只产出首 token 与 KV Cache，decode 实例接过 KV 继续逐 token 生成——从而消除两者的互相干扰，并让各自独立选择资源量与并行策略。

## 为什么需要它

两个阶段的资源特征几乎相反：

| | Prefill | Decode |
| --- | --- | --- |
| 每步处理 | 整个 prompt（数百 token，可并行） | 1 个新 token |
| 计算形态 | 矩阵-矩阵乘 | 矩阵-向量乘 |
| 瓶颈 | **算力**（compute-bound） | **访存带宽**（memory-bound） |
| 对应延迟指标 | **TTFT**（首 token 时间） | **TPOT**（每输出 token 时间） |
| 偏好的批量 | 小（单条长序列就能打满 GPU） | 大（靠批量摊薄权重搬运） |

把它们放在同一批 GPU 上会产生两个问题：

1. **相互干扰**。一个 prefill step 比一个 decode step 长得多。混批时 decode 请求被 prefill 拖住，TPOT 变长；反过来加入 decode 也会抬高 TTFT。即便顺序调度，排队延迟照样传导。
2. **资源与并行策略被耦合**。两个阶段共用一套配置，只能按要求更苛刻的那个指标来配，导致为同时满足两个 SLO 而超额配置资源。

## 它是怎么工作的

```
请求 → [Prefill 实例] --KV Cache--> [Decode 实例] → 流式返回
        算力密集              网络/NVLINK            访存密集
        小 batch                                    大 batch
```

- **实例（instance）** = 恰好管理一份完整模型权重副本的资源单元；应用模型并行时可对应多张 GPU。
- 由于 decode 的 GPU 利用率低，通常**一个 decode 实例配多个 prefill 实例**，以攒出更大的 decode 批量。
- 分离后每一侧可独立选择并行策略与实例数量，按各自的 SLO 优化。

### 并行策略的选择判据

分离之后，prefill 阶段近似一个 M/D/1 队列，可用排队论分析。设单请求执行时间 $D$、到达率 $R$：

$$\text{延迟} = \underbrace{D}_{\text{执行时间}} + \underbrace{\frac{RD^2}{2(1-RD)}}_{\text{排队延迟}}$$

两种并行作用在不同的项上：

| | intra-op（张量并行） | inter-op（流水并行） |
| --- | --- | --- |
| 作用 | 把 $D$ 变小 | 把**服务节拍**变小（$D_m \approx D/\text{stage 数}$） |
| 主要收益 | 降低执行时间 | 扩大排队吞吐能力 |
| 通信需求 | 高（需 NVLINK 域） | 低 |

由此得到判据：

- **低负载**（执行时间主导）→ 偏好 **intra-op**
- **高负载**（排队延迟主导）→ 偏好 **inter-op**
- **SLO 越严** → 越偏 intra-op（因为执行时间本身可能就已超标）
- **通信开销越大**（加速系数越低）→ intra-op 越吃亏

### 跨实例传输 KV

KV Cache 必须从 prefill 侧传到 decode 侧，这是分离引入的唯一新增成本。三条工程要点：

1. **传输只发生在对应层之间**。因此可以按 inter-op stage 把实例切成「段」，**把同一 stage 的两侧段放进同一节点**，强制走节点内高速互联。
2. **用 pull 而非 push**。decode 实例按需拉取，把 prefill 实例的显存当作排队缓冲，避免突发流量撑爆 decode 侧显存，同时天然形成背压。
3. **量级估算**：KV 体积 $\propto$ 层数 × K/V 头数 × head 维度 × 序列长度 × 精度字节数 × 2。用它乘以请求率即得所需带宽。采用 GQA / MQA 的模型这一项显著更小。

## 关键性质与代价

| 收益 | 代价 |
| --- | --- |
| 干扰在定义上归零 | **每一侧各存一份模型权重** |
| 两侧可独立选并行策略与实例数 | 引入跨实例 KV 传输 |
| decode 可攒大 batch 而不牺牲 TPOT | 放置成为一个需要搜索的优化问题 |
| TTFT / TPOT 可分别按 SLO 优化 | **故障影响面被放大**：一个 decode 实例故障会波及其上游全部 prefill 实例 |

## 什么时候不该用

| 场景 | 原因 |
| --- | --- |
| **离线 / 吞吐优先** | 优化目标回到总吞吐，混批填满 GPU 才是对的；chunked-prefill 更合适 |
| **资源受限**（几张甚至一张 GPU） | 分离要求两侧各持一份权重，设计空间被压没 |
| **前缀复用率极高的负载** | prefix caching 本来就跳过了大部分 prefill，可干扰的部分变少，收益被稀释 |

反过来，**长上下文场景对分离更有利**：KV 体积随长度线性增长，而 prefill 计算随长度平方增长，传输的相对占比反而下降；同时两阶段的计算差异被进一步拉大，混批的干扰更严重。

## 常见误解

- **误解**：分离一定更省资源 → **实际**：它多存了一份权重。收益来自「消除干扰 + 按需配比」，在资源本来就紧张时可能得不偿失。
- **误解**：KV 传输是主要瓶颈 → **实际**：合理放置（同 stage 同节点）后，传输可以低到占总延迟千分之一以下。真正的约束是它对**放置自由度**的限制。
- **误解**：分离等价于「把 prefill 和 decode 排开跑」 → **实际**：顺序调度不能消除干扰，排队延迟照样互相传导。必须是**不同的物理资源**。
- **误解**：分离后可靠性不变 → **实际**：两类实例之间产生了依赖，故障会沿依赖传播，可靠性相对「colocate + 副本」是**下降**的。

## 出现在哪些论文里

- [DistServe](../topics/llm-serving/2024-distserve/README.md) —— 系统论证了分离的必要性（干扰 + 资源耦合），用排队论给出并行策略判据，用模拟器搜索最优 placement，并给出「同 stage 同节点」的放置约束把 KV 传输压进 NVLINK。
- [Mooncake（FAST'25）](../topics/llm-serving/2025-mooncake/README.md) —— 生产环境的分离实践，补上了两个新问题：长上下文下的跨节点 prefill（分块流水并行 CPP），以及过载下由「两次决策相隔一段时间」引发的 **prefill / decoding 负载反相震荡**。

## 延伸阅读

- Splitwise、TetriInfer、DéjàVu：同期采用相似分离思路的工作。
- Sarathi-Serve：chunked-prefill，是分离方案的主要竞争路线。
- [KVCache 池化与分层存储](disaggregated-kv-store.md)：把 KV 做成独立存储层，是另一种解耦方式。
- [SLO 容量与 p99 尾延迟](slo-capacity.md) · [KV Cache](kv-cache.md) · [张量并行](tensor-parallelism.md)
