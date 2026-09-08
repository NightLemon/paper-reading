---
concept: "流水并行"
aliases: ["Pipeline Parallelism", "PP", "microbatch pipeline", "1F1B", "weight stashing"]
tags: ["llm-training", "parallelism", "scheduling", "memory"]
papers: ["topics/llm-training/2019-gpipe-pipedream"]
---

# 流水并行

> **一句话定义**：把连续模型层划成多个 stage，让不同数据片段在不同 stage 上交错执行，通过边界激活与梯度传递合作完成前后向。

## 为什么需要它

按层切分可以让整个模型分布到多张设备，但若同一时间只有一批数据，前向和反向的依赖会让许多 stage 空等。引入多个在飞 microbatches 或 minibatches，可以让一个 stage 计算当前批时，其他 stage 处理不同批。

训练还需要保留反向所需的激活与权重版本；流水调度不能只看设备是否忙碌，还要看梯度对哪个前向函数求导、何时更新参数。

## 它是怎么工作的

### 四个独立的设计问题

| 问题 | 需要决定什么 |
| --- | --- |
| 分区 | 哪些层属于哪个 stage，stage 是否复制，如何映射互联拓扑 |
| 执行顺序 | 当前做前向还是反向，有多少数据同时在飞 |
| 更新边界 | 哪些梯度累积为一次 optimizer update，何时同步 |
| 状态版本 | 每批前向、重算、反向使用哪份权重，什么时候回收 |

### GPipe 与原始 PipeDream

GPipe 将一个 minibatch 拆成多个 microbatches，使用同一轮参数计算，累积完成后同步更新。原始 PipeDream 让多个 minibatches 连续流动，在 stage 稳态交替做一个前向、一个反向（1F1B），并在反向后产生更新。

1F1B 本身只描述调度顺序，也可以搭配 minibatch 末尾同步更新。因此“使用 1F1B”不足以判断是否存在旧梯度。

### 气泡模型

对均衡的 $K$ 个 stages、$M$ 个 microbatches、每 stage 前向与反向时间 $t_f,t_b$，忽略通信和重算细节时，同步流水可近似为：

$$T\approx(M+K-1)(t_f+t_b),\quad
f_{bubble}\approx\frac{K-1}{M+K-1}.$$

如果把空闲时间除以有用工作时间，则比例是 $(K-1)/M$；引用时要注明分母。固定 minibatch 大小下，增大 $M$ 会缩小 microbatch，可能降低矩阵乘效率；固定 microbatch 下，增大 $M$ 又改变有效 batch。

### 权重版本与梯度

weight stashing 保存 stage 对某批前向使用的权重，在对应反向时恢复，避免求导函数不匹配。它可以让各 stage 沿不同历史版本计算；可选 vertical sync 进一步让一批跨 stage 使用统一旧版本。

即使所有 stage 都用同一个旧 $w^{t-d}$，把其梯度应用到当前 $w^t$ 仍是陈旧更新：

$$w^{t+1}=w^t-\eta\nabla f(w^{t-d}),$$

与同步 SGD 的 $d=0$ 不同。需要用目标任务的学习结果验证，而不能仅以梯度局部可计算作为同语义的证明。

### 显存与通信

长期参数 / 优化器、在飞批次的边界激活、重算工作集、权重版本和通信缓冲都要计入。重算可以减少内部激活保存，多版本流水则可能增加参数驻留。

stage 边界传递前向激活和反向输入梯度；若 stage 有 replicas，还需相应同步。权重大、边界小的切分可能减少 DP 的全参数通信，边界激活很大的网络则可能更适合 DP。

## 关键性质与代价

| 收益 | 代价与条件 |
| --- | --- |
| 将整个模型按层分到多个设备 | 单 stage 的当前工作集仍必须可执行 |
| 让不同批的计算与通信重叠 | 需要足够在飞数据，保存其状态 |
| 边界通信较集中 | 划分不均、慢 stage 或网络竞争会产生气泡 |
| 可以结合 TP、DP 和状态分片 | 通信组、有效 batch、版本和峰值须共同设计 |
| 同步调度可保留 minibatch 梯度语义 | 跨样本层、随机状态和损失缩放需要特别处理 |

## 常见误解

- **误解**：按层切开就自动获得高利用率 → **实际**：没有多批交错时仍可能只有一个 stage 在工作。
- **误解**：1F1B 就是异步更新 → **实际**：它只规定前后向顺序，update barrier 是另一选择。
- **误解**：weight stashing 消除了所有陈旧性 → **实际**：它保证局部前后向版本匹配，梯度仍可能过时。
- **误解**：microbatches 越多越好 → **实际**：还要权衡小矩阵效率、消息开销、状态存储与有效 batch。
- **误解**：流水总比 DP 少通信 → **实际**：要比较参数量与切分边界激活量，ResNet 类网络可能更适合 DP。
- **误解**：吞吐提高就等于更快达到目标质量 → **实际**：batch 和旧梯度会影响学习进度，应报告 time-to-quality。

## 出现在哪些论文里

- [GPipe / PipeDream 对比精读](../topics/llm-training/2019-gpipe-pipedream/README.md) —— GPipe 用微批梯度累积与重算维持同步更新；PipeDream 将拓扑感知划分、1F1B-RR 与权重版本管理组合。对比中保留了版本、硬件、更新语义和质量目标的差异。

## 延伸阅读

- [激活检查点](activation-checkpointing.md) · [张量并行](tensor-parallelism.md) · [ZeRO 状态分片](zero-state-sharding.md)。
- [Megatron 3D 并行](https://arxiv.org/abs/2104.04473)：同步流水与多维并行的后续设计。
- [Sarathi-Serve](../topics/llm-serving/2024-sarathi-serve/README.md)：推理流水仍会因批次运行时间差异产生气泡，原理与训练的版本陈旧性不同。
