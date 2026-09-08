---
title: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
authors: ["Jianlin Su", "Yu Lu", "Shengfeng Pan", "Ahmed Murtadha", "Bo Wen", "Yunfeng Liu"]
affiliation: "Zhuiyi Technology Co., Ltd., Shenzhen"
venue: "arXiv preprint"
year: 2021
arxiv: "2104.09864v5"
url: "https://arxiv.org/abs/2104.09864v5"
topic: "foundations"
tags: ["transformer", "rope", "positional-encoding", "relative-position", "long-context"]
concepts: ["rotary-position-embedding", "positional-encoding", "self-attention", "kv-cache"]
status: "done"
rating: 5
read_date: "2026-09-08"
---

# RoFormer: Enhanced Transformer with Rotary Position Embedding

> **一句话结论**：对 query 与 key 的二维分量对施加位置相关旋转，使两者内积中的位置关系通过旋转角差自然变为相对距离；这给出一个高效的位置编码构造，但长上下文泛化仍需实验验证。

---

## 1. 元信息

| 项目 | 内容 |
| --- | --- |
| 作者 / 机构 | Jianlin Su、Yu Lu、Shengfeng Pan、Ahmed Murtadha、Bo Wen、Yunfeng Liu；Zhuiyi Technology（追一科技），深圳 |
| 归档年份 | 2021，按首次 arXiv 公开时间归档；本笔记按 v5 精读 |
| 所读版本 | arXiv v5 提交日期 2023-11-08；PDF 首页印刷日期为 2023-11-09，二者均保留，章节与公式以该版本为准 |
| 原文 | [PDF](paper.pdf) · [HTML](paper.html) · [固定版本页面](https://arxiv.org/abs/2104.09864v5) |
| HTML 阅读 | 样式与部分图像从 arXiv 联网加载；离线阅读使用同目录 PDF |
| 代码 | [ZhuiyiTechnology/roformer](https://github.com/ZhuiyiTechnology/roformer) · [Hugging Face 模型文档](https://huggingface.co/docs/transformers/model_doc/roformer) |
| 关键词 | rotary position embedding · relative position · complex representation · linear attention |
| 前置知识 | [Self-Attention](../../../concepts/self-attention.md)、[Positional Encoding](../../../concepts/positional-encoding.md)、二维旋转与复数内积 |

**术语区分**：RoPE 是位置编码方法，RoFormer 是采用该方法的 Transformer 模型。本文的实验模型不等同于今天所有采用 RoPE 的 decoder-only LLM；后来的位置插值、频率缩放也不是本文实验的一部分。

---

## 2. 摘要速览（5 分钟版）

### 2.1 要解决的问题

标准 self-attention 仅根据内容计算分数，需要另外提供位置。Transformer 原论文在输入 embedding 上加正余弦向量，希望模型容易学习相对关系；许多后续方案直接给 token 对添加相对位置项，但这些成对项往往难以保留线性 attention 的可分解计算形式。

RoFormer 先设定一个更明确的目标：query 与 key 各自编码绝对位置，但内积里的位置依赖只通过两者的距离出现。这样既保留逐位置变换，也获得相对位置关系。

### 2.2 核心方法

将每个头的偶数维向量拆成二维分量对。对位置 $m$ 的 query、位置 $n$ 的 key 分别乘以旋转 $R_m$、$R_n$，旋转角为位置乘以预设频率。由于旋转矩阵正交：

$$\tilde q_m^T\tilde k_n=(R_mq_m)^T(R_nk_n)
=q_m^TR_{n-m}k_n.$$

不同二维对使用几何递减频率。实现只需成对交换、符号变化和逐元素 sin/cos 运算，无需真正构造稠密大矩阵。本文的标准 softmax attention 中，位置主要作用于 Q/K，V 保留内容表示。

**关键限定**：上式仍依赖内容 $q_m,k_n$；“只依赖相对位置”指**显式的位置项**，不是说注意力分数由距离唯一决定。

### 2.3 主要结果

原文 §4 统一说明实验使用带 4×V100 的两台云服务器，但没有为每项子实验单列 GPU 分配，因此下表保留这一硬件说明，不将结果当作硬件效率评测。

| 原文位置 | 模型、任务与条件 | 结果与边界 |
| --- | --- | --- |
| §4.1、Table 1 | WMT14 En–De，Transformer-base 对 RoFormer，同训练设置、37K BPE、最后 5 个 checkpoint 平均、beam=4、length penalty=0.6 | BLEU **27.3 → 27.5**，增加 0.2；无误差条 |
| §4.3、Table 2 | BERT-base-uncased 对 RoFormer，GLUE 六项任务、3 epochs、最大长度 512、batch 32，学习率搜索 2–5×10⁻⁵ | MRPC F1 **88.9 → 89.5**，但 SST-2 accuracy **93.5 → 90.7**；不同任务有升有降 |
| §4.5、Table 5 | 中文 CAIL2019-SCM，RoFormer 相同系列预训练模型，比较输入截断 512 与 1024；预训练已包含长度 1536 的阶段 | test accuracy **68.29% → 69.79%**，增加 **1.50 个百分点**；证明该配置中保留更长输入有益，不能证明零训练长度外推 |

另有 Figure 3 的 MLM / Performer training loss 曲线，显示部分设置下随训练步数更快下降；原文没有据此报告端到端训练时间加速比。

### 2.4 我的评价

评分 ⭐⭐⭐⭐⭐，主要来自结构构造的清晰性和可迁移性。把“希望内积具有相对位置性质”写成约束，再选择可组合的旋转变换，比记住一段 sin/cos 代码更有价值。

实验本身远比后来的普及范围窄。精读时需要保留 GLUE 的退化项、长文本训练长度和理论衰减条件，才能分清构造的数学优点与模型能力的实证边界。

---

## 3. 细读

> 本节 3.1–3.5 对应原文 §1–§5；理论推导位于原文 §3.4，所读版本没有独立附录。下文用 $d_h$ 表示单头维度，统一采用列向量记法、零起始频率编号，避免原文公式中个别下标与转置排版不一致影响推导。

### 3.1 Introduction

引言从不同网络如何获得顺序信息切入：RNN 来自递归顺序，CNN 来自局部结构等因素，attention 需要显式处理位置信息。后半段把已有位置方法按绝对 / 相对分类，并指出对 token 对逐项添加相对项会妨碍某些线性 attention 的结合律重排。

作者选择“乘一个旋转”作为可分离的位置变换，每个 token 可以独立得到带位置的 Q/K，再通过内积组合。这个选择同时服务两个目标：显式相对位置性质，以及与高效计算形式的兼容。

> **批注**：兼容某类 linear attention 只说明可构造相应公式。它并不意味着原本的精确 softmax attention 因加 RoPE 就从二次变线性，也不意味着所有相对位置方案都在所有线性架构上不可能实现。

### 3.2 Background and Related Work

#### 2.1 Preliminary

Equations 1–2 将问题形式化为 $q_m=f_q(x_m,m)$、$k_n=f_k(x_n,n)$、$v_n=f_v(x_n,n)$，然后做常规 scaled dot-product attention。这个写法把内容与位置的结合方式留成待设计函数。

后文关注 QK 内积，因为它直接影响 token 间的信息选择；V 是否携带位置是另一个设计选择，不能从 Q/K 的性质自动推出。

#### 2.2 Absolute position embedding

典型方法是先加位置向量，再投影：$q_m=W_q(x_m+p_m)$。学习式位置表的长度受表大小限制；正余弦函数在任意位置可计算，但模型是否能理解没见过的长度需要额外验证。

将加法方案的 QK 展开，会出现内容–内容、内容–位置、位置–内容、位置–位置四项。某些正余弦性质使相对位置容易表达，却不保证经过任意训练出的投影后，这四项只依赖 $m-n$。

#### 2.3 Relative position embedding

原文依次回顾 Shaw 等方法的截断相对向量，以及 Transformer-XL、T5 等对展开项的改写。共有问题是：位置关系往往作为 token 对之间的额外成分进入分数，而不是先分别得到两个独立变换后的向量。

RoPE 接下来改为直接要求 $f_q,f_k$ 的内积满足相对关系。这个论证说明设计动机，没有在这一节对所有位置编码方法做统一质量或速度评测。

> **批注**：选择位置方案时要比较注入位置、参数、计算形式与长度行为。单说“相对位置”不足以区分这些方案，更不能据此预测全部实验结果。

### 3.3 Proposed approach

#### 3.1 Formulation

Equation 11 的目标是：

$$\langle f_q(x_m,m),f_k(x_n,n)\rangle=g(x_m,x_n,m-n).$$

这是对显式位置依赖的约束。它仍允许 $g$ 随两端内容变化。对于多层模型，输入本层的隐藏表示本身也带着先前上下文信息，所以该局部恒等式不能直接推出整个模型对任意序列编辑都保持输出不变。

#### 3.2 Rotary position embedding

**3.2.1 A 2D case**。令投影后的内容向量为 $q=(q_1,q_2)^T$、$k=(k_1,k_2)^T$，二维旋转为：

$$R(\phi)=\begin{pmatrix}\cos\phi&-\sin\phi\\\sin\phi&\cos\phi\end{pmatrix}.$$

在位置 $m,n$ 分别得到 $R(m\theta)q$、$R(n\theta)k$。直接利用矩阵乘法：

$$R(m\theta)^TR(n\theta)=R((n-m)\theta).$$

因此带位置内积只通过旋转角差包含位置。复数写法与之等价：把向量看作 $q_1+iq_2$，乘 $e^{im\theta}$ 完成旋转；两端相乘并对 key 取共轭，再取实部，就得到相同的实内积（Equations 12–13）。

**3.2.2 General form**。把偶数 $d_h$ 维分成 $d_h/2$ 对，各自旋转：

$$R_m=\operatorname{diag}\big(R(m\theta_0),\ldots,R(m\theta_{d_h/2-1})\big),\qquad
\theta_j=10000^{-2j/d_h}.$$

与原文 Equation 15 的一开始于 1 的频率编号等价。每个块都满足旋转群的相加性质，整个块对角矩阵也满足 $R_m^TR_n=R_{n-m}$。

按列向量、带投影完整写出 Equation 16 的关系应为：

$$\tilde q_m^T\tilde k_n
=x_m^TW_q^TR_m^TR_nW_kx_n
=x_m^TW_q^TR_{n-m}W_kx_n.$$

原文右端存在转置 / 下标排版不齐，以上从左端直接展开，不将其当作另一个算法。**投影后旋转**很重要，通常 $R_mW_q\ne W_qR_m$。

Figure 1 对应的就是逐二维对的这套运算。不同频率让表示包含快慢变化的位置尺度，但论文没有证明几何频率底数 10000 在所有任务上最优。

#### 3.3 Properties of RoPE

**Long-term decay**。作者声称随相对距离增加存在长期衰减，并在 §3.4.3 用 Abel 变换与 Figure 2 分析。这里应先区分“某个频率和相关上界的衰减趋势”和“每对 token 的 attention 权重单调下降”。后一种说法并不成立，详见 Q4。

**RoPE with linear attention**。常见 linear attention 将相似度写成 $\phi(q_m)^T\varphi(k_n)$，通过结合律先累加 KV。Equation 19 将旋转乘到特征映射**之后**：

$$o_m^T=
\frac{(R_m\phi(q_m))^T\sum_n(R_n\varphi(k_n))v_n^T}
{\phi(q_m)^T\sum_n\varphi(k_n)}.$$

这里输出写成 $o_m^T$，以保持 Q/K/V 均为列向量的约定。分子可以先累加 $(R_n\varphi(k_n))v_n^T$，因此保留相应的可分解形式；若是因果版本，求和范围改成合法前缀。

**分母保持未旋转形式**是原文的明确选择，以避免旋转后的项相消导致分母为零；分子可能有负项，所以系数不再是严格概率归一化的 attention 权重。这也说明“RoPE 与 linear attention 兼容”不等同于“原 softmax 注意力的等价线性算法”。

#### 3.4 Theoretical Explanation

**3.4.1 Derivation of RoPE under 2D**。作者把二维向量分成模长和相位，寻找让位置关系只依赖差值的构造。一种自然解是保持模长不变，并让 Q/K 增加同一个位置相位函数 $\phi(m)$。

若相邻位置的相位差与绝对位置无关，就有 $\phi(m+1)-\phi(m)=\theta$，对整数位置推出 $\phi(m)=m\theta+\gamma$。选择公共初相位 $\gamma=0$，得到线性随位置旋转的形式。

这里是**给定约束下的一种构造路径**；保持模长是作者采用的解，不能读成证明旋转是全部可能位置编码的唯一形式。

**3.4.2 Computational efficient realization of rotary matrix multiplication**。Equation 34 把块矩阵改成逐元素运算。若一对分量为 $(a,b)$：

$$a'=a\cos(m\theta)-b\sin(m\theta),\qquad
b'=a\sin(m\theta)+b\cos(m\theta).$$

```text
for 每个位置 m 和每个二维分量对 j:
    angle = m * theta[j]
    a, b = 原来的两个分量       # 必须同时读取旧值
    out[2*j]   = a*cos(angle) - b*sin(angle)
    out[2*j+1] = a*sin(angle) + b*cos(angle)
```

一对只做常数次运算，总额外算术随 $d_h$ 线性增长；不需要进行一个 $d_h\times d_h$ 的稠密矩阵乘。具体实现的维度配对顺序必须和权重约定一致，不能随意交换。

**3.4.3 Long-term decay of RoPE**。把每对维度合并成复数，令 $r=d_h/2$、$h_j=q_j\overline{k_j}$、$\Delta=m-n$，则内积是：

$$\operatorname{Re}\sum_{j=0}^{r-1}h_je^{i\Delta\theta_j}.$$

定义部分和 $S_j=\sum_{i=0}^{j-1}e^{i\Delta\theta_i}$，$S_0=0$，并补 $h_r=0$。Abel 变换给出原文 Equations 36–37：

$$\sum_{j=0}^{r-1}h_j(S_{j+1}-S_j)
=-\sum_{j=0}^{r-1}S_{j+1}(h_{j+1}-h_j),$$

$$\left|\sum_{j=0}^{r-1}h_je^{i\Delta\theta_j}\right|
\le\max_j|h_{j+1}-h_j|\sum_{j=0}^{r-1}|S_{j+1}|.$$

Figure 2 展示了所选几何频率下，归一化部分和量随距离的总体衰减趋势。这个上界还乘着内容相关的变化量，且频率项本身会振荡；它没有证明任意内容向量的内积都随距离严格单调，更没有直接证明 softmax 之后的权重单调。

> **批注**：把不等式、数值图和最终模型行为分开阅读。论文的构造恒等式很干净，衰减性质的表述则需要更谨慎；后者不能承担全部长文本效果的因果解释。

### 3.4 Experiments and Evaluation

#### 4.1 Machine Translation

**4.1.1 Experimental Settings**：WMT14 英德，约 450 万句对，基线 Transformer-base。

**4.1.2 Implementation details**：Fairseq / PyTorch，37K 联合 BPE，Adam $\beta_1=0.9,\beta_2=0.98$，学习率从 $10^{-7}$ 升到 $5\times10^{-4}$ 再按步数平方根倒数衰减，label smoothing 0.1；最后 5 个 checkpoint 平均，beam 4、length penalty 0.6。

**4.1.3 Results**：Table 1 为 27.3 对 27.5 BLEU。它支持同设置下的小幅改善；未提供多 seed 方差，所以不能仅凭 0.2 的差异声称显著、稳定领先。

#### 4.2 Pre-training Language Modeling

**4.2.1 Experimental Settings**：BookCorpus 与 Wikipedia，train / validation 为 8:2，基线称为 bert-base-uncased，比较 MLM loss。

**4.2.2 Implementation details**：batch 64、最大长度 512、100K steps，AdamW、学习率 $10^{-5}$。

**4.2.3 Results**：Figure 3 左图中 RoFormer 的 training loss 下降更快。这是按训练步数比较的收敛曲线，不是按墙钟时间报告的训练吞吐。

**需要保留的歧义**：原文多次将所替换的 BERT 位置编码称为 sinusoidal，而标准 BERT 使用学习式绝对位置嵌入。仅据论文文字不能完全确定其实验基线是否经过其他改动；本笔记记录作者描述，不据此改写 BERT 的标准定义。

#### 4.3 Fine-tuning on GLUE tasks

**4.3.1 Experimental Settings**：MRPC、SST-2、QNLI、STS-B、QQP、MNLI；MRPC / QQP 用 F1，STS-B 用 Spearman，其他用 accuracy。

**4.3.2 Implementation details**：Hugging Face Transformers，3 epochs、最大长度 512、batch 32，学习率在 $2,3,4,5\times10^{-5}$ 中选择；按原文表述报告验证集 best-averaged results，但未清楚列出统一的随机种子数量。

**4.3.3 Results**（Table 2）：

| 模型 | MRPC | SST-2 | QNLI | STS-B | QQP | MNLI m/mm |
| --- | --- | --- | --- | --- | --- | --- |
| BERT | 88.9 | 93.5 | 90.5 | 85.8 | 71.2 | 84.6 / 83.4 |
| RoFormer | 89.5 | 90.7 | 88.0 | 87.0 | 86.4 | 80.2 / 79.8 |

三个任务改善，另外三个下降。尤其 QQP 的差异大，表内没有额外分析能说明其原因。原文在此使用 “significantly” 描述提升，但没有给显著性检验，本笔记按数值呈现。

> **批注**：这是反对“相对位置总比绝对位置更好”的直接证据。结构性质提供归纳偏置，任务结果仍由数据、训练和实现共同决定。

#### 4.4 Performer with RoPE

**4.4.1 Implementation details**：Enwik8 字符级任务，12 层、768 维、12 heads 的 Performer，学习率 $10^{-4}$、batch 128、最大长度 1024；采用原文链接的 performer-pytorch 实现。

**4.4.2 Results**：Figure 3 右图在相同训练步数下显示更低 loss。它为 linear attention 加入旋转提供了实验例子；实验没有将 Performer 输出与精确 softmax attention 做等价性验证，也没有报告完整速度–质量扫描。

#### 4.5 Evaluation on Chinese Data

**4.5.1 Implementation**：在 WoBERT 上将绝对位置嵌入改为 RoPE。Table 3 指出 RoFormer / WoBERT 使用 word-level tokenization，BERT / NEZHA 为 character-level。跨这些模型比较时，tokenization 也是变化因素，不能把全部差异归给位置编码。

**4.5.2 Pre-training**：约 34GB 中文 Wikipedia、新闻和论坛数据，按 Table 4 连续进行六阶段训练：

| 阶段 | 最大长度 | Batch size | 步数 | Loss | Accuracy |
| --- | --- | --- | --- | --- | --- |
| 1 | 512 | 256 | 200K | 1.73 | 65.0% |
| 2 | 1536 | 256 | 12.5K | 1.61 | 66.8% |
| 3 | 256 | 256 | 120K | 1.75 | 64.6% |
| 4 | 128 | 512 | 80K | 1.83 | 63.4% |
| 5 | 1536 | 256 | 10K | 1.58 | 67.4% |
| 6 | 512 | 512 | 30K | 1.66 | 66.2% |

各阶段同时改变训练进度、长度与部分 batch size，表中 accuracy 变化不能被当作仅改变长度的独立消融。尤其是模型**已经训练过 1536 长度**，这决定了后续 1024 输入实验的解释。

**4.5.3 Downstream Tasks & Dataset**：CAIL2019-SCM 含 8964 个案件三元组 $(A,B,C)$，判断 $(A,B)$ 是否比 $(A,C)$ 更相近，train / validation / test 按 6:2:2 划分。任务依赖长案件事实描述，输入截断确实可能丢失重要证据。

**4.5.4 Results**（Table 5）：

| 模型及截断长度 | Validation accuracy | Test accuracy |
| --- | --- | --- |
| BERT-512 | 64.13% | 67.77% |
| WoBERT-512 | 64.07% | 68.10% |
| RoFormer-512 | 64.13% | 68.29% |
| RoFormer-1024 | 66.07% | 69.79% |

同一 RoFormer 系列从 512 改到 1024，test 增加 **1.50 个百分点**；对 WoBERT-512 则是 **1.69 个百分点**。原文正文称相对 WoBERT 增加 1.5%，与表格不完全一致，这里按表重算并注明分母。

该结果表明更长截断在此配置有用，但缺少 WoBERT-1024 同长度对照；同时 1024 没有超过预训练到过的 1536。它不验证“只训练 512 后零适应外推到 1024”。

**4.5.5 Limitations of the work**：作者承认尚不能充分解释为何更快收敛，也不能从已有衰减性质充分解释为何长文本任务更好。模型仍基于 Transformer，需要预训练硬件资源。

### 3.5 Conclusions

结论回到旋转构造：独立编码绝对位置、内积呈现相对关系，在部分英文和中文实验中表现出更快的按步收敛和更好的长文本任务结果。

数学构造、特定训练设置下的优化行为、下游任务质量是三个层次。本文对第一个层次给出直接推导，对后两者主要提供经验观察。

---

## 4. 关键问题解析

### Q1: RoPE 是绝对位置编码还是相对位置编码？

**A**：它用绝对位置决定每个向量的旋转角，但在两端内积中体现相对位置。二者描述不同阶段，并不矛盾。

位置 $m$ 的 query 需要知道 $m$ 才能计算 $R_mq$；位置 $n$ 的 key 同理。当两者相乘，$R_m^TR_n=R_{n-m}$，显式位置只剩差值。

仍需强调内容：$q^TR_{n-m}k$ 同时依赖 $q,k$。在固定内容向量下共同平移两个位置，分数保持；若改变了上下文本身导致隐藏表示变化，则该恒等式不足以保证模型输出相同。

### Q2: 为什么一定先投影再旋转？在 embedding 上旋转一次不行吗？

**A**：本文构造让旋转直接位于内积的两端，因此 $R_m^TR_n$ 可以相邻合并。若先旋转输入再投影，会出现：

$$x_m^TR_m^TW_q^TW_kR_nx_n.$$

一般训练出的 $W_q^TW_k$ 不与所有旋转交换，两个旋转无法直接合并为 $R_{n-m}$。这与本文保证的形式不同。

对每层 Q/K 应用旋转，使该层内积都拥有这套结构；仅在最底层处理一次输入不能自动保证所有后续层满足同样关系。V 不需要按本文这套机制旋转，位置已经通过注意力权重影响信息选择。

### Q3: 正交旋转保持范数，为什么还能改变注意力？

**A**：它保持单个向量长度，改变的是不同位置向量之间的夹角。若两端旋转角相同，内积保持；位置不同就有相对旋转，内积可以变化。

**本笔记二维例子**：$q=k=(1,0)$，两端旋转角分别为 $m\theta,n\theta$，则内积为 $\cos((n-m)\theta)$。两端范数始终都是 1，分数却会随角差改变。

因此“范数保持”只说明位置变换本身不会直接放大或缩小 Q/K 的长度；它不保证 softmax 分布不变、模型训练稳定，也不代表全网络梯度范数不变。

### Q4: 论文的 long-term decay 是否说明越远的 token 权重越小？

**A**：不能这样理解。仅上一题的二维例子就会周期性出现高低值，不是严格单调下降。高维频率叠加可能表现总体衰减趋势，但仍带振荡，内容也会改变各频率的权重。

原文 §3.4.3 证明一个与部分和 $S_j$ 有关的上界，Figure 2 展示该量在所画距离范围的变化。上界减小不等于被约束的实际值逐点单调；softmax 还依赖同一 query 的所有候选 key，进一步改变相对权重。

所以更准确的结论是：所选多频旋转提供与距离有关的结构和总体衰减分析。模型仍能通过内容在远处形成强关联，论文也未证明任意远距离的永久衰减保证。

### Q5: RoPE 能计算任意位置，为什么还需要后来的长上下文扩展方法？

**A**：函数可计算解决的是位置表越界，模型泛化解决的是训练分布外的表示与决策。即使 sin/cos 对很大的位置有定义，模型也可能没学会处理新的距离和相位组合。

本文中文训练含最大长度 1536 的阶段，而 Table 5 最长只测 1024，不能作为超训练长度验证。要证明外推，必须固定训练长度与参数，明确测试超过该长度的任务，并设置同等输入信息量的对照。

后来的位置插值、频率缩放会改变位置到相位的映射，常常还涉及适应训练。它们可以作为本文的后续阅读，但本笔记不将其效果归入 RoFormer v5。

### Q6: 为什么 RoPE 能用于 linear attention？分母为什么不跟着转？

**A**：旋转作用在每个位置独立得到的特征向量上，所以分子可以写成 $(R_m\phi(q_m))^T\sum_n(R_n\varphi(k_n))v_n^T$。先累加后乘的结合律仍然可用，避免为每对位置单独构造相对偏置。

旋转之后，原来非负的特征可能出现负分量。若分母也使用旋转内积，正负抵消可能使其接近或等于零。Equation 19 保留未旋转分母来避免新增的相消风险；原始特征映射的分母是否非零仍取决于相应条件。

因此分子的系数可能为负，整体不是概率权重的简单归一化。论文直接承认这一点，不能沿用 softmax 的“非负凸组合”解释这条公式。

### Q7: RoPE 与 KV Cache、chunked prefill 有什么关系？

**A**：下面是根据旋转公式得到的工程推论，原文没有开展 KV serving 实验。常见实现缓存已经应用 RoPE 的 key，位置 $n$ 的 key 一旦算好，后续 query 使用自己的位置旋转后直接与它相乘。新增 token 不要求旧 key 每步重新旋转。

chunked prefill 必须延续整条请求的位置编号。如果每个 chunk 都从零开始旋转，跨 chunk 的相对位置就变了。例如第二块真正位于位置 256，却按位置 0 处理，它与前块的距离关系将不符合原计算。

缓存复用也要考虑模型、前缀、位置编号和 RoPE 频率配置。即使对固定的原始 key 可以代数上补一个相位差，缓存的深层表示还依赖之前上下文，不能据此任意移动文本块就复用全部 KV。详见 [KV Cache](../../../concepts/kv-cache.md) 和 [Chunked Prefill](../../../concepts/chunked-prefill.md)。

### Q8: 怎样设计实验来验证本文没有证实的“长度外推”？

**A**：先规定最大训练长度 $L$，记录全部预训练和适应训练阶段，然后冻结模型，在明显超过 $L$ 的长度上测试。测试应包括需要远距离信息的任务，避免仅仅加长不相关文本。

对照至少控制 tokenization、可见输入信息、训练计算、模型尺寸和任务划分；报告随长度变化的质量、失败率、计算时间和显存。若使用插值或继续训练，应作为单独实验组。

这是本笔记提出的验证方案，本次未执行。它与 Table 5 的差别在于：Table 5 改变截断长度、缺少同长度 WoBERT 对照，且未超过已有训练长度，所以回答的是更窄的问题。

---

## 5. 可迁移的知识点

- [Rotary Position Embedding](../../../concepts/rotary-position-embedding.md) —— 通过独立旋转与内积合成相对位置，区分局部恒等式和模型泛化。
- [Positional Encoding](../../../concepts/positional-encoding.md) —— 对照输入加法、成对偏置与 Q/K 旋转的注入位置。
- [Self-Attention](../../../concepts/self-attention.md) —— RoPE 改变 QK 分数的位置结构，标准 softmax attention 的基本计算仍保留。
- [KV Cache](../../../concepts/kv-cache.md) —— 由位置编号与旋转参数一致性推导缓存复用条件；属于本笔记的系统联系。

**可迁移的方法**：先写出想要的不变性，再寻找可组合的变换；把精确代数性质、数值观察与下游效果分别验证；用相同输入信息量做公平比较，避免把“看得更多”全部归因于“位置表达更好”。

---

## 6. 批判与开放问题

### 6.1 作者承认的局限

- §4.5.5 没有充分解释更快收敛的原因。
- 已有长期衰减性质不能完整解释长文本任务优于其他位置方法的原因。
- RoFormer 仍需要 Transformer 的训练资源；RoPE 本身没有改变标准 attention 的二次复杂度。

### 6.2 我的质疑

1. **长期衰减措辞比证明更强**。原文上界与 Figure 2 不应扩写成任意内容的 attention 随距离单调下降；二维反例已经能说明边界。
2. **英文基线描述不够清楚**。将 BERT 位置编码称为 sinusoidal 与标准定义不一致，无法只凭正文完全确认更改了哪些组件。
3. **GLUE 结果有显著退化项**。SST-2、QNLI、MNLI 都下降，结论不能只列提升的三项。论文也未统一报告方差或统计检验。
4. **长文本实验存在比较因素混合**。预训练用过 1536，测 1024 没有超训练长度；WoBERT 只列 512，tokenization 在部分模型之间还不同。
5. **部分数值与符号需校对**。Table 5 的 1.50 个百分点对应 RoFormer 自身的截断长度比较，非 WoBERT；Equation 16 的投影转置应由左端展开核对。笔记保留这些边界，不把排版问题复制成结论。

### 6.3 开放问题

- 不同频率底数与维度分配如何影响短程精度、长程区分与外推？
- 哪些任务从旋转归纳偏置中受益，哪些任务更适合学习式位置或其他相对位置方案？
- 多层模型中的上下文边界、causal mask 与显式相对位置如何共同影响平移行为？
- 位置扩展与 KV cache 复用如何协调，才能避免配置变化后读取语义不一致的缓存？〔系统问题，本文未实验〕

### 6.4 后续可读

- [Attention Is All You Need](../2017-attention-is-all-you-need/README.md) —— 回看正余弦编码“可能外推”的原始表述。
- [GQA](../2023-gqa/README.md) —— 共享 K/V 与旋转位置属于两个独立的模型结构维度。
- [Sarathi-Serve](../../llm-serving/2024-sarathi-serve/README.md) —— 将位置一致性带入分块 prefill 的执行过程。
- [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation（ALiBi）](https://arxiv.org/abs/2108.12409) —— 用不同机制研究训练长度与测试长度的关系。
- [Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595) —— 后续的位置插值与适应训练路线，不属于本论文原始结果。
