# 方向 I：反事实主体配对驱动的逐层渐进激活与序列分布对齐 Proposal

> 协议标识：`factpair-progressive-alignment-v0.2-draft`；日期：2026-11-21。
> **状态：研究提案，未实现、未预注册、未运行新实验、未训练；无新的真实模型结果。**
> 本文是 `I/` 目录**当前唯一实施口径**。DeepFade 只作历史归档；上一版“先诊断、训练后置，B1/B2/B3 三选一”的流程不再是主线。
> 性质：**plan-only，仅概念方案，不含可运行代码**；不批准直接开始正式训练，不主张知识已删除。

---

## 0. 结论先行：当前方法是什么

本方案保留 Teacher–Student 框架，但改变 Teacher 提供监督的方式：**不再通过额外注意力温度或注意力干预构造 Teacher，而只通过反事实输入替换构造目标表示轨迹。**

对遗忘输入 $X^+$，仅将遗忘事实主体 $s_f$ 替换为关系匹配的保留集主体 $s_r$，得到 $X^-$。同一个冻结 Teacher 分别前向 $X^+$ 与 $X^-$，在残差相加位置或每层隐状态输出位置提取成对激活。Student 接收原始遗忘输入 $X^+$，通过显式 Student-side 损失，在多个层上逐步向 Teacher 的 $X^-$ 激活轨迹靠拢，而不是一次性替换最终表示。

在此之上，增加长序列逐位置 KL：分别使用 prompt-only、Teacher 输出前缀和 Student 输出前缀构造条件分布，对齐 Teacher 与 Student 的自回归输出轨迹，并用 prompt-only 分布审计二者相对初始条件的偏移。

**一句话科学问题**：反事实主体替换能否为 Teacher 提供可迁移的目标表示与输出轨迹；Student 通过逐层渐进激活对齐和逐位置序列分布对齐，能否在保留通用能力的同时内化选择性遗忘，而不是产生答案替换、普遍拒答或可恢复的输出抑制？

### 相对上一版 Proposal 的五个主线修改

| # | 上一版 | 当前主线 |
|---|---|---|
| 1 | Stage A 只做定位诊断，训练后置 | **Teacher–Student 逐层渐进激活对齐是方法主线**；诊断转为验证组件 |
| 2 | 去 $\tau$ 后在 B1/B2/B3 中择一 | **固定为显式 Student-side activation alignment**；原 ASU forget KL 不再充当遗忘动力 |
| 3 | 配对主要是因果探针 | 配对同时提供**逐层目标激活轨迹与反事实 Teacher 输出轨迹** |
| 4 | 渐进性不属于当前主线 | **层集合 $\mathcal L(k)$ 与层权重 $\rho_\ell(k)$ 的渐进 schedule 是核心机制** |
| 5 | 输出侧只做验收 | 新增**长序列逐位置三条件 KL**，作为输出轨迹对齐项与差分审计 |

---

## 1. 历史证据与本方案的吸收方式

| 已核实事实 | 对当前方案的约束 |
|---|---|
| 旧 v0.3 pilot 的 confirmation 未通过，retain donor 平均效应不小于 forget donor | 不得把旧层集合或旧 pilot 当作新机制已成立的证据；必须重新做负对照和新簇确认 |
| subject-last 在部分 factual-recall 场景参与计算，但不是单一事实的独占地址 | 激活抽点扩展为残差相加或层输出；主抽点须预注册，另一抽点作消融 |
| 去掉注意力温度后，原 ASU 冻结 Teacher 与初始 Student 的 forget 输出相同，原 KL 在初始化处无遗忘梯度 | 当前遗忘梯度显式来自 Student 侧激活对齐；不得继续依赖原 ASU forget KL |
| 主体替换、activation steering、表示对齐均已有近邻工作 | 不声称这些组件各自首创；贡献假设限于“反事实 Teacher 轨迹 × 逐层渐进对齐 × 序列轨迹 KL”的组合及其机制证据 |
| 输出沉默、KL 下降、拒答增多或回答缩短均不足以证明删除 | 保留完整回答、donor 答案替换、内部恢复、relearning 与 retain 多轴验收 |

历史结果只用于定义风险和验证门，不作为当前方法的正向结果。

---

## 2. 研究问题与可证伪预测

- **RQ1 表示轨迹**：关系匹配的 $X^-$ 是否提供比噪声、乱序主体、错误 donor 和 matched-retain 更特异的目标激活轨迹？
- **RQ2 渐进内化**：逐层渐进对齐是否比等预算的静态全层对齐更容易让未 patch 的 Student 内化目标行为，并减少 retain 损伤？
- **RQ3 序列轨迹**：逐位置 Teacher-prefix / Student-prefix KL 是否能量化并缩小长序列生成过程中的分布分叉，而不诱发提前结束或答案替换？
- **RQ4 遗忘深度**：即使输出目标事实下降，独立 probe、patching、量化和受控 relearning 下是否仍难以恢复？

### 正向预测

若机制成立，真实配对应优于随机与错配对照；渐进 schedule 应在等训练 token、步数、可训练参数和算力下优于静态对齐；Student 在部署无 hook 时仍保持效应，且至少两类独立抗恢复证据改善。

### 主要替代解释

1. Student 学会把 $s_f$ 的答案替换成 $s_r$ 的答案；
2. 激活扰动导致低置信、拒答或一般能力损伤；
3. 仅改变输出检索过程，内部表示仍可恢复；
4. 序列 KL 通过提前 EOS 或缩短回答降低，而非真实轨迹对齐。

---

## 3. 数据与反事实配对

### 3.1 实体池

| 池 | 来源与用途 | 硬约束 |
|---|---|---|
| $D_f$ | 官方 `I/data/tofu/forget05.jsonl` 的目标作者与事实 | 提供 $X^+$、遗忘目标与目标事实评测 |
| $D_{donor}$ | stage0 Pool 2 的内部 TOFU 保留集作者 | 只构造 $X^-$ 与 Teacher donor 轨迹；不得进入 retain 训练或终局评测 |
| $D_R^{train}$ | Pool 3 内按作者预留的保留训练子集 | 只用于通用效用损失；与 donor、eval 实体互斥 |
| $D_{eval}$ | Pool 3 剩余作者及官方 real_authors / world_facts / holdout | retain、桥接事实、通用能力与恢复评测 |

TOFU 是合成事实数据，只能声称“数据集内真值”。外部 Wikidata donor 降为可选来源域对照，不进入主线。

### 3.2 配对硬门

1. 两条输入使用同一关系模板，仅实体槽不同，槽外文本逐字一致；
2. 使用目标 checkpoint 的 tokenizer 对完整 chat prompt 重编码，token 序列等长，不 padding、不截断；
3. 槽外 token ID 逐位一致，实体差异为单一连续 span；
4. 目标答案与 donor 事实值可溯源且互不相等；
5. forget、donor、retain-train、eval 按实体与别名簇隔离；
6. 远程重叠扫描、基座零样本泄漏扫描和 checkpoint/tokenizer hash 未完成前，不进入正式训练。

配对不提供人工遗忘标签；donor 事实值只用于来源审计和答案替换分析。

---

## 4. Teacher–Student 与逐层渐进激活对齐

### 4.1 输入与模型角色

$$
X^+=X(s_f,r),\qquad X^-=X(s_r,r)
$$

Teacher 为冻结基座 $\theta_0$，Student 从同一 checkpoint 初始化并记为 $\theta_S$。二者均使用标准 Attention；**Teacher 不使用额外注意力温度 $\tau$，Student 也不依赖该机制。**

Teacher 分别前向 $X^+$ 与 $X^-$。在预注册抽点 $P$ 读取：

$$
A^+_{T,\ell}=h_{T,\ell}(X^+;P),\qquad
A^-_{T,\ell}=h_{T,\ell}(X^-;P)
$$

其中 $P$ 可以是残差相加后的隐状态或 DecoderLayer 输出。首版必须选择一个主抽点，另一个只作位置消融，不得混用向量索引。

配对差值定义为：

$$
\Delta A_\ell=A^+_{T,\ell}-A^-_{T,\ell}
$$

它首先只表示“主体配对引起的表示差异”；是否为目标事实的特异表示，必须由对照实验验证。

### 4.2 Student 侧对齐目标

Student 接收原始遗忘输入 $X^+$，对应激活为 $h_{S,\ell}(X^+;P)$。第 $k$ 个训练阶段的激活损失为：

$$
\mathcal L_{act}(k)=
\frac{1}{\max(1,|\mathcal L(k)|)}
\sum_{\ell\in\mathcal L(k)}
\rho_\ell(k)\,
d\!\left(
h_{S,\ell}(X^+;P),
\operatorname{stopgrad}(A^-_{T,\ell})
\right)
$$

其中：

- $\mathcal L(k)$ 是当前参与对齐的层集合；
- $\rho_\ell(k)$ 是逐层对齐权重；
- $d$ 是预注册并带归一化定义的距离函数；
- Teacher 参数、Teacher 激活和 donor 常量全部 stop-gradient；
- 梯度只通过 Student 隐状态更新 Student。

**主训练不直接替换 Student 激活。** 在一次 Student 前向中，各层分别产生对齐约束；随训练推进，通过扩展 $\mathcal L(k)$ 或调整 $\rho_\ell(k)$ 实现渐进内化。运行时逐层插值或 patch 只作为对照，不作为参数遗忘主结果。

### 4.3 渐进 schedule 的最低定义

必须在训练前冻结：

- 初始层集合、层扩展顺序和每阶段步数；
- $\rho_\ell(k)$ 的数值网格与固定系数；
- 激活归一化、距离函数和停止条件；
- 静态全层、反向 schedule、随机 schedule 等等预算对照。

若渐进 schedule 不优于静态全层对齐，应删除“渐进”贡献表述，只保留多层激活对齐。

---

## 5. 长序列逐位置 KL 对齐

### 5.1 输出轨迹来源与遗忘方向

为使 KL 与反事实遗忘方向一致，本 Proposal 默认：

$$
y^T=\operatorname{Gen}(T,X^-),\qquad
y^S=\operatorname{Gen}(S,X^+)
$$

即 Teacher 输出提供反事实目标轨迹，Student 输出描述原始遗忘输入下的当前轨迹。三类**评分上下文**统一使用原始 prompt $x=X^+$，从而避免把 prompt 本身不同混入词表 KL。

若改用 $y^T=\operatorname{Gen}(T,X^+)$，该 KL 会倾向于保持原始 Teacher 行为，只能作为一致性或保留正则，**不得解释为遗忘损失**。

Teacher/Student 的生成 token 均为离散常量并 stop-gradient；Teacher 分布也 stop-gradient，梯度只经 Student logits 回传。

### 5.2 三类条件与分布

设 $x$ 长度为 $L$，Teacher/Student 输出长度分别为 $T_T,T_S$。对目标位置 $t$：

$$
C^x=x
$$

$$
C^T_t=x\oplus y^T_{<t},\qquad t=1,\ldots,T_T
$$

$$
C^S_t=x\oplus y^S_{<t},\qquad t=1,\ldots,T_S
$$

对应分布为：

$$
q^x=p_T(\cdot\mid x)
$$

$$
q^T_t=p_T(\cdot\mid x,y^T_{<t})
$$

$$
q^S_t=p_S(\cdot\mid x,y^S_{<t})
$$

$q^x$ 是 prompt-only 锚点，只自然对应首个输出 token。将其广播到 $t>1$ 只能用于偏移审计，不能冒充真实的第 $t$ 步自回归条件。

### 5.3 主 KL 与差分审计

主对齐项固定为 Teacher 到 Student 的 forward KL：

$$
K^{TS}_t=D_{KL}(q^T_t\parallel q^S_t)
=\sum_v q^T_t(v)[\log q^T_t(v)-\log q^S_t(v)]
$$

prompt-only 差分审计为：

$$
K^{xT}_t=D_{KL}(q^x\parallel q^T_t),\qquad
K^{xS}_t=D_{KL}(q^x\parallel q^S_t)
$$

$$
\Delta^x_t=K^{xT}_t-K^{xS}_t
$$

$\Delta^x_t$ 只比较偏移标量，不识别分布方向；其相等不推出 $q^T_t=q^S_t$，因此不替代 $K^{TS}_t$，也不默认加入训练。

序列主损失为：

$$
\mathcal L_{seqKL}=
\frac1B\sum_b
\frac{\sum_t m_{b,t}K^{TS}_{b,t}}
{\max(1,\sum_t m_{b,t})}
$$

其中 $m_{b,t}$ 同时受双方有效长度、EOS、目标 span 与 padding mask 约束。

### 5.4 首 token 的特殊性

当 $t=1$ 时，$y^T_{<1}=y^S_{<1}=\varnothing$，故：

$$
C^T_1=C^S_1=x,\qquad q^T_1=q^x
$$

此时 Teacher 的反事实输出尚未通过前缀进入条件分布，首 token KL 只提供原始 prompt 下的 Teacher 锚定，不提供反事实首 token 监督。**遗忘方向的 seqKL 默认从 $t=2$ 起计算，$t=1$ 单独记录。** 如未来增加首 token 目标 token CE，必须作为独立损失报告，不能与当前 KL 混称。

### 5.5 长度、EOS、padding 与 causal shift

1. 主 KL 只在双方均未越过 EOS 的共同有效位置计算；较长一方尾部不参与 KL，单独记录长度差、尾部事实出现率和提前结束率；
2. EOS 可作为最后一个有效目标，EOS 后的所有位置屏蔽；不得用 padding 分布伪造尾部对齐；
3. padding 的 attention mask 为 0、监督 label 为 -100、KL mask 为 0，归一化分母只含有效位置；
4. 对目标 $y_t$，读取拼接序列位置 $L+t-1$ 的 logits，即 $x$ 末尾预测 $y_1$、$y_{t-1}$ 位置预测 $y_t$；
5. 左 padding、变长 prompt 和 chat template 必须按每条样本真实 prompt 长度定位，不能使用固定列；
6. Teacher 与 Student 必须共享 tokenizer、词表顺序和模板；生成时固定 decoding、seed 与最大长度，Student 序列按预注册频率刷新。

完整 `[B,T,V]` 分布不长期保存，前向后立即计算逐位置 KL；只持久化标量轨迹、mask、长度和少量调试分布。

### 5.6 同前缀审计

为区分模型差异与前缀分叉，可在少量审计样本增加：

$$
q^{S\mid T}_t=p_S(\cdot\mid x,y^T_{<t})
$$

分别记录 $D_{KL}(q^T_t\parallel q^{S\mid T}_t)$ 与 $D_{KL}(q^{S\mid T}_t\parallel q^S_t)$。二者是诊断量，不构成对 $K^{TS}_t$ 的精确加法分解，也不进入三分布主损失。

---

## 6. 总损失、梯度与训练边界

$$
\mathcal L(k)=
\lambda_{act}\mathcal L_{act}(k)
+\lambda_{seq}\mathcal L_{seqKL}
+\beta\mathcal L_R
$$

$\mathcal L_R$ 在独立 retain-train 上计算，可采用标准答案位 NLL。所有系数、warmup、训练步数和 retain 停止线在运行前冻结，不启用未预注册的 EMA、clip/exp 或单向棘轮。

统一反向传播只更新 Student。部署主结果必须满足无 Teacher、无 donor、无 hook；运行时仍依赖 patch 的版本只能算激活干预对照。

---

## 7. 中间变量与接口口径

| 名称 | 形状或类型 | 含义 |
|---|---|---|
| `paired_prompt_ids` | 两组 `[B,L]` | $X^+,X^-$，槽外 token 一致 |
| `prompt_attention_mask` | `[B,L]` | prompt 有效位 |
| `teacher_act_plus/minus` | `[B,Layers,d]` 或抽点子集 | $A^+_{T,\ell},A^-_{T,\ell}$ |
| `student_act` | `[B,Layers,d]` | Student 在 $X^+$ 上的逐层激活 |
| `active_layers`、`layer_weights` | schedule 变量 | $\mathcal L(k),\rho_\ell(k)$ |
| `teacher_output_ids`、`student_output_ids` | `[B,TT]`、`[B,TS]` | 反事实 Teacher 与当前 Student 输出轨迹 |
| `q_x` | `[B,V]` | prompt-only 锚点 |
| `q_T`、`q_S` | `[B,TT,V]`、`[B,TS,V]` | 两类自回归前缀条件分布 |
| `kl_ts`、`kl_xT`、`kl_xS` | 逐位置标量 | 主对齐与差分审计 |
| `align_mask`、`valid_lengths` | mask / 长度 | 有效 KL 位置与 EOS 边界 |
| `retain_batch` | 独立批次 | $\mathcal L_R$，不得含 donor/eval 实体 |

实现接口至少接收：冻结 Teacher、可训练 Student、配对 prompt 与 mask、抽点定义、层 schedule、Teacher/Student 输出与 mask、tokenizer/EOS/pad 配置、KL 方向、目标 span mask、固定损失系数和 retain batch。

---

## 8. 验证矩阵与 kill-switch

| 阶段门 | 必须验证 | 失败处置 |
|---|---|---|
| G0 数据与实现 | 配对硬门、实体隔离、hook/no-op、层索引、causal shift、首 token、EOS/mask | 任一失败则 BLOCKED，不训练 |
| G1 梯度合法性 | Teacher stop-gradient；Student-side $L_{act}$ 与 $L_{seqKL}$ 有预期梯度；仅更新 Student | 梯度路径错误则停止 |
| G2 配对特异性 | 真实 donor 优于范数匹配 noise、shuffle/wrong-donor、matched-retain、随机层/位置 | 不优于简单对照则判死该 target design |
| G3 渐进性 | 等预算优于静态全层、反向和随机 schedule，retain 合格 | 无独立增益则删除“渐进”主张 |
| G4 序列行为 | 目标事实、donor 答案、其他错答、拒答、长度、EOS、完整回答分别报告 | donor 替换或提前结束解释成立时不得称遗忘 |
| G5 参数内化 | 未 patch Student 复现效应，retain/shared facts 与通用能力合格 | 仅在 hook 下有效则不是参数遗忘主结果 |
| G6 深度验收 | 输出访问变体、独立 probe/patching、Q4、受控 relearning | 仅输出沉默或可恢复则判浅层抑制 |

统计单位是作者/实体簇；多 prompt、多 donor、多 seed 是簇内重复测量。discovery、calibration、confirmation、final 隔离，旧 pilot 不复用为 confirmation。至少两类独立抗恢复证据改善且 retain 合格，才允许使用受限的“深度遗忘”表述。

---

## 9. 当前状态与待冻结项

**当前状态**：机制提案已重写；实测 `NOT_STARTED`；没有新实验、没有训练、没有代码实现；不继承旧 pilot 的正向结论。

**待冻结**：checkpoint/tokenizer/chat template hash、主激活抽点、距离与归一化、层 schedule、序列生成与刷新策略、首 token 是否另设独立监督、loss 系数、目标 span、decoding、训练预算、retain 停止线、统计阈值与确认簇。

---

## 10. 30 秒讲稿

> 我们保留 Teacher–Student，但 Teacher 不再靠注意力温度制造监督，而是只改变输入：把遗忘主体替换成关系匹配的保留主体，得到一条反事实表示轨迹。Student 在原始遗忘输入上逐层、渐进地向这条轨迹对齐，而不是一次性替换激活；输出侧再用 Teacher 反事实序列前缀和 Student 自有序列前缀，逐位置计算长序列 KL。所有目标激活和离散前缀都 stop-gradient，只更新 Student。我们通过噪声、乱序 donor、静态 schedule、答案替换、提前 EOS 和恢复攻击来检验：真正的目标是让 Student 内化选择性遗忘，而不是把答案换成 donor、变拒答或暂时答不出来。

## References

- 本地证据：[反事实主体替换 SubjectLast 对齐去 τ 证据调研](方向I_反事实主体替换SubjectLast对齐去τ_证据调研.md)
- 数据：[stage0 作者分区摘要](data/stage0/summary.md)、[external donor v0.3 候选](data/external_donor_prompt_activation_v0.3/README.md)
- 历史参考：[DeepFade 归档说明](archive/2026-11-21_deepfade/README.md)、[固定层路线归档](archive/2026-11-13_fixed_layer/方向I_当前主线与实施计划.md)
- 外部已核验来源：ASU, ICLR 2026；CSD, Findings of EMNLP 2026；ROME, NeurIPS 2022；Hase et al., NeurIPS 2023；UDS, EMNLP 2026；TOFU, arXiv 2024。
