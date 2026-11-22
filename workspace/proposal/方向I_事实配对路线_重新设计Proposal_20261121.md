# 方向 I：反事实主体配对驱动的逐层渐进遗忘 Proposal

> 协议：`factpair-progressive-alignment-v0.5-reference-draft`；日期：2026-11-22。
> **状态：plan-only。尚未实现、预注册或训练，没有新的实验结果；激活差异处理是核心待调研问题，当前仅保留参考方案。**

## 0. 一句话故事

**当前参考方案中，Teacher 不再通过注意力温度制造遗忘目标，而是把遗忘主体替换成保留主体，得到一条反事实表示与输出轨迹；Student 在原始遗忘输入上逐层、逐位置地向这条轨迹对齐，最终将变化内化到参数中。激活差异的具体处理仍是核心待调研问题。**

我们要验证的不是“输出是否变了”，而是：这种变化是否真正移除了目标事实，同时没有变成 donor 答案替换、普遍拒答、回答缩短或仍可恢复的浅层抑制。

---

## 1. 为什么这样改

原 ASU 用带注意力温度的 Teacher 产生 forget 分布，再用 KL 训练 Student。去掉温度后，若 Teacher 与初始 Student 相同，原 forget KL 在初始化处没有遗忘梯度。

因此，本方案做两项改动：

1. **输入构造目标**：仅把遗忘主体 $s_f$ 换成关系匹配的保留主体 $s_r$，让冻结 Teacher 提供反事实目标轨迹；
2. **显式对齐 Student**：用逐层激活损失提供遗忘梯度，再用 DKT 式逐位置 KL 对齐长序列输出。

历史 pilot 的 confirmation 未通过，因此旧层集合和旧结果不能作为本方案的正向证据。

---

## 2. 方法主线

### 2.1 构造反事实输入

对同一关系 $r$：

$$
X^+=X(s_f,r),\qquad X^-=X(s_r,r)
$$

两条输入除主体槽外逐字一致，token 等长，槽外 token ID 不变。$X^+$ 来自 forget 集，$X^-$ 的 donor 来自 TOFU 内部保留集。

forget、donor、retain-train 和 evaluation 按作者/别名簇隔离。TOFU 只提供“数据集内真值”，不声称真实世界事实。

### 2.2 核心参考设计：Teacher 提供逐层目标激活

> **核心但待调研：** 如何利用 $A^+$、$A^-$ 及其差异构造遗忘目标，目前尚未定稿。本节保留“直接向 $A^-$ 对齐”作为参考方案，不代表已选定最终机制。

Teacher 是冻结基座 $\theta_0$，Student 参数记为 $\theta$，并从同一 checkpoint 初始化。二者均使用标准 Attention，不引入额外注意力温度。

Teacher 分别前向 $X^+$ 和 $X^-$。在抽点 $P$ 读取第 $\ell$ 层激活：

$$
A^+_{T,\ell}=h_{T,\ell}(X^+;P),\qquad
A^-_{T,\ell}=h_{T,\ell}(X^-;P)
$$

$P$ 首选每层 DecoderLayer 输出；残差相加位置作为位置消融。配对差值

$$
\Delta A_\ell=A^+_{T,\ell}-A^-_{T,\ell}
$$

只表示“主体替换引起的表示差异”。后续需要调研应直接对齐 $A^-$、利用 $\Delta A_\ell$ 引导更新，还是采用其他目标构造；当前不预设 $\Delta A_\ell$ 已是纯净的事实差异。

### 2.3 Student 逐层渐进对齐

作为当前参考实现，Student 接收原始遗忘输入 $X^+$。令第 $k$ 个训练阶段参与对齐的层集合为 $\Lambda(k)$，层权重为 $\rho_\ell(k)$，激活遗忘损失为：

$$
\mathcal L_F(k)=
\frac{1}{|\Lambda(k)|}
\sum_{\ell\in\Lambda(k)}
\rho_\ell(k)\,
\operatorname{dist}\!\left(
h_{S,\ell}(X^+;P),
\operatorname{stopgrad}(A^-_{T,\ell})
\right)
$$

训练时逐步扩展 $\Lambda(k)$ 或调整 $\rho_\ell(k)$，而不是一次性替换最终激活。主训练直接优化 Student 参数；运行时 patch 只作对照。

“渐进”必须相对等预算的静态全层、反向和随机 schedule 证明独立收益，否则只保留“多层对齐”表述。

---

## 3. 长序列 KL：采用 DKT 的同前缀定义

### 3.1 反事实 Teacher 策略

令外部输入 $x=X^+$，输入变换 $\phi(x)=X^-$。定义：

$$
p_T(y\mid x):=p_{\theta_0}(y\mid\phi(x))=p_{\theta_0}(y\mid X^-)
$$

$$
q_\theta(y\mid x):=p_\theta(y\mid X^+)
$$

这里 $p_T$ 是反事实目标策略，$q_\theta$ 是当前 Student 策略。第 $t$ 步生成状态为：

$$
s_t=(x,y_{<t})
$$

对任一固定输出前缀 $y_{<t}$，实际前向是：

$$
p_T(a\mid s_t)=p_{\theta_0}(a\mid X^-,y_{<t}),\qquad
q_\theta(a\mid s_t)=p_\theta(a\mid X^+,y_{<t})
$$

其中 $a$ 是下一 token。**每个 KL 内的两个分布必须共享同一个 $y_{<t}$。** 不能直接比较 Teacher 在 $y^T_{<t}$ 下的分布与 Student 在另一前缀 $y^S_{<t}$ 下的分布。

### 3.2 前缀来源与 KL 方向

分别采样：

$$
y^T\sim p_T(\cdot\mid x),\qquad y^S\sim q_\theta(\cdot\mid x)
$$

由它们诱导的第 $t$ 步前缀分布记为 $d_T^t$ 和 $d_\theta^t$。DKT 的序列 forward KL 分解为：

$$
\operatorname{KL}(p_T\parallel q_\theta)(x)
=
\sum_{t=1}^{L_y}
\mathbb E_{s_t\sim d_T^t}
\left[
\operatorname{KL}\!\left(p_T(\cdot\mid s_t)\parallel q_\theta(\cdot\mid s_t)\right)
\right]
$$

其中 $L_y=|y|$。DKT 的核心是：**前缀来源决定在哪里计算损失，KL 方向决定如何比较同一状态下的两个分布。** reverse KL 使用 Student 诱导的前缀并反转散度方向，但不进入首版主损失。

### 3.3 首版序列损失

首版采用两种前缀来源上的 forward KL：

- Teacher-prefix：覆盖反事实目标轨迹，属于 off-policy soft-label SFT；
- Student-prefix：覆盖 Student 实际访问的状态，属于 DAgger-style on-policy SFT。

令 $z\in\{T,S\}$ 表示前缀来源，$s^z_{b,t}=(x_b,y^z_{b,<t})$，$L_{z,b}$ 为对应输出的有效长度，$m^z_{b,t}$ 为有效位置 mask。统一定义：

$$
\mathcal L_z
=
\frac1B\sum_b
\frac{
\sum_{t=1}^{L_{z,b}}m^z_{b,t}\,
\operatorname{KL}\!\left(
p_T(\cdot\mid s^z_{b,t})
\parallel
q_\theta(\cdot\mid s^z_{b,t})
\right)
}{
\max\!\left(1,\sum_t m^z_{b,t}\right)
}
$$

然后混合两条分支：

$$
\boxed{
\mathcal L_{KL}
=
(1-\eta)\mathcal L_T
+\eta\mathcal L_S
}
$$

其中 $\eta\in[0,1]$ 在训练前冻结。两条分支分别计算和归一化，**不要求 $y^T$ 与 $y^S$ 等长，也不逐 token 比较两条不同序列。**

### 3.4 梯度

固定状态 $s_t$ 并冻结 Teacher 时：

$$
\nabla_\theta
\operatorname{KL}\!\left(p_T(\cdot\mid s_t)\parallel q_\theta(\cdot\mid s_t)\right)
=
-\mathbb E_{a\sim p_T(\cdot\mid s_t)}
\left[\nabla_\theta\log q_\theta(a\mid s_t)\right]
$$

因此主分支等价于 Teacher soft target 的交叉熵。Teacher 分布、Teacher 激活和离散输出 token 均 stop-gradient；梯度只经 Student 隐状态和 logits 更新 Student。

---

## 4. 总损失

$$
\boxed{
\mathcal L=
\lambda_F\mathcal L_F
+\lambda_{KL}\mathcal L_{KL}
+\beta\mathcal L_R
}
$$

$\mathcal L_R$ 是独立 retain-train 上的答案位 NLL，用于保留非目标知识与通用能力。所有系数、schedule、生成配置和训练预算在实验前冻结。

部署主结果只保留 Student，不依赖 Teacher、donor 或运行时 hook。

---

## 5. 边界情况与实现检查

| 项目 | 明确口径 |
|---|---|
| 首 token | $s_1=(x,\varnothing)$；Teacher 使用 $X^-$、Student 使用 $X^+$，所以从 $t=1$ 起就有反事实监督 |
| 长度不同 | 两条前缀分支独立处理，不做共同长度截断；另行报告长度差和提前结束率 |
| EOS | EOS 是最后一个有效目标；EOS 后不计算损失 |
| Padding / mask | padding 的 attention mask、label 和 KL mask 均无效；分母只包含有效位置 |
| Causal shift | 若 prompt 长度为 $L_x$，预测 $y_t$ 应读取位置 $L_x+t-1$ 的 logits，即 $x$ 末尾预测 $y_1$、$y_{t-1}$ 位置预测 $y_t$ |
| Tokenizer | Teacher 与 Student 必须共享 tokenizer、词表顺序和 chat template |
| 显存 | 不长期保存完整 `[B,T,V]` 分布；前向后立即计算 KL，只保留标量轨迹和少量调试分布 |

---

## 6. 怎样判断方法真的成立

1. **配对特异性**：真实 donor 必须优于噪声、乱序 donor、wrong-donor、matched-retain 和随机层/位置对照；
2. **渐进收益**：渐进 schedule 必须在等预算下优于静态全层对齐；
3. **不是答案替换**：同时报告原目标事实、donor 答案、其他错答、拒答、长度和 EOS，不能把改答 donor 当作遗忘；
4. **参数内化**：无 hook 的 Student 仍需保持效应，且 retain/shared facts 与通用能力合格；
5. **不是浅层抑制**：结合输出访问变体、独立 probe/patching、量化和受控 relearning。至少两类独立抗恢复证据改善，才允许使用受限的“深度遗忘”表述。

统计单位是作者/实体簇；多 prompt、多 donor 和多 seed 只作为簇内重复测量。旧 pilot 不复用为 confirmation。

---

## 7. 30 秒讲稿

> 我们把 ASU 的“注意力温度 Teacher”改成“反事实输入 Teacher”：只把遗忘主体替换成关系匹配的保留主体，Teacher 由此提供另一条逐层表示和输出轨迹。参考方案中，Student 通过逐层激活损失 $\mathcal L_F$ 渐进靠近目标表示，再按 DKT 的方式，用 $\mathcal L_{KL}$ 在 Teacher 前缀和 Student 前缀上分别做同前缀 forward KL。激活差异究竟应直接对齐、差值引导还是采用其他构造，仍是当前核心待调研问题。训练只更新 Student，部署不需要 hook。

## 参考

- DKT：Zhao et al., [Decoupling KL and Trajectories](papers/【DKT】Decoupling%20KL%20and%20Trajectories-A%20Unified%20Perspective%20for%20SFT,%20DAgger,%20Offline%20RL,%20and%20OPD%20in%20LLM%20Distillation.pdf), 2026，采用其状态、前缀分布、序列 KL 分解和梯度定义。
- [方向 I 历史证据调研](archive/方向I_反事实主体替换SubjectLast对齐去τ_证据调研.md)。
- [stage0 数据分区](data/stage0/summary.md)。
