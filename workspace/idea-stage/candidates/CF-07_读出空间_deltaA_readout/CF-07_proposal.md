---
id: CF-07
rank: 8
title: 读出空间重定义 ΔA_readout（旋转不变）
avg_score: 7.2
verdict: Accept with Revisions (PROVISIONAL, mechanism-based)
decision_source: IDEA_DISCOVERY_20261126.md §5
domain: LLM/LRM Unlearning / 激活差表示域
evidence_level: provisional（机制推断，无实测，无跨模型回执）
baseline_relation: 把 ΔA 的载体从「隐状态向量差」改为「冻结下游读出的标量差 + 局部敏感度方向」
---

# CF-07 · 读出空间 $\Delta A_{\text{readout}}$（旋转不变）

## 0. 故事（30 秒讲稿）

> 基线已经要求「读出经冻结下游重算，不得把中间层向量直接过 LM head」（§2.3 / §4.1），但 $\Delta A$ 本身仍定义在隐状态空间 $A^+-A^-$。UDS / Logit Lens 的教训是：隐状态差会被下游非线性旋转掩盖，观测型投影会误判。我们把 $\Delta A$ 重新定义在**冻结下游读出的泛函**上——不是「向量差了多少」，而是「目标事实读出 $r_f$ 差了多少，以及沿哪个局部方向最敏感」。这样 $\Delta A_{\text{readout}}$ 对隐空间正交旋转天然不变，可解释性落在「读出的差」这个人类能核验的量上。

## 1. 领域问题与空白

- UDS(EMNLP 2026) 明确：CKA/Fisher 反映优化轨迹而非知识存留，Logit Lens/线性探针属**观测型**，一旦遗忘算法旋转或扭曲内部隐空间就误判「知识已消除」；应从被动观测转向因果干预。
- 基线 §4.1 已把读出改成 patch-forward 重算，但 §2.2 的 $\Delta A_{\ell,p}=A^+-A^-$ 仍是隐空间向量差；§2.2.1 把它列为未定稿。
- **近邻空白**：CLReg(2601.22028) 在池化归一化表示上算余弦，SCALPEL 在 SAE 特征空间算贡献差，MUtE 在连续表示上做擦除；**没有人把 $\Delta A$ 的定义域从隐状态向量显式改成「冻结下游读出的标量差 + Jacobian 一阶方向」，并要求正交旋转不变性作为判别实验**。

## 2. 核心假设（可证伪）

**H**：对隐状态施加正交变换 $Q$ 后，隐空间 $\Delta A$ 漂移而 $\Delta A_{\text{readout}}$ 不变；且读出空间 $\Delta A_{\text{readout}}$ 对 §4.1 中性约束的可满足性显著优于隐空间 $\Delta A$。若二者高度共线（无新增信息），或 Jacobian 估计噪声大于信号，则本思路被证伪。

## 3. 具体实现（LaTeX）

**读出泛函**（冻结下游 $\mathcal R$，经 patch-forward，不直接过 LM head）：

$$
r_f(h)=\mathcal R_{y^+}(h),\qquad
\Delta A^{\text{scalar}}_{\ell,p}=r_f\!\left(A^+_{T,\ell}(p)\right)-r_f\!\left(A^-_{T,\ell}(p)\right).
$$

**局部敏感度方向**（Jacobian 一阶，构成方向部分）：

$$
J_{\ell,p}=\frac{\partial\, r_f(h)}{\partial h}\Big|_{h=A^+_{T,\ell}(p)},\qquad
\widehat{J}_{\ell,p}=J_{\ell,p}/\lVert J_{\ell,p}\rVert ,
$$

$$
\Delta A_{\text{readout},\ell,p}\;=\;\big(\ \Delta A^{\text{scalar}}_{\ell,p},\ \widehat{J}_{\ell,p}\ \big)\quad\text{（标量差 + 局部方向）}.
$$

**旋转不变性判别**（硬检验）：对任意正交 $Q\ (Q^\top Q=I)$，

$$
r_f(Q h)=r_f(h)\ \Rightarrow\ \Delta A^{\text{scalar}}_{\ell,p}(Q)=\Delta A^{\text{scalar}}_{\ell,p},\qquad
\big(Q A^+ - Q A^-\big)\ \text{一般}\ \ne\ \big(A^+ - A^-\big).
$$

**目标与内化**：用读出标量差作门、局部方向作投影方向：

$$
T_{\ell,p}=\operatorname{stopgrad}\!\left(\Pi_{\mathcal K}\Big[\,h^{+}_{\ell}(p)-\alpha\,\Delta A^{\text{scalar}}_{\ell,p}\,\widehat{J}_{\ell,p}\,\Big]\right),
$$

$$
\mathcal K=\{h:\ r_{f,p}(h)\le c,\ s_{d_j,p}(h)\le d_j,\ \lVert\Delta r_{E,p}(h)\rVert\le e\},\qquad
\boxed{\ \mathcal L=\lambda_F\mathcal L_F+\beta\mathcal L_R\ } .
$$

Jacobian 估计的样本数、数值稳定方案在 discovery 预注册；全部读出经冻结下游，不直接过 LM head。

## 4. 差异化（novelty 落点）

| 近邻 | 本方案差异 |
|---|---|
| UDS (EMNLP 2026) | 后者用 patch 做**深度评分/KE 定位**；本方案把「读出」作为 $\Delta A$ 的**定义域**，并加旋转不变性判别 |
| Logit Lens / 线性探针 | 观测型、旋转下误判；本方案要求 $\Delta A_{\text{readout}}$ 在正交变换下不变（可证伪） |
| CLReg (2601.22028) / SCALPEL (2609.31056) | 后者在池化表示 / SAE 特征空间；本方案在冻结下游读出的泛函 + Jacobian |
| 基线 v0.6.1 §2.2 | **唯一改动点**：$\Delta A$ 载体从隐状态向量差改为读出标量差 + 局部敏感度方向 |

**novelty 落点**：把「激活差」从几何向量重新定义为**功能读出差**，用旋转不变性把「可解释」变成可证伪的代数性质，而非叙事。

## 5. /idea-evaluator 评估（PROVISIONAL）

| 维度 | 分 | 证据（机制推断，未实测） | Lift |
|---|---|---|---|
| Higher | 7 | 读出空间对齐真实删除目标，但增益依赖「旋转确会糊弄隐空间差」（`mechanism-based`） | 与 CF-03 因果门控叠加 |
| Faster | 7 | 多一次 Jacobian 估计，纯前向可控 | 复用缓存读出 |
| Stronger | 8 | 正交旋转不变 + 中性约束，抗表征旋转/OOD 误判（`mechanism-based`） | 加量化/旋转攻击对照 |
| Cheaper | 7 | 无 SAE、无标注、无外部模型 | 复用 patch-forward 资产 |
| Broader | 7 | 表示域重定义可迁移其它激活差方法（`mechanism-based`） | 统一读出接口 |

**平均分 $\bar S=36/5=7.2>7$。**

- **定位**：Novel Method；一句话故事见 §0。
- **致命缺陷审计**：**2 MAJOR** → (1) F6 旋转不变性 + 「读出空间 vs 隐空间是否共线」判别须预注册；(2) F1 与 UDS / Logit Lens / CLReg 划界须核验。零 CRITICAL。故 `Accept with Revisions`（2 MAJOR）。
- **生命周期**：Frontier exploration 3–9 个月；Jacobian 数值稳定是 Yellow 项，收窄样本后绿。
- **范式探测**：First Principles=Yes、Elephant=Partial、Technology Cycle=No、Hamming=Partial → possible。
- **可行性**：Compute 中（Jacobian）、Data 低、Engineering 中、Timeline 中。
- **裁决**：**Accept with Revisions**，`worth pursuing, pending the validation experiment`。**Revisions**：先定 Jacobian 数值方案与 UDS 划界，再开跑。
- **决定性验证实验**：对隐状态施加正交变换，核验 $\Delta A_{\text{readout}}$ 不变而隐空间 $\Delta A$ 漂移；隐空间 vs 读出空间对 §4.1 中性约束可满足性差异；`needs manual pilot`。
- 分数为执行者机制推断分；`novelty-check = PROVISIONAL`，非跨模型回执。

## 6. 判别实验、失败模式、必测指标

- **判别**：旋转不变性成立；读出空间 $\Delta A$ 与隐空间 $\Delta A$ 不共线；中性约束可满足性差异显著。
- **失败模式**：二者高度共线（无新增信息）；Jacobian 估计噪声大于信号；读出维度塌缩为单一标量、丢失层/位置结构；patch 引入 OOD 伪峰。
- **必测指标**：旋转前后 $\Delta A^{\text{scalar}}$ 方差、共线度、$r_f$、retain NLL、donor 答案率、3 seed std。

## 7. 与方向 I 基线 v0.6.1 的关系

只重定义 §2.2 的 $\Delta A$ 载体；$S_f$ 作用域、Teacher stopgrad、§4.1 中性约束（$\Pi_\mathcal K$）、「读出须经冻结下游重算、不直接过 LM head」、$\lambda_{KL}^{\text{ablation}}=0$、§6 六条验收全部沿用。本方案完全不涉及 Student 激活抑制，是 $\Delta A$ 表示域的重定义，与 V1 不同族。

## 8. 诚信声明与来源

plan-level，不新增实验；claim 待验证；分数机制推断、PROVISIONAL、`REVIEW_UNAVAILABLE`；新结果须过 `/experiment-audit` + `/result-to-claim`。

- 基线 §2.2/§2.3/§4.1；`IDEA_CANDIDATES.md` D9
- 近邻（身份未 D1 核验，不作 BibTeX）：UDS EMNLP 2026、Logit Lens、CLReg arXiv:2601.22028、SCALPEL arXiv:2609.31056、Zhang & Nanda ICLR 2024
