---
id: CF-02
rank: 3
title: 三前向角色分解 —— 删什么 / 别变成什么 / 保什么
avg_score: 7.6
verdict: Strong Accept (PROVISIONAL, mechanism-based)
decision_source: IDEA_DISCOVERY_20261126.md §5
domain: LLM/LRM Unlearning / 反事实激活差的角色分解
evidence_level: provisional（机制推断，无实测，无跨模型回执）
baseline_relation: 把基线两路前向扩为三路；donor 由「对照」升级为「负约束来源」，仍不进目标语义
---

# CF-02 · 三前向角色分解（三方约束 $\Delta^*$）

## 0. 故事（30 秒讲稿）

> 用户只提了 forget 和 donor 两次前向，但「保住通用效用」这一半根本没被观测到。我们明确三路各自的角色：forget 前向回答「删什么」（把 $r_f$ 压进中性死区），donor 前向回答「别变成什么」（作为**负约束**防止改答 donor、防止答案替换），retain 前向回答「保什么」（给效用一块地板）。$\Delta A$ 不再是单一差向量，而是被投影到这三个角色张成的约束结构上，Student 只内化「删什么」那一路的解。可解释性在于：三条约束各自回答一个审计问题，paper 里能一句话讲清「我们为什么没把遗忘做成换答案或普遍拒答」。

## 1. 领域问题与空白

- 基线 v0.6.1 §1 已把 donor-KL 降为消融臂，因为把 $p_{\theta_0}(y|X^-)$ 当 soft target 是 donor 答案替换（forward KL mode-covering，不奖励原事实读不出）。但 §2.2.1 仍把 $A^-$ / $\Delta A$ 当作「待证伪对照」，没把 donor 的**负向**信息用起来。
- 「观察两次前向读不出效用」：效用是另一分布的性质，必须第三次前向（retain）才能约束。E0 历史失败（retain NLL 超上限约 81.6 倍）即缺此约束的后果。
- **近邻空白**：CLReg(arXiv:2601.22028) 用 forget 双视图正对 + retain 负对做**表示塑形**，但显式声明「表示分离本身不等于知识删除」；D2F(arXiv:2610.17654) 用 semantic mediator 分离敏感/非敏感但作用在文本答案层；**没有人把 donor 严格当「禁止方向」、retain 当「效用地板」、forget 当「删除读出」，三者在中间层激活差上联立成约束反演**。

## 2. 核心假设（可证伪）

**H**：把 $\Delta A$ 分解为「删除 / 反替换 / 保效用」三路角色后，三方约束存在非空帕累托解；且 donor 负约束开/关会显著改变 donor 答案率，retain floor 开/关会显著改变 $U$ 与 $r_f$ 的帕累托前沿。若「$r_f$ 降必然 $U$ 崩」或 donor 负约束无法降低答案替换率，则三方不可兼得，本思路被证伪。

## 3. 具体实现（LaTeX）

**三路前向**（$\theta_0$ 冻结）：

$$
h^{+}_{\ell}(p)=h_{T,\ell}(X^+\oplus y^+;p),\quad
h^{-}_{\ell}(p)=h_{T,\ell}(X^-\oplus y^+;p),\quad
h^{r}_{\ell}(p)=h_{T,\ell}(X_r\oplus y_r;p),\qquad p\in S_f .
$$

**角色分解**：把原始配对差投影到三角色方向（身份/事实/效用由探针或梯度基给出）：

$$
\Delta A_{\ell,p}=\underbrace{\Pi_{\text{fact}}\Delta A_{\ell,p}}_{\text{删什么}}
+\underbrace{\Pi_{\text{id}}\Delta A_{\ell,p}}_{\text{别变成什么（禁止方向）}}
+\underbrace{\Pi_{\text{util}}\Delta A_{\ell,p}}_{\text{保什么（地板基）}}+\epsilon_{\ell,p}.
$$

**三方约束反演**（donor 只进不等式约束、不进目标）：

$$
\Delta^{*}_{\ell,p}=\arg\min_{\Delta}\ r_f\!\left(h^{+}_{\ell}(p)+\Delta\right)
$$

$$
\text{s.t.}\quad
r_f(h^{+}+\Delta)\le c,\quad
\underbrace{\langle \Delta,\ \widehat{\Delta A}_{\text{donor}}\rangle\le -\kappa_d}_{\text{donor 负约束：禁止向 donor 语义走}},\quad
\underbrace{U(h^{+}+\Delta)\ge U(h^{+})-\delta}_{U\ \text{在 retain 前向标定}},\quad
s_{d_j,p}\le d_j,\ \lVert\Delta r_{E,p}\rVert\le e,\ \lVert\Delta\rVert\le\varepsilon .
$$

其中 $\widehat{\Delta A}_{\text{donor}}=h^-_{\ell}(p)-h^r_{\ell}(p)$ 仅作禁止方向，$U$ 由第三次 retain 前向的冻结读出标定；**$y^-$ / `donor_fact_value` 不拼入任何前向或 loss**。

**内化**（与基线主损失同形）：

$$
T_{\ell,p}=\operatorname{stopgrad}(h^{+}_{\ell}(p)+\Delta^{*}_{\ell,p}),\qquad
\mathcal L_F(k)=\frac{1}{|\Lambda(k)||S_f|}\sum_{\ell\in\Lambda(k)}\sum_{p\in S_f}\rho_\ell(k)\,
\operatorname{dist}\big(h_{S,\ell}(X^+\oplus y^+;p),\,T_{\ell,p}\big),
$$

$$
\boxed{\ \mathcal L=\lambda_F\mathcal L_F+\beta\mathcal L_R\ },\qquad \lambda_{KL}^{\text{(ablation)}}=0 .
$$

## 4. 差异化（novelty 落点）

| 近邻 | 本方案差异 |
|---|---|
| CLReg (NeurIPS2026, 2601.22028) | 后者拉近双视图/拉远 retain，是「表示塑形正则」；本方案把三路写成**约束反演的硬约束**，donor 是禁止方向而非负样本 |
| D2F (2610.17654) | 后者在文本答案层抽 semantic mediator + DPO；本方案在中间层激活差，且 donly 作负约束不生成反事实答案 |
| v0.5 donor-KL（基线判死） | $\mathcal L_{KL}$ 把 donor 分布当目标（共谋）；本方案 donor 只进不等式约束、$\lambda_{KL}=0$，不构成答案替换 |
| 基线 v0.6.1 §2.3 $T_{\ell,p}$ | **唯一改动点**：反演从「中性可行集最近点」变「三方（+retain floor + donor 负约束）约束问题」 |

**novelty 落点**：把 donor 由「对照/被降级的消融臂」重新定义为**可解释的负约束来源**，并补第三次 retain 前向把「保效用」写成硬约束——直接回答用户「既遗忘又保通用效用」的后半句。

## 5. /idea-evaluator 评估（PROVISIONAL）

| 维度 | 分 | 证据（机制推断，未实测） | Lift |
|---|---|---|---|
| Higher | 8 | 三方约束直接堵住答案替换与效用崩两个捷径（`mechanism-based`） | 强调 $r_f$/$U$ 帕累托 |
| Faster | 7 | 比两路多一次冻结前向，成本可控，无新模块 | 复用 $S_f$ 缓存 |
| Stronger | 8 | donor 负约束 + 效用地板 + 反 EOS 约束，抗输出抑制捷径（`mechanism-based`） | 加 P3/量化恢复 |
| Cheaper | 7 | 无 SAE、无标注、无外部 oracle | 复用 stage0 资产 |
| Broader | 8 | 三角色框架可迁移 MUSE/WMDP/LRM 轨迹（`mechanism-based`） | 与 CF-06 联动 |

**平均分 $\bar S=38/5=7.6>7$。**

- **定位**：Novel Method；一句话故事见 §0。
- **致命缺陷审计**：1 MAJOR（F6 三方约束可行性判别实验必须预注册）+ 2 MINOR（F1 近邻划界、F10 失败模式已命名）。零 CRITICAL。
- **生命周期**：Frontier exploration 3–9 个月，能力匹配绿。
- **范式探测**：First Principles=Yes、Elephant=Yes（答案替换是公认痛点）、Technology Cycle=Partial、Hamming=Partial → disruptive seeds 存在。
- **可行性**：Compute 中（多一次前向）、Data 低、Engineering 中、Timeline 中。
- **裁决**：**Strong Accept**（H/S/B 三维 8+、零 CRITICAL），`worth pursuing, pending the validation experiment`。
- **决定性验证实验**：「只 forget+donor」vs「加 retain floor」两臂测 $U$ 与 $r_f$ 帕累托；donor 负约束开/关测 donor 答案率；`needs manual pilot`。
- 分数为执行者机制推断分；`novelty-check = PROVISIONAL`，非跨模型回执。

## 6. 判别实验、失败模式、必测指标

- **判别**：三方约束可行解是否非空；retain floor 是否真的把 $U$ 托住而 $r_f$ 仍降；donor 负约束是否显著降低答案替换率。
- **失败模式**：$r_f$ 降必然 $U$ 崩（不可兼得）；donor 负约束对答案替换率无影响；$\Delta A$ 三路投影病态（角色子空间重叠）；约束过紧导致无可行解。
- **必测指标**：$r_f$、$U$（retain NLL / MMLU 代理）、donor 答案率、其他错答/拒答/提前 EOS/完整回答率、$\nabla\mathcal L_F\!\cdot g_d$ 余弦、3 seed std。

## 7. 与方向 I 基线 v0.6.1 的关系

保留 §4.1 三条中性约束、§5 donor 数据边界（不拼 $y^-$）、§2.2 梯度边界、$\lambda_{KL}^{\text{ablation}}=0$、§6 六条验收；把 §2.3 的 $T_{\ell,p}$ 从「单方中性可行集」扩为「三方约束解」。donor 信息在此只用于**定位禁止方向**而非指定目标答案，正面回应 §2.2.2 对「重新论证 donor 边界」的义务。历史负结果不复用为正向证据。

## 8. 诚信声明与来源

plan-level，不新增实验；claim 待验证；分数机制推断、PROVISIONAL、`REVIEW_UNAVAILABLE`；新结果须过 `/experiment-audit` + `/result-to-claim`。

- 基线 §1/§2.2.2/§2.3/§3.5/§4.1/§5/§6；`IDEA_INVERSE_TARGET.md` R6
- 近邻（身份未 D1 核验，不作 BibTeX）：CLReg arXiv:2601.22028、D2F arXiv:2610.17654、CPSD arXiv:2611.31932
- 记忆卡 `mem_b42a20_c84bcae`（E0 效用崩）
