---
id: CF-08
rank: 7
title: 零空间 / 抗恢复投影约束 ΔA_null
avg_score: 7.4
verdict: Accept with Revisions (PROVISIONAL, mechanism-based)
decision_source: IDEA_DISCOVERY_20261126.md §5
domain: LLM/LRM Unlearning / 抗恢复激活差投影
evidence_level: provisional（机制推断，无实测，无跨模型回执）
baseline_relation: 把 §2.2.1 待选「零空间 / 抗恢复投影」定稿，联动 §6 第 5 条抗恢复证据
---

# CF-08 · 零空间 / 抗恢复投影 $\Delta A_{\text{null}}$

## 0. 故事（30 秒讲稿）

> 用户要「真遗忘」而非「暂时读不出」。Field & Yampolskiy (ICML 2026 Workshop) 证明 SAE 特征抑制只改「说什么」不改「知道什么」——这是全领域的头号 killer risk。我们把「抗恢复」直接写进 $\Delta A$ 的构造：让遗忘方向避开攻击者会覆盖的高方差子空间（retain 梯度 top-PC），落到底部/零空间 $\mathrm{Null}(G_{\text{retain}}^\top)$，使沿 $\Delta A_{\text{null}}$ 的更新不易被少量微调或量化找回。可解释性来自：我们能说清「为什么这个方向不易被恢复」——因为它与共享知识梯度正交，并与 §4.1 中性约束联动。

## 1. 领域问题与空白

- 基线 §6 第 5 条要求「结合输出访问变体、独立 probe/patching、量化和受控 relearning，至少两类独立抗恢复证据改善，才允许受限『深度遗忘』表述」；但 §2.2.1 未把抗恢复写进 $\Delta A$ 构造。
- 项目实测：ASU-u 的 P2 恢复率 $0.524$、P3 $0.371$、$\text{mean\_recovery}=0.448$（来源：`docs/research_contract.md` C1、`IDEA_REPORT_V4.md`），确认浅层遗忘；R2-C 已把抗恢复前移进 minimax 目标，但在输出/参数层。
- Field & Yampolskiy (ICML 2026 Workshop)「SAE Feature Unlearning is Shallow」：抑制单特征只压生成、理解力/MMLU 不变，直接质疑所有 activation-based 干预是否都浅层。
- **近邻空白**：AlphaEdit 零空间编辑、RepSelect bottom-PC、FALCON 正交投影、R2-A「ASU-Selective 子空间」已因新颖性不足归档（`IDEA_REPORT_V4.md`）；**没有人把「零空间/抗恢复投影」作为反事实配对差 $\Delta A$ 的构造算子、并与 §4.1 中性约束 + 留出第三类恢复攻击交叉验收联动**。

## 2. 核心假设（可证伪）

**H**：把 $\Delta A$ 投影到 retain 梯度 top-PC 的零空间后，所得 $\Delta A_{\text{null}}$ 仍携带足够事实信息（投影后 $r_f$ 可降），且在 P3 relearning / INT4-INT8 量化 / probe 恢复下显著优于原始 $\Delta A$。若零空间与事实方向不相交（投影后无事实信息），或只在带 hook/带约束时有效、无 hook 崩塌，则本思路被证伪。

## 3. 具体实现（LaTeX）

**retain 梯度基**（在 retain-train 上取 top-PC）：

$$
G_{\text{retain}}=\big[\ \nabla_\theta \mathcal L_R^{(1)}\ \cdots\ \nabla_\theta \mathcal L_R^{(m)}\ \big],\qquad
U_{\text{top}}=\mathrm{top\!-\!r\ PC}(G_{\text{retain}}),\qquad
P_{\perp}=I-U_{\text{top}}U_{\text{top}}^{\top}.
$$

**零空间投影 $\Delta A$**（避开攻击者高方差子空间）：

$$
\Delta A_{\text{null},\ell,p}=P_{\perp}\,\Delta A_{\ell,p},\qquad
\Delta A_{\ell,p}=A^+_{T,\ell}(p)-A^-_{T,\ell}(p),\quad p\in S_f .
$$

**与 §4.1 联动的目标**（投影方向作引导、非 donor）：

$$
T_{\ell,p}=\operatorname{stopgrad}\!\left(\Pi_{\mathcal K}\big[\,h^{+}_{\ell}(p)-\alpha\,\widehat{\Delta A_{\text{null},\ell,p}}\,\big]\right),
$$

$$
\mathcal K=\{h:\ r_{f,p}(h)\le c,\ s_{d_j,p}(h)\le d_j,\ \lVert\Delta r_{E,p}(h)\rVert\le e\},\qquad
\boxed{\ \mathcal L=\lambda_F\mathcal L_F+\beta\mathcal L_R\ } .
$$

**抗恢复交叉验收**（呼应 §6 第 5 条）：在 P3（少量同域微调）、INT4/INT8 量化、线性 probe、留出第三类攻击上，比较「$\Delta A_{\text{null}}$ 内化」 vs 「原始 $\Delta A$ 内化」 vs 「仅带 hook 的推理时投影」三臂，且**无 hook 的 Student 仍须保持效应**（参数内化硬要求）。

## 4. 差异化（novelty 落点）

| 近邻 | 本方案差异 |
|---|---|
| AlphaEdit / RepSelect (2606.17168) / FALCON | 后者是通用模型编辑/子空间方法；本方案把零空间约束加在**反事实配对差 $\Delta A$ 的构造**上，作为 v0.6.1 框架内的一个算子 |
| R2-A (ASU-Selective，已归档) | 后者是独立两段式子空间方法且新颖性不足；本方案不训练新模块、是 $\Delta A$ 构造算子，且强制无 hook 内化 |
| R2-C（抗恢复即目标，现役候选） | 后者把恢复攻击前移进 minimax 训练目标（输出/参数层）；本方案是激活差方向的**投影约束**，二者可叠加但机制轴不同 |
| Field & Yampolskiy 2026 | 后者是「浅层」负面证据；本方案把它转成必须过的抗恢复判别门 |
| V1 正交抑制（已证伪） | 后者沿 $\Delta A$ 抹表征且已证伪；本方案投影到零空间且只作定位/读出，不作删除方向 |

**novelty 落点**：把「抗恢复 / 深度遗忘」从**事后验收指标**升级为 $\Delta A$ 的**构造期硬约束**，并用「无 hook 内化 + 留出第三类攻击」堵住「只在带约束时有效」的伪深度。

## 5. /idea-evaluator 评估（PROVISIONAL）

| 维度 | 分 | 证据（机制推断，未实测） | Lift |
|---|---|---|---|
| Higher | 7 | 抗恢复是附加约束，对瞬时 forget-retain 增益不一定最高（`mechanism-based`） | 与 CF-01 逆问题叠加 |
| Faster | 7 | 多一次 retain 梯度 PC 计算，可控 | 复用 $\mathcal L_R$ 梯度 |
| Stronger | 9 | 零空间 + 抗恢复交叉验收 + 无 hook 内化，直接针对「浅层抑制」头号风险（`mechanism-based`） | 加第四类恢复攻击 |
| Cheaper | 6 | 需额外恢复攻击预算（P3/量化/probe） | 复用 R3-2 恢复阶梯 |
| Broader | 8 | 抗恢复投影可迁移其它激活差方法（`mechanism-based`） | 与 R3-2 共享阶梯 |

**平均分 $\bar S=37/5=7.4>7$。**

- **定位**：Novel Method；一句话故事见 §0。
- **致命缺陷审计**：**2 MAJOR** → (1) F1 与 AlphaEdit/RepSelect/R2-A 划界须核验（R2-A 已因新颖性不足归档，必须给出机制差异实验）；(2) F6 抗恢复 + 无 hook 内化判别须预注册。零 CRITICAL。故 `Accept with Revisions`。
- **生命周期**：Frontier exploration 3–9 个月，能力匹配绿。
- **范式探测**：First Principles=Partial、Elephant=Yes（浅层遗忘是公认痛点）、Technology Cycle=Partial、Hamming=Yes → possible。
- **可行性**：Compute 中（恢复攻击）、Data 低、Engineering 中、Timeline 中。
- **裁决**：**Accept with Revisions**，`worth pursuing, pending the validation experiment`。**Revisions**：先补齐 R2-A/RepSelect/AlphaEdit 划界表与 E4 式机制划界实验，再开跑。
- **决定性验证实验**：P3 恢复率、INT4/INT8 量化恢复在「零空间 $\Delta A$」vs「原始 $\Delta A$」下的差异；retain NLL 是否受损；无 hook 内化是否保持；`needs manual pilot`。
- 分数为执行者机制推断分；`novelty-check = PROVISIONAL`，非跨模型回执。

## 6. 判别实验、失败模式、必测指标

- **判别**：投影后是否仍有事实信息（$r_f$ 可降）；恢复曲线下面积是否显著改善；无 hook 是否保持。
- **失败模式**：抗恢复子空间与事实方向不相交（投影后无信息）；retain 崩塌；只在带 hook/带约束时有效、无 hook 崩塌；与 R2-A 失败模式同源（收益来自总平滑而非结构化）。
- **必测指标**：P1/P2/P3/INT4/INT8/Probe 恢复曲线下面积、retain NLL/MMLU、$r_f$、无 hook 效应、donor 答案率、3 seed std。

## 7. 与方向 I 基线 v0.6.1 的关系

定稿 §2.2.1 待选「零空间/抗恢复投影」；$S_f$ 作用域、Teacher stopgrad、§4.1 中性约束（$\Pi_\mathcal K$）、读出经冻结下游重算、$\lambda_{KL}^{\text{ablation}}=0$ 全部沿用；把 §6 第 5 条「至少两类独立抗恢复证据」前移为构造期的判别门，并与 §6 第 4 条「参数内化、无 hook」联动。历史 stage0–14 负结果不复用为正向证据。

## 8. 诚信声明与来源

plan-level，不新增实验；claim 待验证；分数机制推断、PROVISIONAL、`REVIEW_UNAVAILABLE`；新结果须过 `/experiment-audit` + `/result-to-claim`。

- 基线 §2.2.1/§4.1/§6.4/§6.5；`IDEA_CANDIDATES.md` D10；`IDEA_REPORT_V4.md`、`docs/research_contract.md`（C1 恢复率、E4 划界义务）
- 近邻（身份未 D1 核验，不作 BibTeX）：AlphaEdit、RepSelect arXiv:2606.17168、FALCON、Field & Yampolskiy ICML 2026 Workshop、R2-A 归档报告
