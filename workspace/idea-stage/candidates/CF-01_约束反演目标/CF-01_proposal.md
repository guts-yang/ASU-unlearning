---
id: CF-01
rank: 1
title: 约束反演目标 Δ* —— 因果 + 效用 + 身份三约束逆问题
avg_score: 8.0
verdict: Strong Accept (PROVISIONAL, mechanism-based)
decision_source: IDEA_DISCOVERY_20261126.md §5
domain: LLM/LRM Unlearning / 可解释激活目标构造
evidence_level: provisional（机制推断，无实测，无跨模型回执）
baseline_relation: 基线 §2.2.1「ΔA 构造未定稿」的良置化定稿；唯一改动点是 T_{ℓ,p} 的定义
---

# CF-01 · 约束反演目标 $\Delta^*$（三约束逆问题）

## 0. 故事（30 秒讲稿）

> 用户想「看正反事实对的前向差异 → 反推遗忘目标该往哪走」。朴素版有两个洞：观察不等于干预（同一趋势可由无穷多个潜在变化产生，是欠定逆问题），且 forget+donor 两次前向根本没观测「效用」，无从保证保能力。我们不外推趋势，而是把「抹去 $\Delta A$」写成**良置的约束反演**：在冻结 $\theta_0$ 上解一个目标位移 $\Delta^*$，它同时满足「原事实读不出」「retain 效用不动」「不沿主体身份方向走」「落在 retain 梯度零空间」，再 patch 回冻结下游做**因果验证**淘汰 OOD 假峰，最后让 Student 把 $\Delta^*$ 内化进参数。可解释性来自：$\Delta^*$ 的每一条约束都对应一句人能读懂的话（删什么 / 保什么 / 别变成什么 / 抗恢复）。

## 1. 领域问题与空白

- **观察 ≠ 干预**：$\Delta A_{\ell,p}=A^+_{T,\ell}(p)-A^-_{T,\ell}(p)$ 是「主体替换差」（是谁），不是「事实删除差」（是什么）；基线 §2.2.1 明列它混有实体身份、subword、词频、关系传播，历史证据调研判定 $A^+-A^-$ 不能直接读成事实差异（来源：基线 §1、§2.2.1、历史证据调研）。从「看趋势反推」在数学上是欠定逆问题。
- **效用不可从两次前向读出**：效用是 retain 分布上的性质；forget+donor 前向没观测它。E0 失败教训：对全部层统一升温，forget 降了但 retain NLL 全部超上限、最小损伤约 81.6 倍（来源：跨会话记忆卡 `mem_b42a20_c84bcae`）。
- **近邻空白**：CAA/ActAdd 用 $v=\mathrm{mean}(h^+)-\mathrm{mean}(h^-)$ 但只做推理时相加、非持久；SCALPEL(arXiv:2609.31056)/CRISP(arXiv:2508.13650) 在 SAE 特征空间减贡献；CPSD(arXiv:2611.31932) 在输出分布层反转对比差；**没有人在中间层激活差上解「带因果验证 + 效用锚定 + 身份/零空间约束」的逆问题并训练内化**。

## 2. 核心假设（可证伪）

**H**：存在一个非空可行集，使 $\Delta^*$ 同时让目标答案跨度读出 $r_f$ 降到中性死区、retain 侧效用 $U$ 不低于地板、且与身份方向/共享知识梯度零空间相容；且经 patch 因果验证后仍有显著分量。若该可行集为空，或 $\Delta^*$ 退化为原始 $\Delta A$ 趋势，则本思路被证伪。

## 3. 具体实现（LaTeX）

**Stage 1 三次前向（不是两次）**：冻结 $\theta_0$ 分别前向 $X^+\oplus y^+$、$X^-\oplus y^+$、$X_r\oplus y_r$（retain 样本），在 $S_f$ 取 $h^+,h^-,h^r$：

$$
A^{\pm}_{T,\ell}(p)=h_{T,\ell}(X^{\pm}\oplus y^+;p),\quad
\Delta A_{\ell,p}=A^+_{T,\ell}(p)-A^-_{T,\ell}(p),\qquad p\in S_f .
$$

**Stage 2 约束反演（解 $\Delta^*$，不是外推）**：

$$
\Delta^{*}_{\ell,p}=\arg\min_{\Delta}\; r_f\!\left(h^{+}_{\ell}(p)+\Delta\right)+\gamma\lVert\Delta\rVert^2
$$

$$
\text{s.t.}\quad
\begin{cases}
r_f(h^{+}_{\ell}(p)+\Delta)\le c & \text{（原事实读不出，§4.1）}\\
U(h^{+}_{\ell}(p)+\Delta)\ge U(h^{+}_{\ell}(p))-\delta & \text{（效用地板）}\\
\langle \Delta,\, v_{\text{id}}\rangle\le 0 & \text{（不沿主体身份方向）}\\
\Delta\in \mathrm{Null}(G^{\top}_{\text{retain}}) & \text{（retain 梯度 top-PC 零空间）}\\
\lVert\Delta\rVert\le\varepsilon,\quad s_{d_j,p}\le d_j,\quad \lVert\Delta r_{E,p}\rVert\le e & \text{（非 donor / 反 EOS 捷径）}
\end{cases}
$$

其中 $r_f$ = 目标事实 span readout、$U$ = retain 读出效用、$v_{\text{id}}$ = 身份子空间方向、$G_{\text{retain}}$ = retain 梯度 top-PC 基、$c,\delta,\varepsilon$ 在 discovery 冻结。

**Stage 3 因果验证（杀 OOD 假峰，呼应 Zhang & Nanda ICLR 2024）**：把 $\Delta^*$ patch 回冻结 $\theta_0$ 下游做 patch-forward，保留满足 $\Delta r_f<-\eta_c$ 且 $|\Delta U|<\eta_U$ 的分量：

$$
\Delta^{*}_{\text{causal}} = \sum_{\ell,p} w_{\ell,p}\,\Delta^{*}_{\ell,p},\qquad
w_{\ell,p}=\mathbb 1\!\left[\,\mathrm{Eff}^{\text{patch}}_{r_f}(\ell,p)<-\eta_c\ \wedge\ |\mathrm{Eff}^{\text{patch}}_{U}(\ell,p)|<\eta_U\,\right].
$$

**Stage 4 内化（回到基线主损失，形式不变）**：

$$
T_{\ell,p}=\operatorname{stopgrad}\!\left(h^{+}_{\ell}(p)+\Delta^{*}_{\text{causal},\ell,p}\right),
$$

$$
\mathcal L_F(k)=\frac{1}{|\Lambda(k)|\,|S_f|}\sum_{\ell\in\Lambda(k)}\sum_{p\in S_f}\rho_\ell(k)\,
\operatorname{dist}\!\left(h_{S,\ell}(X^+\oplus y^+;p),\,T_{\ell,p}\right),
\qquad
\boxed{\ \mathcal L=\lambda_F\mathcal L_F+\beta\mathcal L_R\ } .
$$

$\mathcal L_R$ 为独立 retain-train 答案位 NLL；$\lambda_{KL}^{\text{(ablation)}}=0$；所有读出经冻结下游重算，不直接过 LM head。

## 4. 差异化（novelty 落点）

| 近邻 | 作用层 | 本方案差异 |
|---|---|---|
| CAA / ActAdd / Spherical Steering (ICML2026) | 推理时残差流加法/旋转 | 本方案解参数内化的逆问题，非推理时 hook、非持久 steering |
| SCALPEL (2609.31056) / CRISP (2508.13650) | SAE 命名特征空间 | 本方案在原始中间层 $\Delta A$ 上，且加因果 + 效用 + 身份三约束与 patch 验证 |
| CPSD (2611.31932) | 输出 token 分布 | 本方案在中间层激活向量差，目标是 $\Delta^*$ 而非反转分布 |
| 基线 v0.6.1 §2.3 | 「中性可行集里离 $A^+$ 最近点」 | **唯一改动点**：$T_{\ell,p}$ 改为「因果验证过、效用锚定的逆问题解」，补第三次 retain 前向、patch 因果验证、身份/零空间约束 |
| V1 全系（已证伪） | 沿 $\Delta A$ 抑制 / 对齐 $A^-$ | 本方案 $\Delta A$ 至多作初始化或负约束，**不作删除方向**、不教 donor 答案 |

**novelty 落点**：把「反事实激活差 → 遗忘目标」从欠定观察问题改造成带因果验证与效用锚定的**约束反演**，并给出可解释的四条语义约束；不与已证伪的「沿 $\Delta A$ 抑制」同族。

## 5. /idea-evaluator 评估（PROVISIONAL）

| 维度 | 分 | 证据（机制推断，未实测） | Lift |
|---|---|---|---|
| Higher | 9 | 三约束 + 因果验证直接针对「浅层抑制/答案替换/效用崩」三大失败模式，机制上应有更干净的 forget-retain 权衡（`mechanism-based`） | 强调 $r_f$ / $U$ 帕累托 |
| Faster | 7 | 三次冻结前向 + 少量 patch 扫描 + 60-step 量级训练，无新模块 | 缓存 $S_f$ 激活 |
| Stronger | 9 | 因果筛选 + 零空间约束 + 反 EOS/donor 约束，对 OOD/恢复攻击有原则性防护（`mechanism-based`） | 加 relearning/量化交叉验收 |
| Cheaper | 7 | 复用冻结 Teacher，无 SAE 训练、无标注、无外部 oracle | 复用 stage0 激活资产 |
| Broader | 8 | 逆问题框架可迁移到 MUSE/WMDP 与 LRM 推理轨迹跨度（`mechanism-based`） | 与 CF-06 联动 |

**平均分 $\bar S = 40/5 = 8.0 > 7$。**

- **定位**：Novel Method（STEM / 实验 + 模型）。一句话故事见 §0。
- **致命缺陷审计**：1 MAJOR（F6 决定性实验必须先预注册：$\Delta^*$ vs 基线最近点 vs 趋势外推三臂）+ 2 MINOR（F1 近邻划界待核验、F3 需 2026 SOTA 作主基线）。**零 CRITICAL**：本方案带新机制，不落 `research-wiki/log.md` 已证伪族。
- **生命周期**：Frontier exploration 3–9 个月；单卡 H20、3 seed、TOFU-1B 量级，能力匹配绿。
- **范式探测**：First Principles=Yes（挑战「激活差趋势即删除方向」）、Elephant=Partial（浅层抑制是公认痛点）、Technology Cycle=Partial、Hamming=Yes → 有 disruptive seeds。
- **可行性**：Compute 中（可逆问题在冻结模型前向、无训练）、Data 低（TOFU 内部 donor）、Engineering 中（patch 扫描）、Timeline 中。
- **裁决**：**Strong Accept**（2 维 8+、零 CRITICAL），`worth pursuing, pending the validation experiment`。
- **决定性验证实验（Top-1 行动）**：三臂对照 $\Delta^*$ 目标 / 基线最近点目标 / 朴素趋势外推目标，测 $r_f$、retain readout、donor 答案率、无 hook 内化程度；`needs manual pilot`。
- 分数为执行者机制推断分，非跨模型独立评审回执；`novelty-check = PROVISIONAL`。

## 6. 判别实验、失败模式、必测指标

- **决定性判别**：$\Delta^*$ 是否在「$r_f$ 降 + $U$ 不崩 + 无 donor 替换」三项上同时优于基线最近点目标；因果筛选 vs 相关筛选的 $\Delta^*$ 在留出恢复攻击下的差异。
- **失败模式**：逆问题无可行解；$\Delta^*$ 退化为 $\Delta A$ 趋势（无新增信息）；效用约束形同虚设（$U$ 崩）；patch 引入 OOD 伪定位峰；与 V1 失败模式同型则撤回。
- **必测指标**：$r_{f,p}$ / $s_{d_j,p}$ / $\Delta r_{E,p}$、retain NLL、donor 答案率、$\nabla\mathcal L_F$ 与 donor 提升 $g_d$ 余弦、提前 EOS、完整回答率、P2/P3/INT8 恢复、无 hook 复现、3 seed std。

## 7. 与方向 I 基线 v0.6.1 的关系

只把 §2.2.1「$\Delta A$ 构造未定稿」与 §2.3 的 $T_{\ell,p}$ 定义定稿为逆问题解；主损失形式、$S_f$ 作用域、Teacher stopgrad、$\lambda_{KL}^{\text{ablation}}=0$、§4.1 中性约束、读出经冻结下游重算、§6 六条验收全部沿用。对应 §2.2.2 裁决 (A) 的精神（配对重回主梯度来源）但走**新机制**：不直接对齐 $A^-$、不教 donor 答案。历史 stage0–14 结果不复用为正向证据。

## 8. 诚信声明与来源

plan-level 候选，不新增实验；所有 claim 标「待验证」；分数机制推断、PROVISIONAL、`REVIEW_UNAVAILABLE`，不伪造回执；新结果须过 `/experiment-audit` + `/result-to-claim`。

- 基线 §1/§2.2/§2.2.1/§2.3/§4.1/§5/§6；`IDEA_INVERSE_TARGET.md` R1/R5/R6
- 近邻（身份未 D1 核验，不作 BibTeX）：CAA/ActAdd、Spherical Steering arXiv:2602.08169、SCALPEL arXiv:2609.31056、CRISP arXiv:2508.13650、CPSD arXiv:2611.31932、Zhang & Nanda ICLR 2024、Field & Yampolskiy (ICML 2026 Workshop)
- 记忆卡 `mem_b42a20_c84bcae`（E0 效用崩教训）
