# "观察前向趋势 → 反推目标变化量"思路评估与候选集

**日期**：2026-11-26　**分支**：Auto/20261122（HEAD d8c3db5）
**基线**：`workspace/idea-stage/方向I_基线_v0.6.1_answer_span.md`（plan-only）
**用户问题**：能否先用 forget 集与 donor 前向传播评估中间潜在向量的变化趋势，再反推"实现遗忘同时保留效用"的潜在变化量应如何设计？
**模式**：构思与规划，pilot skip，无 GPU，不训练、不部署、不改 configs/、不访问其它线路
**诚信声明**：全部 plan-level / 机制推断。论文级 claim 一律「待验证」，依赖 pilot 的结论一律 `needs manual pilot`。novelty-check = PROVISIONAL，research-review = REVIEW_UNAVAILABLE，evidence gate = BLOCKED。

## 0. 一句话评估：方向对，朴素版有两个致命漏洞

**方向对的地方**：这正是基线 §2.2.1「ΔA 构造未定稿」的最自然攻法——用两次前向的差异反推出目标变化量，而不是拍脑袋指定目标。它落在 activation steering / CAA 的经典做法（对比前向求差向量，`v = mean(h⁺) − mean(h⁻)`）里，也接近 CUSE（反事实引导 + on-policy 自蒸馏）与 ACSD（从轨迹对比抽 steering 向量监督冻结 teacher）。

**漏洞一：观察 ≠ 干预（信息论欠定）**。forget/donor 的前向差 $\Delta A_{\ell,p}=A^+_{T,\ell}(p)-A^-_{T,\ell}(p)$ 是**主体替换差**（"是谁"），不是**事实删除差**（"是什么"）。基线 §2.2.1 自己写明它混有实体身份、subword、词频、关系传播，且历史证据调研明确判定 $A^+-A^-$ **不能直接读成事实差异**。更根本的是：同一个可观察趋势可由**无穷多个**潜在变化量产生；"看趋势反推"在数学上是欠定逆问题，必须加因果判据才能定解。记忆卡（方向 I donor 构造 v0.3）也记录了 Zhang & Nanda ICLR 2024 的结论：加噪 donor 是 OOD 伪定位峰——只靠观察会选中假峰。

**漏洞二：forget + donor 两次前向读不出"效用"**。效用（retain）是另一个分布上的性质；你没在前向里观测 retain，就无从约束"保留效用"。这正是历史 E0 基线失败的教训（对全部层统一升温，forget 降了但 retain NLL 全部超预注册上限、最小损伤约 81.6 倍，判决 `do_not_start_e1`）。

**一针见血的修正**：把"外推趋势"改造成**带因果验证 + 效用锚定的约束反演**，并补上**第三次前向（retain）**。不是"趋势指向哪就往哪走"，而是"解出一个满足事实读不出、效用不动、且被 patch 因果验证过的 $\Delta^*$，再让 Student 内化它"。

## 1. 正确打开的姿势：把思路写成良置逆问题

四阶段闭环（每阶段都可独立证伪）：

**Stage 1 观察（三次前向，不是两次）**：冻结 $\theta_0$ 上分别前向 $X^+\oplus y^+$、$X^-\oplus y^+$、$X_{r}\oplus y_{r}$（retain 样本），在答案跨度 $S_f$ 取 $h^+$、$h^-$、$h^{r}$。

**Stage 2 反演（解 $\Delta^*$，不是外推）**：

$$
\Delta^{*}_{\ell,p}=\arg\min_{\Delta}\; r_f\!\left(h^{+}_{\ell}(p)+\Delta\right)
$$

$$
\text{s.t.}\quad
r_f(h^{+}+\Delta)\le c,\;\;
U(h^{+}+\Delta)\ge U(h^{+})-\delta,\;\;
\langle \Delta, v_{\text{id}}\rangle\le 0,\;\;
\Delta\in \mathrm{Null}(G^{\top}_{\text{retain}}),\;\;
\|\Delta\|\le\varepsilon
$$

$r_f$ = 目标事实 span readout；$U$ = retain 侧读出效用；$v_{\text{id}}$ = 身份子空间方向；$G_{\text{retain}}$ = retain 梯度 top-PC。

**Stage 3 因果验证（杀 OOD 假峰）**：把 $\Delta^*$ patch 回冻结 $\theta_0$ 的下游（patch-forward），确认 $r_f$ 真降且 $U$ 不动；只保留因果效应显著的 $\Delta^*$ 分量。

**Stage 4 内化（回到基线主损失）**：$\min_\theta\ \|\,h_{S,\ell}(X^+\oplus y^+;p)-\mathrm{stopgrad}(h^+ +\Delta^*)\,\|^2 + L_R$，仍满足 §4.1 非 donor 中性约束、`λ_KL^{ablation}=0`、读出经冻结下游重算。

**与基线的真实增量**：基线的 $T_{\ell,p}$ 是"中性可行集里离 $A^+$ 最近的点"，**没有**显式解 $\Delta^*$、**没有**第三次 retain 前向、**没有** patch 因果验证、**没有**按身份子空间/零空间约束方向。这四点就是新意的落点。

## 2. 评分口径（跨会话记忆冻结口径）

`S = 0.30·I + 0.25·M + 0.20·T + 0.15·N + 0.10·C`，各维 0–10，不四舍五入，≤0.3 视为同档。I=信息价值 / M=机制契合度 / T=最小实验可判别性 / N=相对近邻差异潜力 / C=简洁性。**分数为执行者机制推断分，非实测成绩。**

## 3. 评分表（6 个候选，全部 > 8）

| # | Idea | I | M | T | N | C | S | 档 |
|---|---|---|---|---|---|---|---|---|
| R1 | 约束反演目标 Δ*（因果 + 效用双约束的逆问题） | 9.5 | 9.5 | 8.5 | 8.5 | 8.0 | **9.000** | 一 |
| R5 | "趋势 ≠ 删除"证伪门（先刀掉朴素版） | 9.0 | 8.5 | 9.5 | 7.5 | 9.0 | **8.750** | 一 |
| R6 | 三前向角色分解（删什么 / 别变成什么 / 保什么） | 9.0 | 9.0 | 8.5 | 8.0 | 8.5 | **8.700** | 一 |
| R3 | 因果闭环作逆问题求解器（patch 验证器进内环） | 8.5 | 9.0 | 9.0 | 7.5 | 8.0 | **8.525** | 二 |
| R2 | on-policy 闭环反演（DAgger 式重估 Δ*） | 9.0 | 8.5 | 8.0 | 8.5 | 7.5 | **8.450** | 二 |
| R4 | 效用锚定零空间反演（把"保效用"写成硬约束） | 8.5 | 9.0 | 8.5 | 7.0 | 8.0 | **8.350** | 二 |

---

## R1 — 约束反演目标 Δ* · S=9.000【待验证】

- **核心假设**：用户思路的良置版——不从 ΔA 趋势外推，而在冻结 $\theta_0$ 上解 §1 的约束逆问题得到 $\Delta^*$，令目标 $T_{\ell,p}=h^+ +\Delta^*$，再训练 Student 内化。
- **唯一改动点**：$T_{\ell,p}$ 的定义从"中性可行集最近点"改为"因果验证过、效用锚定的逆问题解"；主损失形式、$S_f$ 作用域、`λ_KL=0` 均不变。
- **判别实验**：$\Delta^*$ 目标 vs 基线最近点目标 vs 趋势外推目标三臂；测 $r_f$、retain readout、donor 答案率、是否需 hook。`needs manual pilot`。
- **seed / 预算**：3 seed；单卡 H20；逆问题在冻结模型上前向 + 少量 patch，训练 60-step 量级；$c,\delta,\varepsilon$ 实验前冻结。
- **失败判据**：逆问题无可行解；$\Delta^*$ 退化为 $\Delta A$ 趋势（无新增信息）；效用约束形同虚设（$U$ 崩）。
- **不与淘汰重复**：V1 是"沿 ΔA 抑制/对齐 A⁻"且已证伪；这里是"求解目标位移"，ΔA 至多作初始化或负约束，不作删除方向，且不教 donor 答案。

## R5 — "趋势 ≠ 删除"证伪门 · S=8.750【待验证】

- **核心假设**：用户思路的朴素版（趋势外推）必须先被证伪或证实——把 $\Delta A$ 趋势方向当作 $\Delta$ 直接 patch，观察它是让 $y^+$ 读不出，还是只是把主体身份挪走。若趋势方向不能降 $r_f$，则所有"看趋势反推"方法（含 R1 之前的朴素版）当场死亡。
- **唯一改动点**：不改训练，新增一个 pre-registration 门：趋势方向 patch 的因果读出测试。
- **判别实验**：patch $\Delta A$ 趋势方向 vs 等范数随机 vs wrong-donor 方向，比较 $r_f$ 变化；与事实 probe / 身份 probe 的对应。`needs manual pilot`。
- **seed / 预算**：3 seed；纯前向 + hook，零训练成本；应作为其余 5 个 idea 的前置门。
- **失败判据**：趋势方向与随机方向不可分，或只动身份不动事实 → 朴素思路否决，仅保留 R1/R6 的约束反演版。
- **不与淘汰重复**：不是重提 V1（V1 拿趋势方向去训练 Student），而是**先验尸检**这个方向是否有效，直接继承 stage8/9/13 的负结果作为风险提示，不把负结果改写成正向证据。

## R6 — 三前向角色分解 · S=8.700【待验证】

- **核心假设**：用户说"用 forget 和 donor 前向"，但缺了第三路。明确三路角色：forget 前向定"删什么"（$r_f$ 目标）、donor 前向定"不要变成什么"（负约束，防答案替换）、retain 前向定"保什么"（效用地板）。$\Delta^*$ 解一个三方约束问题。
- **唯一改动点**：反演的目标函数从二方（forget/donor）变三方（+retain floor），donor 由"对照"升级为"负约束来源"（仍不进目标语义、不拼 $y^-$）。
- **判别实验**：只 forget+donor vs 加 retain floor 两臂，比较 $U$ 与 $r_f$ 的帕累托；donor 负约束开/关测答案替换率。`needs manual pilot`。
- **seed / 预算**：3 seed；retain 前向为额外一次冻结前向，成本可控；floor $\delta$ 冻结。
- **失败判据**：retain floor 与事实删除方向不可兼得（$r_f$ 降必然 $U$ 崩）；donor 负约束无法降低答案替换率。
- **不与淘汰重复**：donor 只做"禁止"不做"目标"，与 v0.5 donor-KL 共谋（被基线 §1 判死）严格区分；且 $y^-$、`donor_fact_value` 仍不进任何前向拼接入。

## R3 — 因果闭环作逆问题求解器 · S=8.525【待验证】

- **核心假设**：R1 的逆问题若只用相关性目标会选中 OOD 假峰；把 patch 因果验证放进了问题的**内环**：只有 patch 回冻结下游后 $r_f$ 真降且 $U$ 不动的分量才被保留进 $\Delta^*$。
- **唯一改动点**：求解器从"梯度下降拟合目标 readout"改为"patch-forward 因果筛选 + 加权"，与 D3（因果门控）互为框架内实现。
- **判别实验**：因果筛选 vs 相关筛选的 $\Delta^*$ 在留出攻击（relearning / 量化）下的差异；与 UDS 的 KE 层定位一致性。`needs manual pilot`。
- **seed / 预算**：3 seed；因果扫描为前向 + hook，无训练；patch 层集与阈值冻结。
- **失败判据**：因果效应全网均匀（筛选退化）；patch 引入 OOD 伪定位峰（呼应 Zhang & Nanda ICLR 2024 与记忆卡的 prompt-only 退回理由）。
- **不与淘汰重复**：D3/D4 是构造纯化；这里是**求解器机制**，服务于 R1/R6 的反演闭环，不是独立方法。

## R2 — on-policy 闭环反演 · S=8.450【待验证】

- **核心假设**：基线（及 R1）的 $\Delta^*$ 在冻结 $\theta_0$ 上算一次；但 Student 更新后，其实际访问的状态分布已变，原 $\Delta^*$ 失配。按 DAgger/OPD 逻辑，沿 Student 自身轨迹**重估** $\Delta^*$（ACSD 即从轨迹对比抽 steering 向量监督冻结 teacher）。
- **唯一改动点**：反演从一次性改为迭代：每若干步用当前 Student 的 visited states 重算 $\Delta^*$，teacher 冻结、方向/layer/strength 在校准池选定后固定（与 ACSD 一致）。
- **判别实验**：静态 $\Delta^*$ vs on-policy 重估，等预算比较 $r_f$ / $U$ / 跨种子 std；方向跨 step 漂移量。`needs manual pilot`。
- **seed / 预算**：3 seed；重估周期为额外超参，冻结；不改模型架构。
- **失败判据**：重估方向跨步近正交（不稳定）；与静态无差异（分布错配不存在）；on-policy 轨迹把 donor 答案 reintroduce。
- **不与淘汰重复**：与 D7（ΔA 纯度驱动 schedule）不同——D7 选层，R2 改的是目标本身随学生分布演化；近邻 OPD 综述、ACSD、CUSE、DistIL，均非遗忘域，划界在"反演目标 × 遗忘语义"。

## R4 — 效用锚定零空间反演 · S=8.350【待验证】

- **核心假设**：直接回答用户"保留效用"的那一半——把 $\Delta^*$ 限制在 retain 梯度 top-PC 的零空间（RepSelect 高方差方向与通用能力共享、AlphaEdit 零空间编辑），从方向上保证不动共享知识。
- **唯一改动点**：逆问题加硬约束 $\Delta\in\mathrm{Null}(G_{\text{retain}}^{\top})$，与 §4.1 中性约束联动。
- **判别实验**：零空间约束开/关，测 retain NLL / MMLU 代理、P3 恢复率、INT8 恢复；与留出的第三类恢复攻击交叉验收。`needs manual pilot`。
- **seed / 预算**：3 seed；投影秩、retain 梯度样本、攻击预算 η 冻结。
- **失败判据**：零空间与事实方向不相交（投影后无事实信息）；retain 崩；只在带约束时有效、无 hook 崩塌。
- **不与淘汰重复**：R2-A（RepSelect 式子空间）已 `REVISE`／新颖性不足；这里约束加在**反演得到的 Δ\*** 上、作为框架内硬约束，而非独立两段式方法、不训练新模块。近邻 AlphaEdit、RepSelect、FALCON。

## 4. Evidence Gate 与评审状态

| 阶段 | 状态 | 说明 |
|---|---|---|
| research-lit | done（composed） | WebSearch 3 组；近邻：activation steering / CAA / ActAdd、ACSD(2609.38342)、CUSE(2610.14763)、OPD 综述(2604.00626)、DistIL/DAgger、UDS(2605.24614)；gemini / arxiv helper 不可用，优雅跳过；文献身份未做 D1 三层核验，不作 BibTeX |
| idea-creator | done | 6 个候选，评分口径为记忆冻结口径，全部 S>8 |
| novelty-check | PROVISIONAL | 执行者划界，非跨模型独立评审；无 accept 回执 |
| research-review | REVIEW_UNAVAILABLE | codex/copilot/claude/gemini/gemini-cli/knot-cli 全 MISSING，无 `.aris/reviewer.json` |
| research-refine-pipeline | 未生成 | 用户诉求为思路评估 + 构思，未要求 top-1 proposal；且证据门 BLOCKED |
| evidence gate | BLOCKED | `run_state.py` / `idea_discovery_gate.py` 解析链未找到 + 缺 2 份跨模型回执 |

未训练、未跑 GPU pilot、未部署、未改 configs/ 与 requirements、未重训 baseline、未读取 .env 值、未打印或写入任何凭据、未创建 credential helper、未写 git 配置、未访问 `../../lrm-unlearning/`。未伪造 verdict-id、reviewer 模型名或 accept 回执；执行者判断未替代 reviewer / integrity 裁决。

## 5. 进入 /experiment-bridge 前仍需解除的阻塞项

- 提供 `../INIT_REPORT.md`（此前勘验缺失）：Frozen Protocol / active_line_protocol / blockers / heavy_data_root 不得凭空补全
- 解除「全局 init BLOCKED」与「dry-run FAILED」，给出进入实验阶段的明确许可
- 启用至少一个异族 reviewer 后端并取得真实身份回执；恢复 RESUMABLE helper 以执行确定性 gate
- 裁决基线 §2.2.1 方法身份张力 (A)/(B)：R1/R5/R6 走"A 的修正版"（Δ* 进主梯度路径），与用户此前可能倾向的 B 不同
- 冻结三次前向的样本与 tokenizer offset、`c/δ/ε`、投影秩、重估周期、patch 层集
- 确认两套指标口径隔离（官方 TOFU 8 指标 vs U1–U7 / C1–C11 gate），不混用不换算
- 所有依赖 pilot 的判别实验标 `needs manual pilot`；新结果须过 /experiment-audit + /result-to-claim

## 6. 来源

- 基线：`workspace/idea-stage/方向I_基线_v0.6.1_answer_span.md`（§1、§2.2、§2.2.1、§4.1、§5、§6）
- 候选集：`workspace/idea-stage/IDEA_CANDIDATES.md`（D1–D10，D3/D4/D7/D9/D10 为本文机制参照）
- 评分口径：跨会话记忆卡（冻结 rubric S=0.30I+0.25M+0.20T+0.15N+0.10C）
- 历史约束：记忆卡 mem_b42a20_c84bcae（E0 失败、do_not_start_e1）、mem_b42a20_246e24f（prompt-only v0.3、Zhang & Nanda ICLR 2024 OOD 伪峰）
- 文献（WebSearch 摘要，身份未独立核验，不作 BibTeX）：2605.24614、2605.27083、2610.14763、2609.38342、2604.00626、2509.22067、2506.07022；CAA/ActAdd/RepE；ROME causal tracing
