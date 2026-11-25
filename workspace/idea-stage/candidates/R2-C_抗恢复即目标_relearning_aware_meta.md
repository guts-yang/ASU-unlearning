---
id: R2-C
title: 抗恢复即目标 —— relearning-aware meta 目标（把恢复攻击前移进训练）
status: CANDIDATE · PROMOTE
decision_source: workspace/idea-stage/IDEA_REPORT_V4.md（PROMOTE）、workspace/idea-stage/docs/research_contract.md
domain: LLM Unlearning / 深度遗忘 / 对抗式（minimax）训练目标
evidence_level: provisional（深度审计实测；对抗目标本身未实现、未训练）
baseline_relation: 与方向 I 基线 v0.6.1 正交，但共享"抗恢复"关切（见 §7）
---

# R2-C · 抗恢复即目标（relearning-aware meta objective）

## 0. 故事（30 秒讲稿）

> 行为遗忘不等于知识擦除。我们对 ASU-unlearned 模型做了深度审计：P3 微调恢复率最高 $0.371$、平均恢复 $\approx0.448$——**近 $45\%$ 的遗忘增益能被 few-shot / 少量同域微调找回**（来源：`NARRATIVE_ASU_TEMPERATURE_DEPTH.md`、`IDEA_REPORT_V4.md`）。已有工作都把"抗恢复"放在训练**后**当评估指标，发现浅层遗忘只能事后返工。我们的主张是：与其事后验收，不如把恢复攻击直接写进训练目标——内层攻击者用固定预算的良性重学/激活转向试图找回 $y^+$，外层 Student 在遗忘的同时躲避攻击者。只有对固定预算恢复攻击鲁棒的表示才能存活，浅层抑制会被内层攻击者找回并惩罚。这样就直接优化"深度"而非事后检测。

## 1. 领域问题与空白

LLM unlearning 的主流评估只看 forget/retain NLL，会系统性漏掉 **shallow forgetting**：模型只是学会了"不读上下文/提前停答/改答"，知识仍在参数里。PrivUn（Findings EMNLP 2026, arXiv:2604.22076）、SoK dememorization（2026-09）、RepSelect（arXiv:2606.17168）、ALMPU 都指出：只看 forget/retain NLL 不够，必须测 P1/P2/P3 三级攻击与量化/probe 恢复（来源：`IDEA_REPORT_V2.md`、`IDEA_REPORT_V3.md` §查新发现 3）。

**空白**：这些工作要么把抗恢复当**评估维度**，要么把它当所提方法的**附加实验**；TAR 式 tamper-resistant 目标虽有，但未与 ASU 的 forget-teacher 自蒸馏耦合，也未把"恢复攻击阶梯"同时用作训练内攻击与留出验收。用户全部既有实验（stage0-14）从未测抗恢复，这既是风险也是最低成本的贡献机会。

## 2. 核心假设（可证伪）

**H**：把"抗固定预算恢复攻击"前移为训练目标（minimax），能显著降低留出恢复攻击下的回退幅度，且不牺牲 retain。浅层抑制（提前 EOS、改答、上下文盲）会在内层攻击者下暴露并被外层惩罚，因而被优化掉。

**排除的替代解释**：若只是"针对训练内攻击者作弊"（在参与闭环的攻击上变好、留出攻击不变），不算深度遗忘——这正是 C4/RecoveryCertificate 类设计的硬靶心。

## 3. 具体实现

**minimax 目标**（$\theta$ = Student，内层攻击 $\Delta\theta$ 受预算 $\eta$ 约束）：

$$
\min_\theta\ \mathcal L_{\text{ASU}}(\theta)\;+\;\lambda\,\mathbb E_{x}\Big[\max_{\|\Delta\theta\|\le\eta}\ \operatorname{score}\big(y_+\mid \theta+\Delta\theta(x)\big)-\operatorname{score}\big(y_+\mid\theta\big)\Big]
$$

- $\mathcal L_{\text{ASU}}$：ASU forget-teacher 自蒸馏 KL（可与 R2-B 的 $\tau(k)$ 课程叠加）。
- $\operatorname{score}(y_+|\cdot)$：目标答案 $y^+$ 的长度归一化 span log-likelihood（越大=仍支持原事实）。
- 内层攻击类型（**训练前冻结并公开**）：benign relearning（少量同域数据微调）、activation steering、prompt reformulation；预算 $\eta$ 与攻击步数冻结。
- 双环训练：内步用当前 $\theta$ 模拟攻击者最大化恢复；外步用恢复后的 $\operatorname{score}$ 反传更新 $\theta$。
- 实用近似：与 G / TrustRegion 共用 $D_{\text{anchor}}$ 上的 Fisher/K-FAC 曲率 $H$，Cauchy 步 $\delta=-\sqrt{\varepsilon/(g^\top H g+\rho)}\,g$ 约束步长，防 retain/长度崩塌。
- 与 R3-2 深度审计**共享同一套 P1/P2/P3/INT8 攻击阶梯**：训练内攻击只用其中公开的一员，**留出第三类攻击**做独立验收。

**代码落点**（`workspace/auto_experiments/` 为基线，按 `research_contract.md` Next-step Pointer）：新增双环 trainer、攻击集 A、留出攻击者；所有新结果写新 `runs/<run_id>/`，不覆盖源数据/缓存/评测。

## 4. 差异化（novelty 落点）

| 近邻 | 关系 | 划界 |
|---|---|---|
| TAR / tamper-resistant 目标 | 组件先例 | 本候选与 ASU forget-teacher 耦合，并用公开恢复阶梯做训练+留出双层 |
| SoK dememorization / PrivUn / RepSelect / ALMPU | 把抗恢复当评估或附加实验 | 本候选把恢复攻击**前移为训练目标** |
| UNDO / Distillation Robustifies Unlearning (NeurIPS 2025) | 部署后蒸馏过滤潜在知识 | 本候选在训练内优化，可作部署前增强的对照 |
| 方向 I 基线的 C4 / RecoveryCertificate 候选（L7 整合规格） | 同属"抗恢复闭环"家族 | 本候选绑定 ASU 温度线，不依赖 donor 配对；C4 的证书不得由训练内 probe 自证 |

**novelty 不能落在组件本身**（trust region / minimax / 对抗训练有大量先例），只能落在"与 ASU 自蒸馏 + 公开攻击阶梯 + 留出攻击验收"的结合方式，以及由此产出的"行为遗忘≠知识擦除"实证。

## 5. 当前证据与状态（provisional）

- 深度审计 PASS：ASU-u 的 P2 恢复率 $0.524$、P3 恢复率 $0.371$、$\text{mean\_recovery}=0.448$，确认浅层遗忘存在（来源：`research_contract.md`、`IDEA_REPORT_V4.md`）。
- V4 决策 `PROMOTE`：在 ASU 目标上加内步 P3 模拟攻击 + 外步最大化恢复后 forget NLL，作为独立贡献线，与 R3-2 共享攻击阶梯。
- **对抗目标本身尚未实现、未训练**。**未完成义务 E3**：需实现内步攻击并报告恢复曲线，不能只用作事后审计替代。
- 跨模型评审不可用，裁决 provisional，不声称 accepted。

## 6. 判别实验、失败模式、必测指标

**决定性判别**：有闭环 / 无闭环 / 训练内 probe 伪证书**三臂**，且**在留出的第三类攻击上测**。

**失败模式（命中即 NEGATIVE）**

- 只在参与闭环的攻击上变好、留出攻击不变 → 针对审计器作弊，不算深度遗忘；
- 攻击者太弱 → minimax 退化成普通遗忘；太强 → 1B 规模训练不稳；
- 攻击者微调用的同域数据也是 retain 分布 → 抗恢复与 retain 直接冲突，retain 崩塌；
- 遗忘只在带 hook / 带约束时出现，无 hook 崩塌。

**必测指标**：攻击前后 span score 差、攻击者成功率、恢复曲线下面积、retain NLL、长度/EOS/完整回答率、KL 球占用率（与 R2-B/TrustRegion 组合时）、3 seed std。统计单位作者/实体簇。

## 7. 与方向 I 基线 v0.6.1 的关系

- 机制轴不同：v0.6.1 去 $\tau$，用反事实对照 + 答案跨度中性约束 $\mathcal L_F$；本候选保留 ASU 温度线，把对抗恢复写进目标。
- **共享关切**：v0.6.1 §6.5 明确要求"至少两类独立抗恢复证据改善才允许受限『深度遗忘』表述"，且把 probe/patching/量化/受控 relearning 列为硬靶心。R2-C 提供的**公开恢复攻击阶梯与留出验收协议**可直接被方向 I 复用为 G5/G6 验收口径；但方向 I 的历史 pilot 不复用为 confirmation，两者证据链必须分开登记。

## 8. 诚信声明与来源

plan-level 候选，不含新实验；数字来自成文报告并标注来源；未核实处用"据不完全证据显示"。跨模型评审不可用，不伪造回执；新结果须过 `/experiment-audit` + `/result-to-claim`。

- `workspace/idea-stage/IDEA_REPORT_V4.md`、`IDEA_REPORT_V2.md` §Idea C、`IDEA_REPORT_V3.md` §查新发现 3
- `workspace/idea-stage/docs/research_contract.md`（E3 义务、共享攻击阶梯）
- `workspace/idea-stage/NARRATIVE_ASU_TEMPERATURE_DEPTH.md`（$44.8\%$ 恢复）
- `workspace/idea-stage/方向I_基线_v0.6.1_answer_span.md` §6.5
- 文献：PrivUn arXiv:2604.22076、SoK dememorization 2026-09、RepSelect arXiv:2606.17168、UNDO / Distillation Robustifies Unlearning NeurIPS 2025、TAR（tamper-resistant）。
