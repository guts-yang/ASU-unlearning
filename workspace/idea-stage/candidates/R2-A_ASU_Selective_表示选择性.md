---
id: R2-A
title: ASU-Selective —— 注意力平滑 × forget-specific 表示选择性
status: CANDIDATE · REVISE（机制通过 / 新颖性不足，条件性重启）
decision_source: workspace/idea-stage/IDEA_REPORT_V4.md（REVISE）、workspace/idea-stage/docs/research_contract.md
domain: LLM Unlearning / 表示子空间选择性 / ASU 两段式增强
evidence_level: provisional（机制 gate PASS；差异化未证实，无跨模型评审）
baseline_relation: 与方向 I 基线 v0.6.1 正交（见 §7）
---

# R2-A · ASU-Selective（REVISE 候选）

> **状态提醒**：本文件按用户"未淘汰的作为候选"要求保留为候选，但**不是主线**。V4 决策为 `REVISE`：机制 gate 通过，但查新显示 RepSelect/ADU/ALMPU 已密集占据该空间，差异化不足；只有在特定耦合能被明确证伪区分时才重启，否则不投入主线（来源：`IDEA_REPORT_V4.md`、`research_contract.md`）。research-wiki 中 `idea:asu-selective-repselect-2026` 标记为 `archived / negative`。

## 0. 故事（30 秒讲稿）

> 方向 I 的反事实配对被实测证伪后，我们转向一个更朴素的问题：真正决定"深度遗忘"的，是不是**表示子空间的选择性**？RepSelect 证明 forget 梯度 SVD 里 top-PC 是与 retain 共享的高方差方向（攻击可恢复、且伤通用能力），**bottom-PC 才是 forget-specific**。于是设想两段式：阶段 1 用已验证有效的 ASU 温度自蒸馏做输出层遗忘；阶段 2 只压缩 forget-specific 低方差子空间，保留共享 top-PC，把浅层遗忘变深。诚实地说，这个故事目前**讲不圆**：RepSelect/ADU/ALMPU 已把 head 级 + 子空间 + 抗恢复空间占满，我们唯一可能站得住的差异化是"ASU 输出遗忘 × attention K/V 子空间"的耦合——而这一点尚未被实验证明。因此它是一份带重启门槛的候选，不是推荐方向。

## 1. 领域问题与思路来源

V2 实测复盘（H20 / TOFU-1B / 3 seed）给出类级结论："反事实配对构造 forget 目标"+"表示层正交/对齐"+"梯度手术"三条路线系统性失败，且都弱于原 ASU（`IDEA_REPORT_V2.md`）。新查新（RepSelect arXiv:2606.17168）给出转向依据：SVD 分解 forget 梯度，**top-PC 与 retain/攻击共享，bottom-PC 才是 forget-specific**；只压缩 bottom-PC，抗 relearning 提升 4–50×，跨 Llama3/Qwen3.5/Gemma4/DeepSeek-V2；局限：仅 MLP 模块、未测 attention K/V（来源：`IDEA_REPORT_V2.md` 查新表）。

**思路**：把"输出行为遗忘（ASU）"与"内部表示深度（子空间选择性）"视作正交互补的两段，而非再加一个新模块。

## 2. 核心假设（可证伪）

**H**：ASU 温度自蒸馏已在输出层实现有效遗忘（实测 ref 最优）但属浅层；在 forget-specific bottom-PC 子空间做表示约束可把遗忘变深，同时不碰与 retain 共享的 top-PC，从而不伤通用能力。

**重启前提（硬门槛）**：必须先证明 "ASU 输出遗忘 × attention K/V 子空间" 的耦合区别于 RepSelect（仅 MLP 权重梯度 SVD）/ ADU（attention 路径 adapter）/ ALMPU（head 扰动）。若做不出差异，须诚实标为"RepSelect 在 ASU 上的复现/迁移"，不构成新贡献。

## 3. 具体实现

- **阶段 1**：用现成 ASU 得到 unlearned 模型 $M_u$。
- **阶段 2**：在 $M_u$ 上对 forget set 做 SVD（分别在 MLP 与 attention K/V），把权重梯度/激活更新限制在 bottom-PC 子空间（或等价：对 top-PC 方向施加 retain-anchor 约束），得到 $M_u'$。
- **评估**：P1/P2/P3/INT8 恢复阶梯（复用 R3-2 协议）+ MMLU / retain / 抗恢复。
- **前置 gate（kill risk）**：若 $M_u$ 的 forget 梯度 SVD 中 bottom-PC 与 ASU 平滑后的注意力变化正交（两者作用于不相交子空间），叠加可能无收益 → 先做**子空间重叠诊断**（类似已有 stage7b，但对象是 forget 梯度而非 ΔA）。

## 4. 当前证据与 REVISE 依据（provisional）

- 机制 gate `PASS`：ASU-Mu 上 `v_proj/down_proj` forget-grad SVD，$\text{mean\_selectivity}=0.531$（top-PC 与 retain 梯度重合显著高于 bottom-PC）（来源：`IDEA_REPORT_V4.md`）。
- 决策 `REVISE`：R3 查新显示 RepSelect/ADU/ALMPU 已密集占据该空间，差异化不足；仅在能证明耦合区别时重启，否则不投入主线。
- 未完成义务 `E4`：若重启，必须先给出与 RepSelect/ADU/ALMPU 的机制划界实验。
- 跨模型评审不可用，`PROCEED WITH CAUTION` 为执行者 provisional，不记 cross-family accepted。

## 5. 差异化与失败模式

| 近邻 | 占据点 | 本候选唯一可能差异 |
|---|---|---|
| RepSelect (arXiv:2606.17168) | forget 梯度 bottom-PC、仅 MLP | 与 ASU 温度线耦合；首次检验 attention K/V 上的选择性（RepSelect 自承留白） |
| ADU (arXiv:2608.23020) | attention 路径级遗忘、adapter | 不训练新模块，两段式叠加 |
| ALMPU (AAAI 2026) | head 敏感定位 + 抗恢复扰动 | 子空间选择性而非 head 扰动 |

**失败模式**：bottom-PC 与 ASU 注意力变化正交（叠加无收益）；与 RepSelect 字面高度重合被判变体；压缩误伤 retain；只在 hook 下有效。命中即维持 REVISE 或归档，不放宽阈值。

## 6. 与方向 I 基线 v0.6.1 的关系

机制轴不同：方向 I 去 $\tau$、用反事实对照 $X^-/X^+$ 与答案跨度 $S_f$ 上的中性约束 $\mathcal L_F$；本候选保留 ASU 温度线、在权重/激活 SVD 子空间操作，不依赖 donor 配对。两者共享"深度遗忘"关切，可共用 R3-2 恢复阶梯做验收，但证据链分开登记。

## 7. 诚信声明与来源

plan-level 候选，无新实验；数字标注来源；跨模型评审不可用，不伪造回执；新结果须过 `/experiment-audit` + `/result-to-claim`。

- `workspace/idea-stage/IDEA_REPORT_V4.md`（决策与 mean_selectivity）、`IDEA_REPORT_V2.md` §Idea A 与查新、`IDEA_REPORT_V3.md` §淘汰清单
- `workspace/idea-stage/docs/research_contract.md`（E4、C4 条件性贡献）
- `research-wiki/ideas/asu-selective-repselect-2026.md`（archived / negative 反重复记忆）
- 文献：RepSelect arXiv:2606.17168、SAGO arXiv:2604.14808 (ACL 2026)、ADU arXiv:2608.23020、ALMPU DOI 10.1609/aaai.v40i23.39017、PrivUn arXiv:2604.22076、ASU arXiv:2603.01285。
