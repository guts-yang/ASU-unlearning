---
id: R3-1
title: SensiTemp —— 层-头自适应温度场（降级为诊断分析）
status: CANDIDATE · REVISE（收益来自更强总平滑，非结构化分配）
decision_source: workspace/idea-stage/IDEA_REPORT_V4.md（REVISE）、workspace/idea-stage/docs/research_contract.md
domain: LLM Unlearning / forget-teacher 温度分配 / ASU 机制诊断
evidence_level: provisional（等预算对照实测；无跨模型评审）
baseline_relation: 是 R2-B（时间维度 τ 课程）被否决的"结构维度"对照面
---

# R3-1 · SensiTemp（REVISE 候选）

> **状态提醒**：按"未淘汰作为候选"保留，但 V4 已把本方向**降级为诊断分析**，不得声称"温度场带来帕累托改进"。research-wiki 中 `idea:sensitemp-adaptive-temperature-2026` 标记为 `archived / negative`。可发表的增量是 low2high τ 调度（见 R2-B），不是结构化温度场（`IDEA_REPORT_V4.md`、`research_contract.md`）。

## 0. 故事（30 秒讲稿）

> ASU 用一个标量 $\tau$ 平滑所有层所有 head，必然同时压掉与事实无关的语法/局部注意力，这是其温度脆弱与 utility 损失的根因。我们曾设想：按"该层该 head 的 factual-vs-function token 敏感度"分配 $\tau_{\ell,h}$，让 forget-teacher 只在真正承载事实的层/head 平滑，语法 head 保持 $\tau=1$——不训练任何新模块，只重设计 teacher 的温度分配，保持 ASU 单机制简洁性。实测给我们上了一课：敏感度场确实非均匀（layer CV=1.11，head CV=9.22），但 SensiTemp 场 $dF=+1.747$ **没有超过等平均 $\tau$ 全局对照 $+1.792$**，打乱场 $+1.517$。收益来自"更强总平滑"，不是"结构化分配"。于是这个方向的诚实定位从"提出温度场方法"收缩为"一个有价值的阴性诊断 + 把时间维度（R2-B）让到前台"。

## 1. 领域问题与思路来源

- ASU 论文：factual token 对 $\tau$ 远敏感于 function token（Fig.2b），故全局单一 $\tau$ 能"只伤事实不伤语法"；但论文自承**温度选择脆弱、需 binary search**，并把 "dynamic temperature adjustment / layer-specific interventions / selective unlearning" 列为 future work（来源：`IDEA_REPORT_V3.md` §查新发现 2）。
- V3 查新发现 1：head-level + 抗恢复已是红海（ADU 2608.23020 / ALMPU AAAI2026 / ATTENTION SWITCH 2611.08752 都训练新模块），真正的空白一度被判断在"ASU 自身的温度机制"。
- stage0 实测张力：last-token ΔA 逐层递增（L1 0.051 → L8 0.171 → L16 0.380），而 ASU 平滑浅层即有效——提出"事实差异主要在浅层还是深层"的科学问题（来源：`IDEA_REPORT_V3.md`）。

## 2. 核心假设（已被等平均对照否决）

**原 H**：按 factual/function 敏感度给每层每 head 分配 $\tau_{\ell,h}$，结构化温度场应优于标量 $\tau$（帕累托改进）。

**实测裁决**：`REVISE`。敏感度非均匀成立，但等平均 $\tau$ 全局对照 $dF=+1.792\ge$ SensiTemp $+1.747$（打乱场 $+1.517$）——**收益来自更强总平滑而非分配**（来源：`IDEA_REPORT_V4.md`、`research_contract.md`）。因此不能声称温度场有效，只能作为诊断分析保留，并把 `low2high` τ 调度（R2-B）作为可发表增量。

## 3. 具体实现（作为诊断协议保留）

1. **一次性诊断**：在 forget set 用 ASU Appendix N 的 factual/function token 标注协议，扫 $\tau\in\{1,1.5,2,2.5,3\}$，得敏感度矩阵 $S[\ell,h]=\Delta\text{NLL}_{\text{factual}}-\Delta\text{NLL}_{\text{function}}$。
2. **温度场**：$\tau_{\ell,h}=1+\alpha\cdot\text{norm}(S[\ell,h])$，只在 $S$ 高的层/head 平滑；冻结 teacher 做标准 ASU 自蒸馏。
3. **强制对照**：全局 $\tau$（原 ASU）、**等平均 $\tau$ 全局**（关键消融）、随机层/head 场、逐层场、逐 head 场、位置 core/cap 场。
4. **kill switch**：若 $S[\ell,h]$ 近似均匀（方差 < 阈值），温度场退化为全局 $\tau$ → 直接否决（1–2 GPU·h 诊断，先验尸检再训练）。

## 4. 当前证据与 REVISE 依据（provisional）

- V4 决策 `REVISE`：敏感度非均匀（layer CV=1.11，head CV=9.22），但结构化分配无净收益；降级为诊断分析。
- V3 曾评 `Strong Accept（待 probe 验证）` 五维机制分 Higher8/Faster8/Cheaper9/Stronger7/Broader7；V4 实测推翻其净收益，故以 V4 为准。
- 跨模型评审不可用，裁决 provisional；research-wiki 标记 archived/negative。

## 5. 差异化与失败模式

| 近邻 | 关系 |
|---|---|
| ADU / ALMPU / ATTENTION SWITCH | 都训练新模块；SensiTemp 不训练新模块、用功能视角（敏感度）而非位置视角分配 |
| ASU 标量 $\tau$ | SensiTemp 是结构维度；R2-B 是时间维度（后者胜出） |
| Lacuna "Budgeted Attention Teachers" | 非同行评审 hypothesis 页，按诚信口径**不作可引用 prior art**（来源：`IDEA_REPORT_V3.md` 声明） |

**失败模式**：已被等平均对照命中（收益非结构化）；敏感度矩阵近似均匀导致退化；用位置 core/cap 的非正式划分冒充功能依据。维持 REVISE，不放宽阈值。

## 6. 与方向 I 基线 v0.6.1 的关系

SensiTemp 属"保留并重新分配 $\tau$"的 ASU 线；方向 I 基线是"去掉 $\tau$、改用反事实对照 + 答案跨度中性约束 $\mathcal L_F$"的相反轴。两者正交。本候选最重要的复用价值是它的**等平均 $\tau$ 对照纪律**——方向 I 的"渐进 schedule 必须相对等预算静态全层/反向/随机证明独立收益"（基线 §2.3、§6.2）是同一类防自欺门槛，应保持一致。

## 7. 诚信声明与来源

plan-level 候选，无新实验；数字标注来源；跨模型评审不可用，不伪造回执。

- `workspace/idea-stage/IDEA_REPORT_V4.md`（REVISE 与 $dF$ 数据）、`IDEA_REPORT_V3.md`（Idea 1、查新发现 1/2、五维分）
- `workspace/idea-stage/docs/research_contract.md`（降级为诊断、low2high 为可发表增量）
- `research-wiki/ideas/sensitemp-adaptive-temperature-2026.md`（archived / negative）
- 文献：ASU arXiv:2603.01285、ADU arXiv:2608.23020、ALMPU DOI 10.1609/aaai.v40i23.39017、ATTENTION SWITCH arXiv:2611.08752。
