# Research Contract — 反事实配对的逐层正交遗忘（Idea 1）

> 聚焦交接契约：W1(idea) → W1.5(experiment-bridge)。来源：`workspace/idea-stage/IDEA_REPORT.md`。
> 状态：plan-only；本轮无实测、无跨模型评审回执（见报告 §External Critical Review）。

## Selected Idea
**Idea 1 — Δ-Gated Progressive Orthogonal Alignment**。用关系匹配 donor 得到配对差 ΔA_ℓ，估计"事实相关方向"，把 Student 在 X⁺ 的层激活沿该方向正交抑制、逐层渐进内化；不教模型假事实（区别于 CFT/CiPO），不做绝对激活对齐（区别于 PACT）。

**Selection rationale**：三项独立文献机制支撑（FALCON 正交投影、AlphaEdit 零空间、FDD 分层稀疏），且正面规避 CFT 的幻觉溢出/梯度冲突与 PACT 的 context blindness 两大负面证据。

## Core Claims
- C1：ΔA_ℓ 主成分主要编码目标事实方向而非主体身份（**待 probe 证实，是存亡假设**）。
- C2：沿 ΔA 正交抑制能移除目标事实，同时保留主体表征与通用能力。
- C3：渐进层 schedule 在等预算下优于静态全层/反向/随机 schedule。
- C4：效应参数内化于 Student，不依赖运行时 hook，且抗 relearning ≥2 类证据。

## Minimum Convincing Evidence
- E1（gate）：ΔA probe——线性 probe 在 ΔA 上判别目标事实 vs 主体身份，事实方向显著（kill switch：主体身份占主导则收缩/否决）。
- E2（gate）：等预算 schedule 对照——渐进 vs 静态全层 vs 反向 vs 随机，在 forget/retain 权衡上渐进有统计显著优势（否则降级为"多层对齐"表述，Claim C3 撤销）。
- E3：配对特异性——真实 donor 优于 noise/shuffled/wrong-donor/matched-retain + 随机层/位置对照。
- E4：非答案替换——同时报告原事实、donor 答案、其他错答、拒答、长度、EOS；donor 答案上升不算遗忘。
- E5：参数内化 + 抗恢复——无 hook Student 仍有效；relearning 曲线、probe/patching、量化、relation-swap probe 中 ≥2 类改善。
- E6：retain/shared facts 与 MMLU/GSM8K 无显著退化；加 context 敏感性探针防语义漂移。

## Next-step Pointer
`/experiment-bridge` 依 `workspace/refine-logs/EXPERIMENT_PLAN.md` 实现 Idea 1，先跑 E1/E2 两个 gate 实验；`/result-to-claim` 判定 C1/C3，`/ablation-planner` 安排 schedule 消融。

## Review Status
novelty-check = PROCEED WITH CAUTION (provisional, 无跨模型身份)；research-review = REVIEW_UNAVAILABLE；evidence gate 预期 BLOCKED，直到启用真实跨模型评审后端。
