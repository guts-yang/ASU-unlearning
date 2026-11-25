# Research Contract — V4 自动实验验证后选定的主线

> 来源：`workspace/idea-stage/IDEA_REPORT_V4.md`（V1/V2/V3 报告 + 本轮 H20 实测 + research-wiki 反重复记忆）。
> 状态：gate 已完成；主线 idea 已按实测结果重排。跨模型评审仍不可用，裁决为 provisional。
> **2026-11-25 结构规整更新**：本报告仍以 V4 决策为准；候选 proposal 现在各自独立维护于 `../candidates/`，历史报告与旧基线归档于 `../archive/`，方向 I 权威基线为 `../方向I_基线_v0.6.1_answer_span.md`，完整整合见 `../IDEA_REPORT.md`。本契约是 W1 → W1.5 的会话恢复文件（#2）。

## Selected Ideas（按自动决策排序）

1. **R2-B / τ(k) 课程** — PROMOTE（低成本增量主线） → `../candidates/R2-B_ASU温度课程_tau_k.md`
   - 60-step 等预算：`low2high` dF=+0.949 优于 `global(τ=2)` dF=+0.800，retain 未受损（dR=-0.256）。
   - 优化后方法：以 `τ: 1.5→3` 线性课程为默认 ASU forget-teacher 调度；`high2low` 仅作消融。
2. **R2-C / 抗恢复即目标** — PROMOTE → `../candidates/R2-C_抗恢复即目标_relearning_aware_meta.md`
   - Gate：深度审计 ASU-u 的 P2 恢复率 `0.524`、P3 恢复率 `0.371`、`mean_recovery=0.448`，确认浅层遗忘。
   - 优化后方法：ASU 目标 + 内步 P3 模拟攻击 + 外步最大化恢复后 forget NLL；与 R3-2 共享攻击阶梯。
3. **R3-2 / ASU 遗忘深度审计基准** — PROMOTE → `../candidates/R3-2_ASU遗忘深度审计基准.md`
   - 对 base / ASU-u / GradAscent 统一跑 P1/P2/P3/INT8，输出恢复曲线下面积与 factual/function token 级报告。
4. **R2-A / ASU-Selective** — REVISE（机制通过、新颖性不足） → `../candidates/R2-A_ASU_Selective_表示选择性.md`
   - Gate：bottom-PC 更 forget-specific（mean_selectivity=0.531），但 R3 查新显示 RepSelect/ADU/ALMPU 已密集占据该空间。
   - 仅在能证明 "ASU 输出遗忘 × attention K/V 子空间" 耦合区别于上述工作时重启，否则不投入主线。
5. **R3-1 / SensiTemp** — REVISE → `../candidates/R3-1_SensiTemp_层头自适应温度场.md`
   - 敏感度非均匀（layer CV=1.11，head CV=9.22），但 sens 场 dF=+1.747 未超过等平均 τ 全局对照 +1.792（打乱场 +1.517）；收益来自更强总平滑而非结构化分配。降级为诊断分析。
6. **R3-3 / 浅-深解耦** — REVISE → `../candidates/R3-3_浅深解耦双阶段遗忘.md`
   - L12 suppress dF=+0.133 / dR=+0.099 仅边缘信号；严格阈值通过数=0，与 V2 抑制失败模式一致。仅保留为消融。
7. **V1 全系 / 反事实配对 + ΔA 正交对齐** — ELIMINATED
   - stage0-14 H20 实测已证伪；写入反重复记忆，禁止复现。归档说明见 `../archive/README.md`。

## Core Claims（更新后）
- C1：ASU 的输出层遗忘是浅层的，P2/P3 可恢复 ≥30% 的 forget-NLL 增益。
- C2：`low2high` τ 课程在等步数下优于固定 τ=2，且 retain 不退化。
- C3：relearning-aware 外步目标能进一步降低 P3 恢复率。
- C4（条件）：forget-gradient bottom-PC 选择性只有在与 RepSelect/ADU/ALMPU 明确划界后才可作为贡献。

## Minimum Convincing Evidence
- E1：R3-2 已给出 P1/P2/P3/INT8 全套数字（本轮完成）。
- E2：R2-B 需 3 seed 重复确认 low2high > global（本轮为 seed 0）。
- E3：R2-C 需实现内步攻击并报告恢复曲线，不能只用作事后审计替代。
- E4：R2-A 若重启，必须先给出与 RepSelect/ADU/ALMPU 的机制划界实验。
- E5：所有新结果必须过 `/experiment-audit` 与 `/result-to-claim`。

## Next-step Pointer
`/experiment-bridge` 以 `workspace/auto_experiments/` 为代码基线，先实现 R2-C 对抗目标与 R2-B 3-seed 验证；`workspace/auto_experiments/results/` 为 gate 证据源。

## 口径提醒（2026-11-25 追加）
方向 I 权威基线 `../方向I_基线_v0.6.1_answer_span.md`（去 τ + 反事实对照 + 答案跨度中性约束）与上述 6 个 ASU 温度线候选**机制轴不同**；且该基线所属的反事实配对路线在 V4 中被整批 ELIMINATED。两者关系与待裁决事项见 `../IDEA_REPORT.md` §5.1，本契约不自行判定基线有效性。会话恢复时：ACTIVE 候选从本契约的 Selected Ideas 读取，方向 I 基线单独读取，不混淆。

## Review Status
novelty-check = PROVISIONAL；research-review = REVIEW_UNAVAILABLE；无跨模型回执，不声称 accepted。
