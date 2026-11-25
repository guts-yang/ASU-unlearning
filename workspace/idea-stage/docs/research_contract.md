# Research Contract — V4 自动实验验证后选定的主线

> 来源：`workspace/idea-stage/IDEA_REPORT_V4.md`（V1/V2/V3 报告 + 本轮 H20 实测 + research-wiki 反重复记忆）。
> 状态：gate 已完成；主线 idea 已按实测结果重排。跨模型评审仍不可用，裁决为 provisional。
> **2026-11-25 结构规整更新**：本报告仍以 V4 决策为准；现役候选维护于 `../candidates/`（R2-C、R3-2、R3-3），已归档候选（R2-B、R2-A、R3-1）与历史报告、旧基线在 `../archive/`，方向 I 权威基线为 `../方向I_基线_v0.6.1_answer_span.md`，完整整合见 `../IDEA_REPORT.md`。本契约是 W1 → W1.5 的会话恢复文件（#2）。

## Selected Ideas（按自动决策排序）

1. **R2-B / τ(k) 课程** — **ARCHIVED · NEGATIVE（2026-11-25 3-seed 识别门 FAIL）** → `../archive/candidates/R2-B_ASU温度课程_tau_k.md`
   - 原 60-step seed 0 比较：`low2high` dF=+0.949 vs `global(τ=2)` +0.800；补跑等均值 τ=2.25 常数对照后，后者 3/3 seed 均更高（均值 1.857 vs 1.449）。
   - 结论：收益由平均平滑水平解释，时间课程无净收益；不再作为主线，保留为阴性诊断与反重复记忆。
2. **R2-C / 抗恢复即目标** — PROMOTE → `../candidates/R2-C_抗恢复即目标_relearning_aware_meta.md`
   - Gate：深度审计 ASU-u 的 P2 恢复率 `0.524`、P3 恢复率 `0.371`、`mean_recovery=0.448`，确认浅层遗忘。
   - 优化后方法：ASU 目标 + 内步 P3 模拟攻击 + 外步最大化恢复后 forget NLL；与 R3-2 共享攻击阶梯。
3. **R3-2 / ASU 遗忘深度审计基准** — PROMOTE → `../candidates/R3-2_ASU遗忘深度审计基准.md`
   - 对 base / ASU-u / GradAscent 统一跑 P1/P2/P3/INT8，输出恢复曲线下面积与 factual/function token 级报告。
4. **R2-A / ASU-Selective** — **ARCHIVED · negative（2026-11-25）** → `../archive/candidates/R2-A_ASU_Selective_表示选择性.md`
   - Gate：bottom-PC 更 forget-specific（mean_selectivity=0.531），但 R3 查新显示 RepSelect/ADU/ALMPU 已密集占据该空间；E4 机制划界未做。
   - 不投入主线，已归档。
5. **R3-1 / SensiTemp** — **ARCHIVED · negative（2026-11-25）** → `../archive/candidates/R3-1_SensiTemp_层头自适应温度场.md`
   - 敏感度非均匀（layer CV=1.11，head CV=9.22），但 sens 场 dF=+1.747 未超过等平均 τ 全局对照 +1.792（打乱场 +1.517）；收益来自更强总平滑而非结构化分配。
   - 已归档；其诊断价值与 R2-B 的失败模式同源。
6. **R3-3 / 浅-深解耦** — REVISE → `../candidates/R3-3_浅深解耦双阶段遗忘.md`
   - L12 suppress dF=+0.133 / dR=+0.099 仅边缘信号；严格阈值通过数=0，与 V2 抑制失败模式一致。仅保留为消融。
7. **V1 全系 / 反事实配对 + ΔA 正交对齐** — ELIMINATED
   - stage0-14 H20 实测已证伪；写入反重复记忆，禁止复现。归档说明见 `../archive/README.md`。

## Core Claims（更新后）
- C1：ASU 的输出层遗忘是浅层的，P2/P3 可恢复 ≥30% 的 forget-NLL 增益。
- C2（已否证，2026-11-25）：`low2high` τ 课程看似优于固定 τ=2，但控制平均 τ 后不成立——等均值 τ=2.25 常数在 3/3 seed 上优于 `low2high`（均值 dF 1.857 vs 1.449）；原比较混杂了平均平滑水平。R2-B 已归档。
- C3：relearning-aware 外步目标能进一步降低 P3 恢复率。
- C4（条件）：forget-gradient bottom-PC 选择性只有在与 RepSelect/ADU/ALMPU 明确划界后才可作为贡献。

## Minimum Convincing Evidence
- E1：R3-2 已给出 P1/P2/P3/INT8 全套数字（本轮完成）。
- E2：R2-B 需 3 seed 重复确认 low2high > global（本轮为 seed 0）。
- E3：R2-C 需实现内步攻击并报告恢复曲线，不能只用作事后审计替代。
- E4：R2-A 若重启，必须先给出与 RepSelect/ADU/ALMPU 的机制划界实验。
- E5：所有新结果必须过 `/experiment-audit` 与 `/result-to-claim`。

## Next-step Pointer
`/experiment-bridge` 以 `workspace/auto_experiments/` 为代码基线，先实现 R2-C 对抗目标、补 R3-2 的 Probe/恢复曲线；R2-B 已 3-seed 判负并归档，不再排入；`workspace/auto_experiments/results/` 为 gate 证据源。

## 口径提醒（2026-11-25 追加，裁决 2 后更新）
方向 I 权威基线 `../方向I_基线_v0.6.1_answer_span.md`（去 τ + 反事实对照 + 答案跨度中性约束）与上述 6 个 ASU 温度线候选**机制轴不同**；且该基线所属的反事实配对路线在 V4 中被整批 ELIMINATED。会话恢复时：ACTIVE 候选**仅**从本契约的 Selected Ideas 读取，方向 I 基线单独读取，不混淆。

**2026-11-25 用户裁决（2 · 归档）**：§5.1 口径张力已裁决完毕——整条反事实配对路线**正式归档**，`v0.6.1` 基线转为仅供存档的设计文档，不作为实现规格、不进入实验队列、不投入算力；§2.2.2 的 (A) 裁决记录与后续设计义务停止执行。V1 全系 `ELIMINATED` 与 `research-wiki/log.md` 反重复记忆**继续有效，不解除、不例外化**。裁决说明见 `../IDEA_REPORT.md` §5.1。

**候选 proposal 的阅读口径**：`../candidates/` 下 6 份 proposal 的 §6/§7 中凡引用「方向 I 基线 v0.6.1」「口径张力需用户裁决」「其 §6 验收口径可复用」等表述，均按裁决 2 理解为**历史参考**：方向 I 已归档，其验收口径不作为这 6 个候选的执行规格或正向证据；候选自身的判别实验、失败模式与指标以各文件正文为准。各文件正文保留不改。

## Review Status
novelty-check = PROVISIONAL；research-review = REVIEW_UNAVAILABLE；无跨模型回执，不声称 accepted。
