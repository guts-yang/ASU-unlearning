# Non-Claim Diagnostic Report

- claim_level: `false`
- evidence_gate: `NOT_APPLIED`
- updated_at: `2026-11-26T12:58:43Z`
- baseline: ASU/open-unlearning TOFU `forget10/retain90/holdout10`, `Llama-3.2-1B-Instruct`

## Method Results

| # | Method | Mode | Train/Eval | Verdict | Key result |
|---:|---|---|---|---|---|
| 0 | sanity | 5-step global tau | yes/yes | PASSED | forget 0.5025→2.0562; retain 0.4651→1.4441 |
| 1 | CF-01 约束反演目标 | operator review + adjacent ablation | no/no | NOT_IMPLEMENTABLE | 约束反演目标与 patch 因果筛选算子未实现；相邻解耦消融未见 retain-safe 收益。 |
| 2 | CF-03 因果门控 activation gate | 20-step sens vs sens_shuffled | yes/yes | NEGATIVE | 敏感度门控臂未在 forget 与 retain 上同时优于打乱对照，首轮诊断不支持独立收益。 |
| 3 | CF-02 三前向角色分解 | operator review | no/no | NOT_IMPLEMENTABLE | 三前向角色分解、retain floor 与 donor 负约束未实现。 |
| 4 | CF-04 事实身份双探针解耦 | existing pooled probe | no/no | INCONCLUSIVE | pooled ΔA 的 retain-orthogonal 检查通过，但缺少事实/身份双探针 AUC。 |
| 5 | CF-06 LRM 推理轨迹 deltaA | track review | no/no | NOT_APPLICABLE | 需要 LRM/CoT 轨迹数据与模型，冻结 LLM TOFU-1B 基线不适用。 |
| 6 | CF-05 多 donor 边际化 | operator review | no/no | NOT_IMPLEMENTABLE | K-donor 堆叠/SVD 共有子空间与簇隔离未实现。 |
| 7 | CF-08 零空间抗恢复投影 | existing depth audit | no/no | INCONCLUSIVE | ASU 深度审计显示浅层遗忘，但投影模型缺完整 P3/INT8 恢复曲线。 |
| 8 | CF-07 读出空间 deltaA readout | operator review | no/no | NOT_IMPLEMENTABLE | 读出空间载体与 Jacobian 数值方案未预注册/未实现。 |
| 9 | CF-09 结构化对照族残差化 | adjacent ablation review | no/no | NOT_IMPLEMENTABLE | 结构化噪声基与残差回归算子未实现；相邻抑制/正交对齐臂全部未过门。 |
| 10 | CF-10 纯度驱动渐进 schedule | 20-step high2low vs global | yes/yes | PENDING | — |
| 11 | R2-C 抗恢复即目标 | operator review + audit | no/no | PENDING | — |
| 12 | R3-2 ASU 遗忘深度审计基准 | existing P1/P2/P3/INT8 evaluation | no/yes | PENDING | — |
| 13 | R3-3 浅深解耦双阶段遗忘 | existing V4/stage9 evaluation | no/no | PENDING | — |

## Constraint
This report is diagnostic only. It does not invoke the ARIS evidence gate, does not record accepted/provisional, and must not be cited as reproduction, novelty, or claim evidence.
