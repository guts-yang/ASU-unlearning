# Initial Experiment Results

Date: 2026-11-26
Plan: refine-logs/EXPERIMENT_PLAN.md

## M0 Sanity
- PASSED: 5-step global tau; forget 0.5025→2.0562, retain 0.4651→1.4441.

## Candidate Screening

| Candidate | Mode | Train | Eval | Verdict | Summary |
|---|---|---|---|---|---|
| CF-01 | operator review + adjacent ablation | no | no | NOT_IMPLEMENTABLE | 约束反演目标与 patch 因果筛选算子未实现；相邻解耦消融未见 retain-safe 收益。 |
| CF-03 | 20-step sens vs sens_shuffled | yes | yes | NEGATIVE | 敏感度门控臂未在 forget 与 retain 上同时优于打乱对照，首轮诊断不支持独立收益。 |
| CF-02 | operator review | no | no | NOT_IMPLEMENTABLE | 三前向角色分解、retain floor 与 donor 负约束未实现。 |
| CF-04 | existing pooled probe | no | no | INCONCLUSIVE | pooled ΔA 的 retain-orthogonal 检查通过，但缺少事实/身份双探针 AUC。 |
| CF-06 | track review | no | no | NOT_APPLICABLE | 需要 LRM/CoT 轨迹数据与模型，冻结 LLM TOFU-1B 基线不适用。 |
| CF-05 | operator review | no | no | NOT_IMPLEMENTABLE | K-donor 堆叠/SVD 共有子空间与簇隔离未实现。 |
| CF-07 | operator review | no | no | NOT_IMPLEMENTABLE | 读出空间载体与 Jacobian 数值方案未预注册/未实现。 |

## Summary
- Diagnostic only; claim_level=false; evidence gate NOT_APPLIED.
- Not a faithful reproduction for NOT_IMPLEMENTABLE/INCONCLUSIVE methods.
