# Initial Experiment Results

Date: 2026-11-26
Plan: refine-logs/EXPERIMENT_PLAN.md

## M0 Sanity
- PASSED: 5-step global tau run completed; forget NLL 0.5025 → 2.0562, retain NLL 0.4651 → 1.4441.

## M1-M13 Candidate Screening

| Candidate | Priority | Verdict | Summary |
|---|---:|---|---|
| CF-01 | rank 1 / 8.0 | NOT_IMPLEMENTABLE | 约束反演目标与因果筛选算子未实现；相邻解耦消融未见收益 |
| CF-03 | rank 2 / 7.8 | INCONCLUSIVE | 敏感度门 PASS 且 idea1 优于 random/wrong，但行为指标未满足严格成功判据 |
| CF-02 | rank 3 / 7.6 | NOT_IMPLEMENTABLE | 三角色可行集与 donor 负约束未实现；相邻解耦消融无改善 |
| CF-04 | rank 4 / 7.4 | INCONCLUSIVE | pooled ΔA retain-orthogonal PASS，但缺事实/身份双探针 AUC |
| CF-06 | rank 5 / 7.4 | NOT_APPLICABLE | 需 LRM/CoT 轨迹，冻结基线为 LLM TOFU-1B |
| CF-05 | rank 6 / 7.4 | NOT_IMPLEMENTABLE | K-donor 堆叠/SVD 共有子空间未实现 |
| CF-08 | rank 7 / 7.4 | INCONCLUSIVE | 深度审计与 retain-orth PASS，但缺投影模型的完整恢复曲线 |
| CF-07 | rank 8 / 7.2 | NOT_IMPLEMENTABLE | 读出空间载体与 Jacobian 数值方案未预注册/未实现 |
| CF-09 | rank 9 / 7.2 | NOT_IMPLEMENTABLE | 结构化噪声基与残差回归算子未实现；相邻抑制/正交对齐门全部 false |
| CF-10 | rank 10 / 7.2 | NEGATIVE | 纯度谱非平坦 PASS，但等预算 progressive 未优于 ref |
| R2-C | PROMOTE | NOT_IMPLEMENTABLE | 双环 trainer 与训练内攻击未实现；深度审计只证明浅层遗忘 |
| R3-2 | PROMOTE | POSITIVE_SCREEN | P1/P2/P3/INT8 阶梯 PASS，方法间有分化 |
| R3-3 | REVISE / ablation only | NEGATIVE | V4 严格阈值通过数 0，stage9 全体 pass_gate=false |

## Summary
- 1 POSITIVE_SCREEN (R3-2), 3 INCONCLUSIVE (CF-03, CF-04, CF-08), 6 NOT_IMPLEMENTABLE + 1 NOT_APPLICABLE (CF-06), 2 NEGATIVE (CF-10, R3-3).
- All source candidates are plan-level; no result is a faithful reproduction or paper-ready claim.
- Ready for /auto-review-loop: only after implementing missing operators for positive/inconclusive candidates.
