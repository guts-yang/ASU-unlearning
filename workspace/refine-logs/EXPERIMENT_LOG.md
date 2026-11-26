# Experiment Log

## R000 — 2026-11-26
- System: sanity global tau, 5 steps, seed 0
- Result: forget=2.0562, retain=1.4441
- Verdict: PASSED sanity

## CF-01 — 2026-11-26
- Priority: rank 1 / 8.0
- Verdict: NOT_IMPLEMENTABLE
- Summary: 约束反演目标与因果筛选算子未实现；相邻解耦消融未见收益
- Reproduce: see evidence paths in auto_experiments/results/screening/CF-01.json

## CF-03 — 2026-11-26
- Priority: rank 2 / 7.8
- Verdict: INCONCLUSIVE
- Summary: 敏感度门 PASS 且 idea1 优于 random/wrong，但行为指标未满足严格成功判据
- Reproduce: see evidence paths in auto_experiments/results/screening/CF-03.json

## CF-02 — 2026-11-26
- Priority: rank 3 / 7.6
- Verdict: NOT_IMPLEMENTABLE
- Summary: 三角色可行集与 donor 负约束未实现；相邻解耦消融无改善
- Reproduce: see evidence paths in auto_experiments/results/screening/CF-02.json

## CF-04 — 2026-11-26
- Priority: rank 4 / 7.4
- Verdict: INCONCLUSIVE
- Summary: pooled ΔA retain-orthogonal PASS，但缺事实/身份双探针 AUC
- Reproduce: see evidence paths in auto_experiments/results/screening/CF-04.json

## CF-06 — 2026-11-26
- Priority: rank 5 / 7.4
- Verdict: NOT_APPLICABLE
- Summary: 需 LRM/CoT 轨迹，冻结基线为 LLM TOFU-1B
- Reproduce: see evidence paths in auto_experiments/results/screening/CF-06.json

## CF-05 — 2026-11-26
- Priority: rank 6 / 7.4
- Verdict: NOT_IMPLEMENTABLE
- Summary: K-donor 堆叠/SVD 共有子空间未实现
- Reproduce: see evidence paths in auto_experiments/results/screening/CF-05.json

## CF-08 — 2026-11-26
- Priority: rank 7 / 7.4
- Verdict: INCONCLUSIVE
- Summary: 深度审计与 retain-orth PASS，但缺投影模型的完整恢复曲线
- Reproduce: see evidence paths in auto_experiments/results/screening/CF-08.json

## CF-07 — 2026-11-26
- Priority: rank 8 / 7.2
- Verdict: NOT_IMPLEMENTABLE
- Summary: 读出空间载体与 Jacobian 数值方案未预注册/未实现
- Reproduce: see evidence paths in auto_experiments/results/screening/CF-07.json

## CF-09 — 2026-11-26
- Priority: rank 9 / 7.2
- Verdict: NOT_IMPLEMENTABLE
- Summary: 结构化噪声基与残差回归算子未实现；相邻抑制/正交对齐门全部 false
- Reproduce: see evidence paths in auto_experiments/results/screening/CF-09.json

## CF-10 — 2026-11-26
- Priority: rank 10 / 7.2
- Verdict: NEGATIVE
- Summary: 纯度谱非平坦 PASS，但等预算 progressive 未优于 ref
- Reproduce: see evidence paths in auto_experiments/results/screening/CF-10.json

## R2-C — 2026-11-26
- Priority: PROMOTE
- Verdict: NOT_IMPLEMENTABLE
- Summary: 双环 trainer 与训练内攻击未实现；深度审计只证明浅层遗忘
- Reproduce: see evidence paths in auto_experiments/results/screening/R2-C.json

## R3-2 — 2026-11-26
- Priority: PROMOTE
- Verdict: POSITIVE_SCREEN
- Summary: P1/P2/P3/INT8 阶梯 PASS，方法间有分化
- Reproduce: see evidence paths in auto_experiments/results/screening/R3-2.json

## R3-3 — 2026-11-26
- Priority: REVISE / ablation only
- Verdict: NEGATIVE
- Summary: V4 严格阈值通过数 0，stage9 全体 pass_gate=false
- Reproduce: see evidence paths in auto_experiments/results/screening/R3-3.json
