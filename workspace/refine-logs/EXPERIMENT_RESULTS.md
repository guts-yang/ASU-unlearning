# Initial Experiment Results

Date: 2026-11-26
Plan: refine-logs/EXPERIMENT_PLAN.md

## M0 Sanity
- PASSED: 5-step global tau; forget 0.5025→2.0562, retain 0.4651→1.4441.

## Candidate Screening

| Candidate | Mode | Train | Eval | Verdict | Summary |
|---|---|---|---|---|---|
| CF-01 | operator review + adjacent ablation | no | no | NOT_IMPLEMENTABLE | 约束反演目标与 patch 因果筛选算子未实现；相邻解耦消融未见 retain-safe 收益。 |

## Summary
- Diagnostic only; claim_level=false; evidence gate NOT_APPLIED.
- Not a faithful reproduction for NOT_IMPLEMENTABLE/INCONCLUSIVE methods.
