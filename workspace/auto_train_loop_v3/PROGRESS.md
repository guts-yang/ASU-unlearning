# PROGRESS — 自主训练闭环 v3（复现指标驱动）

**run_id**: `auto-train-v3-20261123`  **next_round**: 3  **phase**: `terminated`
**best**: 0.20473700374568476 (idea_R2_alpha_hinge_4773d71b0720)
**llm_backend**: `REVIEW_UNAVAILABLE` — 确定性规则引擎，非 LLM
**冻结目标（全部基于 3-seed 复现）**: rho_mean<=0.38, rho_std<=0.05, 所有 seed |dF-1.1416|<=0.15, dR<=-0.2, refusal<=0.1
**已复现 idea**: 2  **TARGET_MET**: True

| round | lever | val_bpb | rho_mean | rho_std | worst_df | worst_dR | worst_ref | replicated | good | status |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | baseline | 0.498423 | 0.4170 | 0.0000 | 0.0543 | -0.2960 | 0.031 | True | False | keep |
| 2 | alpha_hinge | 0.204737 | 0.1774 | 0.0000 | 0.0049 | -0.2542 | 0.031 | True | True | keep |

## 最后一轮各判据

- rho_mean_le_target: PASS
- rho_std_le_max: PASS
- all_df_in_band: PASS
- all_dr_le_max: PASS
- all_generation_ok: PASS
- replicated: PASS
- rho_mean 差距: -0.2026
- rho_std 差距: -0.0500

## 硬禁止

counterfactual, gradient_surgery, projection, tau_field, tau_schedule

## 预算

- rounds 2/5, gpu 462/1800s, crash 0
