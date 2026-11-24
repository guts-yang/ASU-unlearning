# PROGRESS — 自主训练闭环

**run_id**: `auto-train-20261123`  **next_round**: 9  **phase**: `terminated`
**best**: 0.4206 (idea_R1_baseline_d8f03a3b50d0)
**llm_backend**: `REVIEW_UNAVAILABLE` — 思路由确定性规则引擎生成，非 LLM
**终止门控（冻结）**: rho<=0.3, dF>=1.0, dR<=-0.2, |dF-1.1416|<=0.15, 3-seed std<=0.05, refusal<=0.1
**generation_gate**: NOT_IMPLEMENTED → `TARGET_MET` 结构性不可达，预算耗尽须如实报告未达标

| round | lever | idea | val_bpb | rho | dF | dR | status |
|---|---|---|---|---|---|---|---|
| 1 | baseline | `idea_R1_baseline_d8f03a3b50d0` | 0.420600 | 0.4206 | +1.1416 | -0.3013 | keep |
| 2 | hinge_beta | `idea_R2_hinge_beta_d83ece9abcc0` | 1.035594 | 0.4178 | +1.5401 | -0.2984 | discard |
| 3 | hinge_beta | `idea_R3_hinge_beta_c5d64d266a0f` | 1.422966 | 0.4085 | +1.8046 | -0.3001 | discard |
| 4 | hinge_beta | `idea_R4_hinge_beta_35f67f0986a7` | 2.220063 | 0.3820 | +2.3537 | -0.2971 | discard |
| 5 | hinge_beta | `idea_R5_hinge_beta_fe1947004426` | 2.689962 | 0.3400 | +2.6949 | -0.2928 | discard |
| 6 | hinge_margin | `idea_R6_hinge_margin_4032da3e548f` | 1.388231 | 0.4040 | +1.7844 | -0.2992 | discard |
| 7 | hinge_margin | `idea_R7_hinge_margin_b366f50bd71a` | 1.753841 | 0.4391 | +2.0048 | -0.2944 | discard |
| 8 | hinge_margin | `idea_R8_hinge_margin_aeb2b32bc1f7` | 2.258359 | 0.4695 | +2.3209 | -0.2848 | discard |

## 与目标的距离（最后一轮）

- rho_le_target: FAIL
- df_ge_min: PASS
- dr_le_max: PASS
- df_in_band: FAIL
- seed_confirmed: FAIL
- generation_gate: NOT_IMPLEMENTED
- rho 差距: +0.1695（需 <= 0.3）

## 已证伪 / 硬禁止（反重复记忆）

counterfactual, gradient_surgery, projection, tau_field, tau_schedule

## 预算

- rounds 8/8, gpu 508/900s, consecutive_crash 0
