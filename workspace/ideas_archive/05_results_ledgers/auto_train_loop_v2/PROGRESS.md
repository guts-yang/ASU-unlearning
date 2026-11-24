# PROGRESS — 自主训练闭环 v2

**run_id**: `auto-train-v2-20261123`  **next_round**: 11  **phase**: `terminated`
**best**: 0.204737 (idea_R4_alpha_hinge_4773d71b0720)
**llm_backend**: `REVIEW_UNAVAILABLE` — 确定性规则引擎，非 LLM
**冻结目标**: rho<=0.38, |dF-1.1416|<=0.15, dR<=-0.2, refusal<=0.1, 3-seed std(rho)<=0.05
**v2 修复**: generation_gate 与 3-seed 确认轮已实跑，TARGET_MET 不再结构性不可达

| round | lever | idea | val_bpb | rho | dF | dR | refusal | status |
|---|---|---|---|---|---|---|---|---|
| 1 | baseline | `idea_R1_baseline_d8f03a3b50d0` | 0.498423 | 0.4170 | +1.1959 | -0.2960 | 0.031 | keep |
| 2 | alpha_hinge | `idea_R2_alpha_hinge_b3f0a2fec7ac` | 1.123706 | 0.2198 | +0.5523 | -0.2651 | 0.000 | discard |
| 3 | alpha_hinge | `idea_R3_alpha_hinge_37d5b766bd34` | 0.668043 | 0.1952 | +0.8397 | -0.2553 | 0.031 | discard |
| 4 | alpha_hinge | `idea_R4_alpha_hinge_4773d71b0720` | 0.204737 | 0.1774 | +1.1367 | -0.2542 | 0.031 | keep |
| 5 | alpha | `idea_R5_alpha_67483d892e4d` | 1.092469 | 0.3415 | +0.6410 | -0.2764 | 0.031 | discard |
| 6 | alpha_hinge | `idea_R6_alpha_hinge_b76c0112444d` | 0.623497 | 0.3610 | +0.9800 | -0.2669 | 0.031 | discard |
| 7 | alpha_hinge | `idea_R7_alpha_hinge_8729f540a07b` | 0.520947 | 0.3100 | +1.2689 | -0.2708 | 0.062 | discard |
| 8 | alpha | `idea_R8_alpha_7347f35b47b4` | 2.621227 | 0.5460 | +2.4918 | -0.3030 | 0.125 | discard |
| 9 | alpha_hinge | `idea_R9_alpha_hinge_204e1b220729` | 0.873363 | 0.2799 | +1.5240 | -0.2726 | 0.062 | discard |
| 10 | hinge_beta | `idea_R10_hinge_beta_d83ece9abcc0` | 1.133501 | 0.4161 | +1.6066 | -0.2935 | 0.031 | discard |

## 与目标的距离（最后一轮）

- rho_le_target: FAIL
- df_in_band: FAIL
- dr_le_max: PASS
- generation_ok: PASS
- rho 差距: +0.0361（需 <= 0.38）
- 3-seed 确认记录: 2 轮

## 硬禁止（反重复记忆）

counterfactual, gradient_surgery, projection, tau_field, tau_schedule

## 预算

- rounds 10/10, gpu 1127/1200s, crash 0
