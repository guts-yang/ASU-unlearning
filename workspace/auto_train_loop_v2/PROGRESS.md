# PROGRESS — 自主训练闭环 v2

**run_id**: `auto-train-v2-20261123`  **next_round**: 3  **phase**: `idle`
**best**: 0.4206 (idea_R1_baseline_d8f03a3b50d0)
**llm_backend**: `REVIEW_UNAVAILABLE` — 确定性规则引擎，非 LLM
**冻结目标**: rho<=0.38, |dF-1.1416|<=0.15, dR<=-0.2, refusal<=0.1, 3-seed std(rho)<=0.05
**v2 修复**: generation_gate 与 3-seed 确认轮已实跑，TARGET_MET 不再结构性不可达

| round | lever | idea | val_bpb | rho | dF | dR | refusal | status |
|---|---|---|---|---|---|---|---|---|
| 1 | baseline | `idea_R1_baseline_d8f03a3b50d0` | 0.420600 | 0.4206 | +1.1416 | -0.3013 | 0.031 | keep |
| 2 | alpha_hinge | `idea_R2_alpha_hinge_b3f0a2fec7ac` | 1.161411 | 0.2209 | +0.5279 | -0.2714 | 0.000 | discard |
| 3 | alpha_hinge | `idea_R3_alpha_hinge_37d5b766bd34` | 0.000000 | nan | +nan | +nan | nan | crash |

## 与目标的距离（最后一轮）

- rho_le_target: FAIL
- df_in_band: FAIL
- dr_le_max: FAIL
- generation_ok: FAIL
- rho 差距: +nan（需 <= 0.38）
- 3-seed 确认记录: 0 轮

## 硬禁止（反重复记忆）

counterfactual, gradient_surgery, projection, tau_field, tau_schedule

## 预算

- rounds 3/10, gpu 215/1200s, crash 1
