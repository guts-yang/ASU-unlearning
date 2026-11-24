# PROGRESS — 自主训练闭环 v4（已通过 idea + 独立复现）

**run_id**: `auto-train-v4-20261124`  **next_round**: 6  **phase**: `terminated`
**种子 idea（已通过）**: `{'kind': 'alpha_hinge', 'alpha': 0.05, 'hinge_beta': 0.15, 'hinge_margin': 1.2, 'lr': 5e-06, 'retain_weight': 1.0, 'grad_clip': 1.0}`，来源 v3 R2 `good=True` / val_bpb=0.204737
**复现轴**: 3 个独立数据顺序 [0, 1, 2]（非 3 seeds；v3 已证明 seed 无作用）
**best**: 1.0746454179048364 (idea_R3_alpha_tune_3d7fbc4482f4)
**llm_backend**: `REVIEW_UNAVAILABLE` — 确定性规则引擎，非 LLM
**冻结目标**: rho_mean<=0.38, rho_std<=0.05, 所有 order |dF-1.1416|<=0.15, dR<=-0.2, refusal<=0.1
**TARGET_MET**: False  **已独立复现 idea**: 5

| round | lever | val_bpb | rho[0,1,2] | rho_mean | rho_std | worst_df | replicated | good | status |
|---|---|---|---|---|---|---|---|---|---|
| 1 | seed | 1.472872 | [0.1774, 0.2997, 0.4278] | 0.3016 | 0.1022 | 0.6312 | True | False | keep |
| 2 | stability_lr | 1.737578 | [0.2357, 0.238, 0.3358] | 0.2698 | 0.0467 | 0.9163 | True | False | discard |
| 3 | alpha_tune | 1.074645 | [0.1328, 0.2558, 0.3825] | 0.2570 | 0.1019 | 0.4092 | True | False | keep |
| 4 | stability_lr | 2.348332 | [0.6188, 0.6025, 0.5142] | 0.5785 | 0.0459 | 1.1187 | True | False | discard |
| 5 | alpha_tune | 2.915255 | [0.2217, 0.444, 0.5101] | 0.3919 | 0.1233 | 1.4428 | True | False | discard |

## 最后一轮判据

- rho_mean_le_target: FAIL
- rho_std_le_max: FAIL
- all_df_in_band: FAIL
- all_dr_le_max: PASS
- all_generation_ok: FAIL
- replicated: PASS
- rho_mean 差距: +0.0119
- rho_std 差距: +0.0733

## 预算

- rounds 5/5, gpu 1214/1800s, crash 0
