# PROGRESS — 自主训练闭环 v5（稳定化机制 + 独立复现）

**run_id**: `auto-train-v5-20261124`  **next_round**: 6  **phase**: `terminated`
**种子 idea（v4 已独立复现的最佳 idea）**: `{'kind': 'alpha_hinge', 'alpha': 0.04, 'hinge_beta': 0.15, 'hinge_margin': 1.2, 'lr': 5e-06, 'retain_weight': 1.0, 'grad_clip': 1.0, 'stabilizer': {'type': 'none'}}`，来源 v4 R3 / val_bpb=1.074645
**复现轴**: 3 个独立数据顺序 [0, 1, 2]（非 3 seeds；v3 已证明 seed 无作用）
**best**: 0.8213303055199495 (idea_R2_ema_337a7e05df1f)
**llm_backend**: `REVIEW_UNAVAILABLE` — 确定性规则引擎，非 LLM
**冻结目标**: rho_mean<=0.38, rho_std<=0.05, 所有 order |dF-1.1416|<=0.15, dR<=-0.2, refusal<=0.1
**TARGET_MET**: False  **已独立复现 idea**: 5

| round | lever | val_bpb | rho[0,1,2] | rho_mean | rho_std | worst_df | replicated | good | status |
|---|---|---|---|---|---|---|---|---|---|
| 1 | seed | 1.094645 | [0.1328, 0.2558, 0.3825] | 0.2570 | 0.1019 | 0.4092 | True | False | keep |
| 2 | ema | 0.821330 | [0.1085, 0.1734, 0.2461] | 0.1760 | 0.0562 | 0.3486 | True | False | keep |
| 3 | ema | 0.922514 | [0.1031, 0.1321, 0.1883] | 0.1412 | 0.0354 | 0.4671 | True | False | discard |
| 4 | retain_mix | 2.403337 | [0.247, 0.5308, 0.4026] | 0.3935 | 0.1160 | 1.0619 | True | False | discard |
| 5 | ema | 1.802439 | [0.197, 0.2078, 0.2382] | 0.2143 | 0.0175 | 1.0288 | True | False | discard |

## 最后一轮判据

- rho_mean_le_target: PASS
- rho_std_le_max: PASS
- all_df_in_band: FAIL
- all_dr_le_max: PASS
- all_generation_ok: PASS
- replicated: PASS
- rho_mean 差距: -0.1657
- rho_std 差距: -0.0325

## 预算

- rounds 5/5, gpu 940/1800s, crash 0
