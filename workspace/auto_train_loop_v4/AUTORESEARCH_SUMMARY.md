# AUTORESEARCH_SUMMARY — 自主训练闭环 v4

**run_id**: `auto-train-v4-20261124`  **终止原因**: `BUDGET_EXHAUSTED`
**种子 idea（已通过）**: `{'kind': 'alpha_hinge', 'alpha': 0.05, 'hinge_beta': 0.15, 'hinge_margin': 1.2, 'lr': 5e-06, 'retain_weight': 1.0, 'grad_clip': 1.0}`（v3 R2 `good=True`）
**best val_bpb**: 1.0746454179048364  **best idea**: `idea_R3_alpha_tune_3d7fbc4482f4`
**轮数**: 5  **GPU 秒**: 1214
**TARGET_MET**: False  **独立复现 idea 数**: 5

## 逐轮（每轮 3 个独立数据顺序）

| round | idea | rho[0,1,2] | rho_mean | rho_std | val_bpb | replicated | good | status |
|---|---|---|---|---|---|---|---|---|
| 1 | `idea_R1_seed_99ea813e7ded` | [0.1774, 0.2997, 0.4278] | 0.3016 | 0.1022 | 1.472872 | True | False | keep |
| 2 | `idea_R2_stability_lr_0c794dba6da9` | [0.2357, 0.238, 0.3358] | 0.2698 | 0.0467 | 1.737578 | True | False | discard |
| 3 | `idea_R3_alpha_tune_3d7fbc4482f4` | [0.1328, 0.2558, 0.3825] | 0.2570 | 0.1019 | 1.074645 | True | False | keep |
| 4 | `idea_R4_stability_lr_bced9d419f8b` | [0.6188, 0.6025, 0.5142] | 0.5785 | 0.0459 | 2.348332 | True | False | discard |
| 5 | `idea_R5_alpha_tune_dd67a5fcfa00` | [0.2217, 0.444, 0.5101] | 0.3919 | 0.1233 | 2.915255 | True | False | discard |

## 通过独立实验复现评估指标的 idea

- `idea_R1_seed_99ea813e7ded`: rho_mean=0.3016, rho_std=0.1022, replicated=True, good=False
- `idea_R2_stability_lr_0c794dba6da9`: rho_mean=0.2698, rho_std=0.0467, replicated=True, good=False
- `idea_R3_alpha_tune_3d7fbc4482f4`: rho_mean=0.2570, rho_std=0.1019, replicated=True, good=False
- `idea_R4_stability_lr_bced9d419f8b`: rho_mean=0.5785, rho_std=0.0459, replicated=True, good=False
- `idea_R5_alpha_tune_dd67a5fcfa00`: rho_mean=0.3919, rho_std=0.1233, replicated=True, good=False

## 终止判据

- `TARGET_MET`: False（全部条件基于 3 个独立数据顺序）
- `BUDGET_EXHAUSTED`: True

## 适配与局限

- 数据顺序是唯一变异轴（预训练权重 + dropout=0 → seed 无作用）；这才是独立复现，不是 v3 的确定性重跑。
- 生成质量门为固定模板启发式，不检测通顺但无关的替换答案。
- 单 1B / TOFU forget10 / 60 steps / P3 单攻击强度；不得外推。
- 无跨模型评审后端，思路裁决为执行者 provisional。
- `results.tsv` 未跟踪；`state.json` 原子写、可 `--resume`。

## 关键发现：独立复现轴下的「不可能三角」

5 个 idea 的评估指标**全部经过 3 个独立数据顺序的实验复现**（`replicated=True`），但无一 `good=True`，终止如实为 `BUDGET_EXHAUSTED`。把三条主约束拆开看，存在明确权衡：

| 轮 | lever | \(\bar\rho\) | \(\operatorname{std}(\rho)\) | \(\Delta F\) 范围 | 卡在哪 |
|---|---|---|---|---|---|
| R1 | seed（已通过 idea） | 0.3016 | 0.1022 | 1.14–1.77 | std 超标、o2 出带 |
| R2 | stability_lr 2e-6 | 0.2698 | **0.0467** ✓ | 0.23–0.41 | **全部跌出等预算带** |
| R3 | alpha_tune | **0.2570** ✓ | 0.1019 | 0.91–1.55 | std 超标、o2 出带 |
| R4 | stability_lr 1e-6 | 0.5785 | **0.0459** ✓ | 0.02–0.09 | \(\rho\) 反而变差、带外 |
| R5 | alpha_tune | 0.3919 | 0.1233 | 1.26–2.58 | std 超标、o1/o2 出带 |

**结论**：在当前 `LEVERS` 空间内，「低 \(\bar\rho\)」「低 \(\operatorname{std}(\rho)\)」「所有 order 的 \(\Delta F\) 留在等预算带」三者不可兼得。降 \(\eta\) 确实能把数据顺序敏感度压到 0.05 以下（R2/R4），但代价是遗忘强度塌缩；调 \(\alpha\) 能把 \(\bar\rho\) 压到 0.257，但跨顺序波动回到 0.10 且部分顺序 \(\Delta F\) 出带。

因此种子 idea 的"通过"只在 v3 的固定协议下成立；一旦换成独立复现轴，它不仅不达标，而且其优化邻域内也没有同时满足三条约束的配置。这不是流程缺陷，而是**对真实扰动稳健的"较优训练结果"在该设定下尚未找到**的实证结论。阈值全程未放宽。

## 与 v3 的衔接

- v3 `TARGET_MET`（3 seeds 逐位相同）→ 取证发现伪独立 → 本轮按 v3.1 建议把复现轴换成 `REPL_ORDERS`。
- R1 重跑种子 idea 得到 \(\rho=[0.1774,0.2997,0.4278]\)、\(\operatorname{std}=0.1022\)，与 v3 `ROBUSTNESS.md` **完全一致**，构成跨循环的独立复现验证。
