# AUTORESEARCH_SUMMARY — 自主训练闭环 v5

**run_id**: `auto-train-v5-20261124`  **终止原因**: `BUDGET_EXHAUSTED`
**种子 idea（v4 已独立复现最佳）**: `{'kind': 'alpha_hinge', 'alpha': 0.04, 'hinge_beta': 0.15, 'hinge_margin': 1.2, 'lr': 5e-06, 'retain_weight': 1.0, 'grad_clip': 1.0, 'stabilizer': {'type': 'none'}}`（v4 R3）
**best val_bpb**: 0.8213303055199495  **best idea**: `idea_R2_ema_337a7e05df1f`
**轮数**: 5  **GPU 秒**: 940
**TARGET_MET**: False  **独立复现 idea 数**: 5

## 逐轮（每轮 3 个独立数据顺序）

| round | idea | rho[0,1,2] | rho_mean | rho_std | val_bpb | replicated | good | status |
|---|---|---|---|---|---|---|---|---|
| 1 | `idea_R1_seed_0c704e9ad66c` | [0.1328, 0.2558, 0.3825] | 0.2570 | 0.1019 | 1.094645 | True | False | keep |
| 2 | `idea_R2_ema_337a7e05df1f` | [0.1085, 0.1734, 0.2461] | 0.1760 | 0.0562 | 0.821330 | True | False | keep |
| 3 | `idea_R3_ema_84c6d13c4c96` | [0.1031, 0.1321, 0.1883] | 0.1412 | 0.0354 | 0.922514 | True | False | discard |
| 4 | `idea_R4_retain_mix_d272674845f4` | [0.247, 0.5308, 0.4026] | 0.3935 | 0.1160 | 2.403337 | True | False | discard |
| 5 | `idea_R5_ema_f3b9379d9d06` | [0.197, 0.2078, 0.2382] | 0.2143 | 0.0175 | 1.802439 | True | False | discard |

## 通过独立实验复现评估指标的 idea

- `idea_R1_seed_0c704e9ad66c`: rho_mean=0.2570, rho_std=0.1019, replicated=True, good=False
- `idea_R2_ema_337a7e05df1f`: rho_mean=0.1760, rho_std=0.0562, replicated=True, good=False
- `idea_R3_ema_84c6d13c4c96`: rho_mean=0.1412, rho_std=0.0354, replicated=True, good=False
- `idea_R4_retain_mix_d272674845f4`: rho_mean=0.3935, rho_std=0.1160, replicated=True, good=False
- `idea_R5_ema_f3b9379d9d06`: rho_mean=0.2143, rho_std=0.0175, replicated=True, good=False

## 终止判据

- `TARGET_MET`: False（全部条件基于 3 个独立数据顺序）
- `BUDGET_EXHAUSTED`: True

## 适配与局限

- 数据顺序是唯一变异轴（预训练权重 + dropout=0 → seed 无作用）；这才是独立复现，不是 v3 的确定性重跑。
- 生成质量门为固定模板启发式，不检测通顺但无关的替换答案。
- 单 1B / TOFU forget10 / 60 steps / P3 单攻击强度；不得外推。
- 无跨模型评审后端，思路裁决为执行者 provisional。
- `results.tsv` 未跟踪；`state.json` 原子写、可 `--resume`。

## 关键发现：EMA 打破了一半的「不可能三角」

v4 的三角是「低 \(\bar\rho\) / 低 \(\operatorname{std}(\rho)\) / 所有 order 的 \(\Delta F\) 在带内」不可兼得。v5 的 EMA 把前两项同时解决了，但把 \(\Delta F\) 压垮：

| 轮 | stabilizer | \(\bar\rho\) | \(\operatorname{std}(\rho)\) | \(\Delta F\) 范围 | 卡在哪 |
|---|---|---|---|---|---|
| R1 | none（种子） | 0.2570 | 0.1019 | 0.91–1.55 | std 超标、部分出带 |
| R2 | EMA 0.90 | **0.1760** | 0.0562 | 0.79–1.17 | std 略超、o0 出带 |
| R3 | EMA 0.95 | **0.1412** | **0.0354** ✓ | 0.67–0.92 | **全部出带** |
| R5 | EMA 0.99 | 0.2143 | **0.0175** ✓ | 0.11–0.22 | 全部出带、\(\bar\rho\) 回升 |
| R4 | retain_mix 8 | 0.3935 | 0.1160 | 1.63–2.20 | 全项恶化 |

**结论**：EMA 是与数据顺序无关的有效稳定机制（R3/R5 的 \(\operatorname{std}\) 分别 0.0354/0.0175，远优于 v4 最好的 0.0459），但它同时稀释了遗忘强度，使等预算带成为唯一剩余瓶颈。

## 事后组合探针（不写入 5 轮账本）

`COMBINATION_PROBE.md` 用相同 3 个独立数据顺序测了 EMA×遗忘压力组合：

| combo | \(\bar\rho\) | std | \(\Delta F\) | worst_df | val_bpb | good |
|---|---|---|---|---|---|---|
| EMA0.95 + α=0.06 | 0.2264 | 0.0658 | 0.85 / 1.02 / 1.29 | 0.2896 | 0.8025 | 否 |
| **EMA0.95 + β=0.20** | **0.1260** | **0.0263** | 0.89 / 0.92 / 1.14 | 0.2532 | **0.5685** | 否 |
| EMA0.90 + α=0.06 | 0.2983 | 0.1012 | 1.05 / 1.44 / 1.79 | 0.6502 | 1.4859 | 否 |

`EMA0.95 + β=0.20` 把 val_bpb 从循环内 best 0.8213 降到 **0.5685**，\(\bar\rho\) 与 std 双双达标，唯一失败是 `worst_df=0.2532>0.15`（前两个 order 的 \(\Delta F\) 仍偏低）。这说明三角**被显著逼近但未被打破**：提高遗忘压力能把部分 order 拉回带内，却会让跨 order 的 \(\Delta F\) 离散度变大。

**因此 `TARGET_MET` 仍为 False，终止如实为 `BUDGET_EXHAUSTED`；5 个 idea 的评估指标全部经过 3 个独立数据顺序的实验复现。**

## 下一步（不建议放宽阈值）

要真正打破三角，需要**按层 / 按样本自适应**的稳定化机制（例如只对承载事实的层做 EMA、或按样本梯度范数自适应权重），而不是全局 EMA × 全局压力。另一个方向是重新审视 \(\Delta F_{\text{band}}=1.1416\) 这个锚本身是否应随方法而变——但那属于**重新定义问题**，必须在新的 Problem Anchor 下、经独立复现后冻结，不能在本循环内临时放宽。
