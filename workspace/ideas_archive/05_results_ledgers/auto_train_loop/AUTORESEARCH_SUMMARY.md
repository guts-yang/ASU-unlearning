# AUTORESEARCH_SUMMARY — 自主训练闭环

**run_id**: `auto-train-20261123`  **终止原因**: `BUDGET_EXHAUSTED`
**best val_bpb**: 0.4206  **best idea**: `idea_R1_baseline_d8f03a3b50d0`
**轮数**: 8  **GPU 秒**: 508

## 逐轮结果

| round | idea | lever | val_bpb | rho | dF | dR | status |
|---|---|---|---|---|---|---|---|
| 1 | `idea_R1_baseline_d8f03a3b50d0` | baseline | 0.420600 | 0.4206 | +1.1416 | -0.3013 | keep |
| 2 | `idea_R2_hinge_beta_d83ece9abcc0` | hinge_beta | 1.035594 | 0.4178 | +1.5401 | -0.2984 | discard |
| 3 | `idea_R3_hinge_beta_c5d64d266a0f` | hinge_beta | 1.422966 | 0.4085 | +1.8046 | -0.3001 | discard |
| 4 | `idea_R4_hinge_beta_35f67f0986a7` | hinge_beta | 2.220063 | 0.3820 | +2.3537 | -0.2971 | discard |
| 5 | `idea_R5_hinge_beta_fe1947004426` | hinge_beta | 2.689962 | 0.3400 | +2.6949 | -0.2928 | discard |
| 6 | `idea_R6_hinge_margin_4032da3e548f` | hinge_margin | 1.388231 | 0.4040 | +1.7844 | -0.2992 | discard |
| 7 | `idea_R7_hinge_margin_b366f50bd71a` | hinge_margin | 1.753841 | 0.4391 | +2.0048 | -0.2944 | discard |
| 8 | `idea_R8_hinge_margin_aeb2b32bc1f7` | hinge_margin | 2.258359 | 0.4695 | +2.3209 | -0.2848 | discard |

## 终止判据核对

- `TARGET_MET`: False
- `BUDGET_EXHAUSTED`: True
- `generation_gate` 未实现、`llm_backend` = REVIEW_UNAVAILABLE，因此本轮不可能 `TARGET_MET`；
  预算耗尽必须报告 **未达标**，不得放宽容差或伪称成功。

## 思路生成与应用

- `idea_engine.py` 的确定性诊断规则 + 有限 `LEVERS` 白名单 + `seen_ideas` 反重复。
- 每轮 `ideas/R{n}.json` → `current_idea.json` → `train.py` 解释执行 → `rounds/R{n}.json` 回写。
- `keep` 的 idea 成为后续基线锚；`discard` 保留在账本作为负结果证据。

## 适配声明

- 未使用 `val_bpb` 原义 / 300 秒墙钟 / `uv run` / 主仓库分支 / `git reset --hard`。
- 单 seed、60 steps、1B/TOFU、P3 仅一种攻击强度；不得外推。
- `results.tsv` 保持未跟踪；`state.json` 原子写、可 `--resume`。

## 可重复性验证

R1 baseline 与独立循环 `shrink-story/loop2` 的 anchor B0 在相同配置下**逐位一致**：

| 量 | 本循环 R1 | loop2 B0 | 差 |
|---|---|---|---|
| \(\rho_{P3}\) | 0.420600 | 0.420600 | 0 |
| \(\Delta F\) | 1.141628 | 1.141628 | 0 |
| \(\Delta R\) | -0.301287 | -0.301287 | 0 |

`--resume auto-train-20261123` 正确返回 `run already terminated: BUDGET_EXHAUSTED`，未重新训练、未污染账本。

## 本轮实质结论

`hinge_beta` 从 0.05 增到 0.30 时 \(\rho\) 单调下降（0.4178 → 0.3400），但 \(\Delta F\) 同步膨胀（1.5401 → 2.6949）。对称的 \(|\Delta F-\Delta F_{\text{band}}|\) 惩罚使这些候选全部被 `discard`——**闭环正确地拒绝了"靠忘得更狠换取更低恢复率"的退化路径**，这是 Round 2 指标缺陷修复后期望的行为。最佳仍是 baseline（0.4206），目标 \(\rho\le0.30\) 未达成，如实报告 `BUDGET_EXHAUSTED`。
