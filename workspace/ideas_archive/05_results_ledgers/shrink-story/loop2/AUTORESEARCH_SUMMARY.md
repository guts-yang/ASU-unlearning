# AUTORESEARCH_SUMMARY — 收缩型故事 Round 2（抗恢复 / 子空间）

**last_keep_commit**: `2c4cd3e`  **停止原因**: `max_experiments=6` 到达
**单一指标**: `val_bpb` = 等预算深度遗憾 $r_2(c)$（越低越好，anchor 固定 0）
**anchor B0**: $\rho_{P3}=0.420600$

## 结果

- candidate `keep`: 0 / 5
- best $r_2$ = 0.000000

| round | status | r2 | rho_P3 | peak GB | description |
|---|---|---|---|---|---|
| B0 | keep | +0.000000 | 0.420600 | 51.3 | anchor: equal-budget depth baseline global tau=2.25 |
| B1 | discard | +0.027804 | 0.422691 | 51.3 | candidate: RepSelect bottom-PC forget update |
| B2 | discard | +0.022247 | 0.420559 | 51.3 | candidate: top-PC reverse control |
| B3 | discard | +0.028512 | 0.422389 | 51.3 | candidate: random orthogonal direction control |
| B4 | discard | +0.027469 | 0.423895 | 51.3 | candidate: SAGO-style element-wise sign gate |
| B5 | discard | +0.029642 | 0.408529 | 51.3 | candidate: single-loop hinge anti-recovery constraint |

## 对第 5 幕的影响

- **第 5 幕同样收缩**：所有子空间/抗恢复近似都未在等预算下降低 $\rho_{P3}$；故事最终只剩「等预算对照协议 + 恢复深度指标 + 简单标量 τ 是强基线」三个贡献。
- B5 只是单环 hinge 近似，`R2-C` 的双环 meta 目标**仍未实现**；若 B5 有收益，须补 matched-$\Delta F$ 对照排除「靠增大 $\Delta F$ 稀释 $\rho$」的退化路径。
- 投影算子 / sign mask 在训练前用基座模型算一次并固定，不是每步重算的严格 RepSelect / SAGO；属已知近似。

## 适配声明
- 未使用 `val_bpb` 原义 / 300 秒墙钟 / `uv run` / 主仓库分支 / `git reset --hard`；`discard` 为文件级回退。
- 单 seed、60 steps、1B/TOFU、P3 仅一种攻击强度与一种优化器，不得外推。跨模型评审不可用。

## 取证补实验（循环结束后，不写入 6 轮账本）

`MATCHED_DF_CONTROL.md` / `rounds2/B6.json`：

| 配置 | ΔF | ρ_P3 |
|---|---|---|
| B0 anchor τ=2.25 | +1.1416 | 0.420600 |
| B5 hinge_keep | +1.8046 | 0.408529 |
| B6 matched-ΔF 对照（τ=2.607，无 hinge） | +1.7914 | 0.590908 |

**两个必须如实记录的结论**：

1. **B5 被固定指标误杀**：在 ΔF 几乎相同（差 0.0132）时，B5 的 ρ 比无抗恢复项的对照低 0.1824，说明 hinge 约束确实降低了 P3 可恢复性。但 \(r_2\) 给 \(\kappa=2\) 记了 0.04 的复杂度惩罚，且第三项只惩罚「忘得更少」，最终 B5 得到 \(r_2=+0.0296\) 被 `discard`。这是**简单性准则过强 + 指标不对称**造成的假阴性，不是方法无效。
2. **指标缺陷**：\(r_2\) 第三项 `1.5*max(0, dF_a - dF_c)` 应改为对称的 `1.5*|dF_c - dF_a|`，或强制在同一 ΔF 预算带比较。prepare.py 是固定文件，本轮不改；修复只能放进 Round 3 新循环。

**但不能据此主张「抗恢复更深」**：hinge 可能只是让模型在 forget 上产生退化输出（拒答/乱码），P3 微调后 NLL 仍高不等于知识被擦除。必须补生成质量、答案分布、拒答率与 3-seed 复核；单 seed、单攻击强度、单优化器不得外推。`R2-C` 双环 meta 目标仍未实现。
