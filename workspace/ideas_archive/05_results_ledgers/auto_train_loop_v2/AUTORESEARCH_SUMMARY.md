# AUTORESEARCH_SUMMARY — 自主训练闭环 v2

**run_id**: `auto-train-v2-20261123`  **终止原因**: `BUDGET_EXHAUSTED`
**best val_bpb**: 0.204737  **best idea**: `idea_R4_alpha_hinge_4773d71b0720`
**轮数**: 10  **GPU 秒**: 1127
**3-seed 确认轮**: 2

## 逐轮结果

| round | idea | lever | val_bpb | rho | dF | dR | refusal | status |
|---|---|---|---|---|---|---|---|---|
| 1 | `idea_R1_baseline_d8f03a3b50d0` | baseline | 0.498423 | 0.4170 | +1.1959 | -0.2960 | 0.031 | keep |
| 2 | `idea_R2_alpha_hinge_b3f0a2fec7ac` | alpha_hinge | 1.123706 | 0.2198 | +0.5523 | -0.2651 | 0.000 | discard |
| 3 | `idea_R3_alpha_hinge_37d5b766bd34` | alpha_hinge | 0.668043 | 0.1952 | +0.8397 | -0.2553 | 0.031 | discard |
| 4 | `idea_R4_alpha_hinge_4773d71b0720` | alpha_hinge | 0.204737 | 0.1774 | +1.1367 | -0.2542 | 0.031 | keep |
| 5 | `idea_R5_alpha_67483d892e4d` | alpha | 1.092469 | 0.3415 | +0.6410 | -0.2764 | 0.031 | discard |
| 6 | `idea_R6_alpha_hinge_b76c0112444d` | alpha_hinge | 0.623497 | 0.3610 | +0.9800 | -0.2669 | 0.031 | discard |
| 7 | `idea_R7_alpha_hinge_8729f540a07b` | alpha_hinge | 0.520947 | 0.3100 | +1.2689 | -0.2708 | 0.062 | discard |
| 8 | `idea_R8_alpha_7347f35b47b4` | alpha | 2.621227 | 0.5460 | +2.4918 | -0.3030 | 0.125 | discard |
| 9 | `idea_R9_alpha_hinge_204e1b220729` | alpha_hinge | 0.873363 | 0.2799 | +1.5240 | -0.2726 | 0.062 | discard |
| 10 | `idea_R10_hinge_beta_d83ece9abcc0` | hinge_beta | 1.133501 | 0.4161 | +1.6066 | -0.2935 | 0.031 | discard |

## 终止判据

- `TARGET_MET`: False
- `BUDGET_EXHAUSTED`: True
- 阈值全程冻结，未放宽；若未达标则如实报告差距。

## 3-seed 确认

- R4: rhos=[0.1774, 0.2997, 0.4278], std=0.1022, pass=False
- R7: rhos=[0.31, 0.6083, 0.5965], std=0.1379, pass=False

## 思路生成与回灌

- `idea_engine.py` 确定性诊断 + 有限 `LEVERS`（含 `alpha_hinge` 组合）+ `seen_ideas` 反重复。
- `ideas/R{n}.json` → `current_idea.json` → `train.py` → `rounds/R{n}.json` → `state.history`。

## 适配与局限

- 未使用 `val_bpb` 原义 / 300 秒墙钟 / `uv run` / 主仓库分支 / `git reset --hard`。
- 生成质量门为固定模板启发式，不等价于人工答案质量评估；不检测通顺但无关的替换答案。
- 单 1B 模型 / TOFU forget10 / answer-NLL / P3 单攻击强度；不得外推。
- 跨模型评审不可用，裁决为执行者 provisional。
- `results.tsv` 未跟踪；`state.json` 原子写、可 `--resume`。

## 关键发现（v2 相对 v1 的实质进展）

1. **`TARGET_MET` 从结构性不可达变为实证可判定**：R4 与 R7 的初筛（\(\rho\) / \(\Delta F\) 带 / retain / 生成质量）全部通过，自动触发了 3-seed 确认轮。v1 因 `generation_gate` 恒 False 从未走到这一步。
2. **两次确认均失败，流程正确拒绝了伪达标**：
   - R4：\(\rho=[0.1774, 0.2997, 0.4278]\)，\(\operatorname{std}=0.1022>0.05\)
   - R7：\(\rho=[0.3100, 0.6083, 0.5965]\)，\(\operatorname{std}=0.1379>0.05\)

   同一 `alpha_hinge` 配置跨 seed 的 \(\rho\) 波动达 0.25–0.30，说明 seed 0 上看到的"较优"主要是训练随机性，不是稳健改进。这是本轮最重要的科学结论，也正是统计稳健门控的设计目的。
3. **best 仍是未确认的单 seed 值** \(0.204737\)（R4），不能作为达标结果报告；终止原因如实为 `BUDGET_EXHAUSTED`。
4. **口径差异**：v2 为与显存共存改用分块 token 加权 NLL，与 v1 的逐序列平均略有差异（baseline \(\rho\) 0.4170 vs 0.4206、\(\Delta F\) 1.1959 vs 1.1416）；v2 内部比较有效，跨版本绝对值不直接可比。
5. **已知局限**：生成质量门只在 seed 0 实测，确认轮用 `--skip-gen` 并沿用主轮拒答率；更严格版本应对每个 seed 各测一次生成质量。拒答判定为固定模板启发式，不检测"通顺但无关的替换答案"。

## 结论

流程本身满足"无人工干预持续运行、环节衔接清晰、终止判据可计算、全程可追踪"的要求；但**在本设定与当前预算内没有出现满足冻结目标的稳健配置**。达标与否由实测决定，未通过放宽阈值伪称成功。若要推进，下一步应优先提升跨 seed 稳健性（更多 seed、固定数据顺序之外的扰动控制、或降低 hinge 对初始化的敏感度），而不是继续调 \(\tau\) 分配。
