# program.md — 收缩型故事 Round 2（固定，不修改）

> 触发：`loop/AUDIT_FEEDBACK.md` 线索 A —— 第 3 幕已被等预算对照推翻，不再尝试 τ 的空间/时间分配；
> 转向 `STORY_DESIGN.md` 第 5 幕（抗恢复 / 子空间选择性），且**每个新候选必须先声明等预算 anchor**。
> 与 Round 1 完全独立的新循环、新指标、新 git 仓库；本文件在循环期间不可改。

## 目标

检验第 5 幕的可检验核心问题：**在固定平滑预算下，能否在不牺牲遗忘强度与 retain 的前提下降低 P3 恢复率 \(\rho_{P3}\)**（即让遗忘"更深"，而不只是让 forget NLL 更大）。

## 硬约束

1. **只读**：`prepare.py`（含固定指标 `compute_regret`、P3 协议、从实测读取的 \(F_{\text{base}}\)）、`program.md`、上游 `common.py` 与全部既有结果。
2. **唯一可编辑代码**：`train.py`，每轮只改顶部 `CONFIG`。
3. **固定预算**：全局 \(\tau=2.25\)（Round 1 的 A1 anchor）、60 steps、\(\alpha=0.1\)、\(\eta=5\times10^{-6}\)、seed 0、TOFU forget10/retain90、Llama-3.2-1B。
4. **P3 协议固定**：训练后对 forget 子集做 20 步 SGD（\(\eta=10^{-5}\)，batch 4），测恢复后 forget NLL \(F_3\)；\(F_{\text{base}}=0.4621\) 从 `audit_depth.json` 读取。
5. **单一指标**（越低越好）：
   \[ r_2(c)=[\rho_c-\rho_a]+1.5\max(0,\Delta R_c-\Delta R_a)+1.5\max(0,\Delta F_a-\Delta F_c)+0.02\kappa(c) \]
   其中 \(\rho_{P3}=(F_1-F_3)/(F_1-F_{\text{base}})\)。第三项专门堵住"少忘一点 → 分母变小 → 看起来更深"的退化路径。
6. **禁止新增依赖**；日志重定向 `python train.py > run.log 2>&1`，只 grep `^val_bpb:` / `^peak_vram_mb:`。
7. **`results.tsv` 不提交**（5 列、untracked）；`discard` 用文件级回退，禁止 `git reset --hard`。
8. 不使用 `val_bpb` 原义 / 300 秒墙钟 / `uv run` / 主仓库分支 / 跨模型 reviewer。

## Run order（6 轮，跑完即停）

| 轮 | kind | CONFIG.projection | \(\kappa\) | 说明 |
|---|---|---|---|---|
| B0 | anchor | `none` | 0 | 等预算 depth anchor（必须最先跑，产出 \(\rho_a,\Delta F_a,\Delta R_a\)） |
| B1 | candidate | `bottom_pc` | 1 | RepSelect 式：forget 更新只保留 bottom-PC 方向 |
| B2 | candidate | `top_pc` | 1 | 反对照：只保留 top-PC（应与 retain 冲突） |
| B3 | candidate | `random_ortho` | 1 | 同 rank 的随机正交方向（打乱对照） |
| B4 | candidate | `sign_gate` | 1 | SAGO 式 element-wise sign gating：与 retain 梯度反号的分量置零 |
| B5 | candidate | `hinge_keep` | 2 | 抗恢复约束：hinge 要求 forget NLL 不低于 margin（单环近似，非双环 meta） |

## 判定

- \(r_2(c)<0\) → `keep`，主张存活；否则 `discard`（主张收缩）。
- anchor 固定 \(r_2=0\)，不参与竞标。
- 全部 `discard` 也算有效结论：第 5 幕在等预算下同样不能成立，故事进一步收缩为"负结果 + 评估基准"。

## 停止与诚实性

`max_experiments = 6`。停止后写 `AUTORESEARCH_SUMMARY.md`，并把结果回填 `STORY_DESIGN.md`。
单 seed、60 steps、1B/TOFU；P3 只测一种攻击强度和一种优化器，不得外推为"抗恢复能力"。
`R2-C` 的双环 meta 目标**未实现**，B5 只是单环 hinge 近似，必须如实标注。
