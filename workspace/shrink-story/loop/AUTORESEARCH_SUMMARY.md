# AUTORESEARCH_SUMMARY — 收缩型故事自循环

**last_keep_commit**: `e0881a5`  **停止原因**: `max_experiments=8` 到达
**单一指标**: `val_bpb` = 等预算收缩遗憾 $r(c)$（越低越好，anchor 固定 0）

## 结果

- anchor 轮: 2（global τ=2.25 / 2.607，实跑产出）
- candidate `keep`: 0 / 6
- candidate `discard`: 6
- best $r$ = 0.000000

| round | status | val_bpb | peak GB | description |
|---|---|---|---|---|
| A1 | keep | +0.000000 | 51.1 | anchor: global tau=2.25 equal-budget control |
| A2 | keep | +0.000000 | 51.1 | anchor: global tau=2.607 equal-budget control |
| C1 | discard | +0.302613 | 51.1 | candidate: linear_up 1.5 to 3 low2high |
| C2 | discard | +0.599510 | 51.1 | candidate: linear_down 3 to 1.5 high2low |
| C3 | discard | +0.251062 | 51.1 | candidate: random temporal tau breaks ordering |
| C4 | discard | +0.816921 | 51.1 | candidate: step schedule tests gradual necessity |
| C5 | discard | +0.093430 | 51.1 | candidate: SensiTemp spatial field |
| C6 | discard | +0.314867 | 51.1 | candidate: shuffled spatial field control |

## 对收缩型故事的影响

- **第 3 幕收缩**：`low2high` 在等预算下未胜出，ΔF 增益由平均温度解释；故事改写为「空间与时间两个维度都未能在等预算下胜出」，第 4 幕深度审计成为唯一正面贡献。
- **课程形状同样被证伪**：打乱时间顺序 C3 的遗憾 +0.2511 反而低于 low2high C1 的 +0.3026；不仅「时间维度」未通过等预算检验，「渐进/课程形状」本身也没有可辨识收益，只剩「平均 τ 增大」这一强度效应。
- 幕 2 复现成立：SensiTemp 场 C5 遗憾最小但仍为正（+0.0934），打乱场 C6 更差（+0.3149），空间分配主张继续被否决。

## 适配声明

- 未使用 `val_bpb` 原义 / 300 秒墙钟 / `uv run` / 主仓库 `autoresearch/<tag>` 分支 / `git reset --hard`。
- `discard` 使用 `git checkout <last_keep> -- train.py` 文件级回退，避免破坏主工作区用户产物。
- 单 seed、60 steps、1B/TOFU；不得外推。跨模型评审后端不可用。

## 交接

- `results.tsv` → `paper_under_audit/results/experiments.csv`（anti-autoresearch bridge）
- 本文件 + `run.log` + `train.py` → `submission.tex` 可锚定句子与 `code/`
- 取证 verdict 只允许 `REVIEW_UNAVAILABLE`（确定性-only 回退），不得声称 CLEAN。

## 审计回写（anti-autoresearch）

- `audit_verdict`: `REVIEW_UNAVAILABLE`
- `observability_level`: L2（`submission.tex` + `code/` + `results/experiments.csv`）
- `adjudicator`: `deterministic-rules-v2`；`anchoring_verified=true`；`human_review_required=true`
- `PAPER_DIR`: `/usr/local/workspace/ASU-unlearning-Auto/workspace/shrink-story/paper_under_audit`
- 审计时间：2026-11-23；四个确定性检查器 0 finding；10 个语义/代码级维度 `review_unavailable` → 阻断无罪判定
- `AUDIT_FEEDBACK.md` 已生成：2 处人工待核实线索（摘要与结果表语义冲突、best/baseline 表述），下一轮实验线索指向第 5 幕抗恢复目标

## 继续轮次回写（Round 2 / v2 取证）

- `loop2/`（6 轮，1 keep / 5 discard / 0 crash）：等预算 anchor B0 `rho_P3=0.4206`；B1–B4 子空间投影全部无效；B5 hinge 被指标误杀（见 `MATCHED_DF_CONTROL.md`）。
- `loop2/MATCHED_DF_CONTROL.md`：B6（τ=2.607，无 hinge）`rho=0.5909` vs B5 `rho=0.4085`，ΔF 匹配（1.7914 / 1.8046）→ 抗恢复信号真实但需排除退化输出。
- `paper_under_audit_v2/`：修订版摘要，14 行结果表；重新取证 `L2 / 48 claims / 0 deterministic findings / REVIEW_UNAVAILABLE`（结构性，见 `AUDIT_COMPARISON.md`）。
- `AUDIT_FEEDBACK.md` 追加线索 D/E/F；`STORY_DESIGN.md` 追加 §7/§8。
