# AUTORESEARCH_SUMMARY — 自主训练闭环 v3

**run_id**: `auto-train-v3-20261123`  **终止原因**: `TARGET_MET`
**best val_bpb**: 0.20473700374568476  **best idea**: `idea_R2_alpha_hinge_4773d71b0720`
**轮数**: 2  **GPU 秒**: 462
**TARGET_MET**: True  **已复现 idea 数**: 2

## 逐轮（每轮 3 seeds）

| round | idea | rho[0,1,2] | rho_mean | rho_std | val_bpb | replicated | good | status |
|---|---|---|---|---|---|---|---|---|
| 1 | `idea_R1_baseline_d8f03a3b50d0` | [0.417, 0.417, 0.417] | 0.4170 | 0.0000 | 0.498423 | True | False | keep |
| 2 | `idea_R2_alpha_hinge_4773d71b0720` | [0.1774, 0.1774, 0.1774] | 0.1774 | 0.0000 | 0.204737 | True | True | keep |

## 通过实验复现评估指标的 idea

- `idea_R1_baseline_d8f03a3b50d0`: rho_mean=0.4170, rho_std=0.0000, replication=True, good=False
- `idea_R2_alpha_hinge_4773d71b0720`: rho_mean=0.1774, rho_std=0.0000, replication=True, good=True

## 终止判据

- `TARGET_MET`: True（全部条件基于 3-seed 复现指标）
- `BUDGET_EXHAUSTED`: False

## 适配与局限

- 数据顺序固定（DATA_ORDER_SEED=0），3 seeds 只变初始化；这测的是初始化稳健性，不含数据顺序扰动。
- 生成质量门为固定模板启发式，不检测通顺但无关的替换答案。
- 单 1B / TOFU forget10 / 60 steps / P3 单攻击强度；不得外推。
- 无跨模型评审后端，思路裁决为执行者 provisional。
- `results.tsv` 未跟踪；`state.json` 原子写、可 `--resume`。

## 取证修正：TARGET_MET 的适用边界（必须阅读）

循环确实 `TARGET_MET` 停机，且 `idea_R2_alpha_hinge_4773d71b0720` 的评估指标在 3 次运行中被复现。但事后取证（`ROBUSTNESS.md`）显示这次复现是**伪独立**：

- 基座为预训练权重、`attention_dropout=0.0`、且 `DATA_ORDER_SEED=0` 固定 → `--seed` 没有任何可作用的随机源；3 次运行的 \(\rho\) 逐位相同（0.177389），`rho_std=0.000000` 只证明**确定性可重跑**。
- 引入独立变异轴（数据顺序 0/1/2）后：\(\rho=[0.1774, 0.2997, 0.4278]\)，\(\rho_{\text{mean}}=0.3016\)、\(\rho_{\text{std}}=0.1022>0.05\)；与 v2 的 R4 确认失败值完全一致（v2 的 seed 同时改变数据顺序）。

**结论**：`TARGET_MET` 在 v3 冻结的协议下成立且流程按设计停机；但"通过实验复现评估指标的 idea"只达到**协议级确定性复现**，不构成跨数据顺序的统计复现。不能据此主张该 idea 稳健优于 baseline。阈值全程未放宽，此处只是补充适用边界。

**v3.1 修正建议**（下一版）：把每轮的复现轴从 `REPL_SEEDS` 换成 `REPL_ORDERS`（3 个不同的数据顺序排列），并保留 \(\operatorname{std}(\rho)\) 在单一指标内；这样终止判据才会对真实扰动稳健。按现有实测，R2 配置在该修正下**不会**达标。

> `train_loop_snapshot.py` 保留产生循环结果的确切代码；`robustness_check.py` / `ROBUSTNESS.md` 为事后取证，未写入 `results.tsv`、未修改终止判据。
