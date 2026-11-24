# 自主训练闭环 v5：跳出「不可能三角」的稳定化机制

**日期**：2026-11-24　**技能**：`/idea-discovery` + `/autoresearch-loop`
**目录**：`workspace/auto_train_loop_v5/`
**承接**：v4 结论——在调 \(\eta\)/\(\alpha\) 的 lever 空间里，「低 \(\bar\rho\)、低 \(\operatorname{std}(\rho)\)、所有 order 的 \(\Delta F\) 留在等预算带」三者不可兼得；继续调这两个量是死胡同。

## 0. 本轮改动（只改 lever 家族，其余冻结）

| 项 | v4 | **v5** |
|---|---|---|
| 起点 | v3 R2（固定协议下 `good=True`） | **v4 R3**：v4 中 `val_bpb` 最佳、且评估指标已在 3 个独立数据顺序上复现的 idea（`alpha=0.04, β=0.15, \bar\rho=0.2570, \operatorname{std}=0.1019`） |
| 新增 lever | `stability_lr` / `alpha_tune` / `beta_tune` | **`ema`**（指数移动平均权重，decay 0.90/0.95/0.99）、**`tail_avg`**（最后 10/20 步参数平均）、**`retain_mix`**（retain batch 8/12）——均为**与数据顺序无关的标准稳定化机制** |
| 复现轴 | 3 个独立数据顺序 | **不变**（0/1/2） |
| 单一指标 / 终止判据 | \(\bar\rho+2\operatorname{std}+\) 最坏约束 | **完全冻结不变**，保证跨版本可比 |
| 预算 | 5 轮 | 5 轮 |

这是上一轮明确建议的下一步：不再调 \(\eta\)/\(\alpha\)，而是引入与数据顺序无关的正则来降低敏感度，同时保住 \(\Delta F\) 预算带。

## 1. 每轮启动与结束条件

**启动 S1–S6**：`preflight()`（CUDA / 模型 / 数据 / `audit_depth.json` / `common.py`）；`state.phase ∈ {idle, done}`；上一轮 `replications/R{n-1}.json` 的 `order_results` 恰为 3 份且逐份通过 `verify_run()`；预算未耗尽（`MAX_ROUNDS=5` / `MAX_GPU_SECONDS=1800`）；`idea_spec` 通过 schema + `seen_ideas` 反重复；未达 `good`。

**结束**：3 个 order 各自返回 0、grep 到 `^rho_p3:` / `^refusal_rate:`；`runs/R{n}_o{k}.json` 全部落盘且字段有限；`replications/R{n}.json` 写出 \(\bar\rho,\operatorname{std},\text{val\_bpb},\text{checks},\text{good}\)；判定 `keep/discard/crash`；追加 `results.tsv` + 原子 `state.json` + `PROGRESS.md`；git commit。失败 → `crash`，连续 3 次 → `HALTED`。`state.json` 原子写，`--resume` 续跑。

## 2. 评估与思路生成

每个 order 实测：token 加权 answer NLL 的 \(F_1,\Delta F,\Delta R\)；P3 恢复攻击（20 步 SGD）后 \(F_3\)；\(\rho_{P3}=(F_1-F_3)/(F_1-F_{\text{base}})\)；32 条 greedy 生成拒答/退化率。

**独立复现聚合**：

\[ \bar\rho=\tfrac13\sum_o\rho_o,\qquad \operatorname{std}(\rho)=\sqrt{\tfrac13\sum_o(\rho_o-\bar\rho)^2} \]

\[ \text{val\_bpb}=\bar\rho+2\operatorname{std}(\rho)+1.5\max_o\lvert\Delta F_o-\Delta F_{\text{band}}\rvert+1.5\max(0,\max_o\Delta R_o-\Delta R_{\max})+0.01\kappa+2\max(0,\max_o\text{refusal}_o-0.10) \]

**稳定化机制**（在 `train.py` 里显式分支，训练结束后、评估前应用）：

- `ema`：每步维护影子参数 \( \theta_{\text{ema}}\leftarrow d\,\theta_{\text{ema}}+(1-d)\,\theta \)，最后用 \(\theta_{\text{ema}}\) 评估；
- `tail_avg`：对最后 \(k\) 步参数取平均后评估；
- `retain_mix`：把 retain batch 从 4 增到 \(n\)，用更稳的梯度降低顺序敏感度；
- `none`：种子 idea 原样。

**诊断规则**（确定性，无 LLM 后端 → `REVIEW_UNAVAILABLE`）：

| 诊断 | lever |
|---|---|
| 无历史 | `seed`（重新验证 v4 R3） |
| \(\operatorname{std}(\rho)>0.05\) | `ema` → `tail_avg` → `retain_mix` |
| 任一 order \(\Delta F\) 出带 | `retain_mix`（梯度稳定） |
| \(\bar\rho>0.38\) | `beta_tune` |
| 全过 | `alpha_tune` 保守探针 |

`seen_ideas`（SHA）反重复；`tau_field` / `tau_schedule` / `projection` / `counterfactual` / `gradient_surgery` 硬禁止。输出 `idea_spec` 含 hypothesis、params（含 `stabilizer`）、kill criterion、source、complexity、replication_orders、`seed_idea`。

## 3. 思路回灌

```
idea_engine.next_idea(state)
  → ideas/R{n}.json → current_idea.json（原子写）
  → train.py --data-order k --out runs/R{n}_o{k}.json  (k=0,1,2)
  → replications/R{n}.json（独立复现聚合）
  → state.history → 下一轮诊断输入
```

`train.py` 是唯一可编辑代码、通用解释器，稳定化分支显式实现；未识别 `stabilizer` 直接 `ValueError`。`keep` 更新 `best_idea`，`discard` 保留为负结果证据；不做 `git reset --hard`。

## 4. 终止：何为"比较好的训练结果"

冻结（`prepare.aggregate().good`），与 v4 完全一致：

\[ \text{good}=\bigwedge\begin{cases}
\bar\rho\le0.38,\ \operatorname{std}(\rho)\le0.05\\
\forall o:\ \lvert\Delta F_o-1.1416\rvert\le0.15,\ \Delta R_o\le-0.20\\
\forall o:\ \text{refusal}_o\le0.10\ \land\ \text{mean\_len}_o\ge8\\
\text{3 orders 齐全}
\end{cases} \]

终止三种：`TARGET_MET` / `BUDGET_EXHAUSTED` / `HALTED`。预算耗尽如实报告未达标及差距，**禁止放宽阈值**。

## 5. 记录与追踪

`results.tsv`（5 列、未跟踪）、`state.json`（原子、可 resume、含 `seed_idea`）、`ideas/`、`runs/R{n}_o{k}.json`、`replications/R{n}.json`、`logs/`、`PROGRESS.md`、`AUTORESEARCH_SUMMARY.md`、独立嵌套 git 仓库的逐轮 commit。

可重复：`run_id` 固定、`INIT_SEED=0`、`REPL_ORDERS=(0,1,2)` 显式、`prepare.py`/`program.md`/`DESIGN_V5.md` 循环期只读、`--resume`/`--reset`。无人工干预：`AUTO_PROCEED=true`，3-order 复现自动完成。

## 6. 适配与局限

- 不使用 `val_bpb` 原义 / 300 秒墙钟 / `uv run` / 主仓库分支 / `git reset --hard`。
- EMA / tail-avg 影子参数占约 4.7GB 显存；评估协议与 v4 完全一致，指标可比。
- 数据顺序是唯一变异轴（预训练权重 + `dropout=0` → seed 无作用）；这测的是对数据排列的稳健性。
- 生成质量门是固定模板启发式，不检测通顺但无关的替换答案。
- 单 1B / TOFU forget10 / 60 steps / P3 单攻击强度；不得外推。
- 无跨模型评审后端，思路裁决为执行者 provisional。

## 7. 实跑结果（2026-11-24）

5 轮 / 940 GPU 秒 / 0 crash / `BUDGET_EXHAUSTED` / `TARGET_MET=False`；5 个 idea 的评估指标全部在 3 个独立数据顺序下被实验复现。best val_bpb 0.821330（R2，EMA 0.90）。

- **EMA 打破了一半三角**：R3（EMA 0.95）\(\bar\rho=0.1412\)、\(\operatorname{std}=0.0354\) 双双达标，但 \(\Delta F=0.67\text{–}0.92\) 全部跌出等预算带。
- **事后组合探针** `COMBINATION_PROBE.md`：`EMA0.95 + β=0.20` 把 val_bpb 压到 0.5685，\(\bar\rho=0.1260\)、\(\operatorname{std}=0.0263\) 达标，唯一失败 `worst_df=0.2532>0.15`。三角被逼近但未打破。
- 阈值全程冻结，如实停机。详见 `AUTORESEARCH_SUMMARY.md`。
