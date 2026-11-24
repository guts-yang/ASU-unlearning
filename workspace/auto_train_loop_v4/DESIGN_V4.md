# 自主训练闭环 v4：以「已通过的 idea」为起点、以「独立复现」为评估指标

**日期**：2026-11-24　**技能**：`/idea-discovery` + `/autoresearch-loop`
**目录**：`workspace/auto_train_loop_v4/`

## 0. 本轮两个新约束如何满足

| 约束 | 落实 |
|---|---|
| **针对已经通过的 idea** | 种子 idea 固定为 `auto_train_loop_v3/replications/R2.json` 中 `good=True` 的 `alpha_hinge(α=0.05, β=0.15, margin=1.2)`（\(\bar\rho=0.1774\)、val_bpb=0.204737）。整个 `LEVERS` 空间都是在这一个 idea 上做**单变量/组合微调**，不再从 baseline 重新摸索，也不重新引入已证伪方向 |
| **有通过实验复现评估指标的 idea** | 复现轴从 `3 seeds` 换成 **`3 个独立数据顺序`**。v3 取证已证明：预训练权重 + `dropout=0` + 固定数据顺序 → seed 无作用、3 次运行逐位相同，是伪独立。v4 让数据顺序成为唯一变异轴，每个 idea 的 \(\bar\rho,\operatorname{std}(\rho)\) 都来自真实扰动下的独立运行 |

**种子 idea 在 v4 里不直接继承"通过"标签**：R0 用独立复现轴重新验证它，结果作为后续所有比较的基线。这是对上一轮取证结论的直接执行。

## 1. 每轮启动与结束条件

**启动 S1–S6**：`preflight()`（CUDA / 模型 / 数据 / `audit_depth.json` / `common.py`）；`state.phase ∈ {idle, done}`；上一轮 `replications/R{n-1}.json` 的 `order_results` 恰为 3 份且逐份通过 `verify_run()`；预算未耗尽（`MAX_ROUNDS=5` / `MAX_GPU_SECONDS=1800`）；`idea_spec` 通过 schema + `seen_ideas` 反重复；未达 `good`。

**结束**：3 个 data order 各自返回 0、grep 到 `^rho_p3:` / `^refusal_rate:`；`runs/R{n}_o{k}.json` 全部落盘且字段有限；`replications/R{n}.json` 写出 \(\bar\rho,\operatorname{std},\text{val\_bpb},\text{checks},\text{good}\)；判定 `keep/discard/crash`；追加 `results.tsv` + 原子 `state.json` + `PROGRESS.md`；git commit。任一步失败 → `crash`，不静默前进；连续 3 次 → `HALTED`。

`state.json` 原子写（tmp + `replace`），`--resume` 从 `next_round` 续跑。

## 2. 评估与思路生成

**每个 order 实测**：token 加权 answer NLL 的 \(F_1,\Delta F,\Delta R\)；P3 恢复攻击（20 步 SGD）后 \(F_3\)；\(\rho_{P3}=(F_1-F_3)/(F_1-F_{\text{base}})\)（\(F_{\text{base}}\) 从 `audit_depth.json` 读取）；32 条 greedy 生成的拒答/退化率与平均长度。

**独立复现聚合**：

\[ \bar\rho=\tfrac13\sum_{o}\rho_o,\qquad \operatorname{std}(\rho)=\sqrt{\tfrac13\sum_o(\rho_o-\bar\rho)^2} \]

\[ \text{val\_bpb}=\bar\rho+2\operatorname{std}(\rho)+1.5\max_o\lvert\Delta F_o-\Delta F_{\text{band}}\rvert+1.5\max(0,\max_o\Delta R_o-\Delta R_{\max})+0.01\kappa+2\max(0,\max_o\text{refusal}_o-0.10) \]

\(\operatorname{std}(\rho)\) **在指标内部**，可复现性本身就是被优化的目标。

**思路生成**（确定性规则引擎，无 LLM 后端、`REVIEW_UNAVAILABLE`）→ `idea_spec` 含 hypothesis / params / expected_effect / kill_criterion / source / complexity / replication_orders / `seed_idea`。

| 诊断 | lever | 目的 |
|---|---|---|
| 无历史 | `seed` | 用独立轴重新验证已通过 idea |
| \(\operatorname{std}(\rho)>0.05\) | `stability_lr`(2e-6/1e-6) → `stability_retain`(1.5/2.0) → `stability_clip`(0.5/0.2) | **降低数据顺序敏感度** |
| 任一 order \(\Delta F\) 出带 | `alpha_tune` {0.04,0.06,0.07} | 拉回等预算带 |
| \(\bar\rho>0.38\) | `beta_tune` {0.10,0.20} | 压低恢复率 |
| 全过 | `margin_tune` {1.0,1.5} | 保守探针 |
| 候选已试 / 连续无改善 | 反重复跳过 / 结构化 pivot | 防重复与停滞 |

`FORBIDDEN_LEVERS` 硬禁止：`tau_field`、`tau_schedule`、`projection`、`counterfactual`、`gradient_surgery`（前三轮已证伪）。

## 3. 优化思路如何回灌下一轮

```
idea_engine.next_idea(state)   # 输入含上一轮的 rho_mean / rho_std / worst_df
  → ideas/R{n}.json
  → current_idea.json（原子写）
  → train.py --data-order k --out runs/R{n}_o{k}.json   (k=0,1,2)
  → replications/R{n}.json（独立复现聚合）
  → state.history 追加 → 下一轮诊断输入
```

`train.py` 是唯一可编辑代码、通用思路解释器；每轮变化的是 `idea_spec`（数据）。所有 lever 都有显式分支，未识别 `kind` 直接 `ValueError` 拒绝。`keep` 更新 `best_idea`；`discard` 保留为负结果证据；不做 `git reset --hard`。

## 4. 终止：何为"比较好的训练结果"

冻结在 `prepare.aggregate().good`，**全部条件基于 3 个独立数据顺序**：

\[ \text{good}=\bigwedge\begin{cases}
\bar\rho\le0.38,\quad \operatorname{std}(\rho)\le0.05 & \text{独立复现下的深度与稳健性}\\
\forall o:\ \lvert\Delta F_o-1.1416\rvert\le0.15 & \text{所有 order 在等预算带}\\
\forall o:\ \Delta R_o\le-0.20 & \text{所有 order 不损 retain}\\
\forall o:\ \text{refusal}_o\le0.10\ \land\ \text{mean\_len}_o\ge8 & \text{所有 order 输出不退化}\\
\text{3 orders 齐全} & \text{确为独立复现}
\end{cases} \]

终止三种：`TARGET_MET`（正常停机并输出**已独立复现的 idea**）、`BUDGET_EXHAUSTED`（如实报告未达标 + 差距 + 最佳 idea）、`HALTED`（连续 3 轮 crash）。**阈值全程冻结，禁止放宽伪称达标。**

## 5. 记录与追踪

| 产物 | 内容 | git |
|---|---|---|
| `results.tsv` | autoresearch 5 列（每轮一行，val_bpb 为独立复现指标） | 不提交 |
| `state.json` | 原子状态机：run_id / next_round / phase / best / seen_ideas / history / budget / `replicated_ideas` / `seed_idea` | 提交 |
| `ideas/R{n}.json` | 每轮思路（含 `seed_idea` 指针、kill criterion） | 提交 |
| `runs/R{n}_o{k}.json` | 每个独立数据顺序的完整实测 | 提交 |
| `replications/R{n}.json` | \(\bar\rho,\operatorname{std},\text{val\_bpb},\text{checks},\text{good}\) | 提交 |
| `logs/`、`run.log` | 重定向输出 | 忽略 |
| `PROGRESS.md` / `AUTORESEARCH_SUMMARY.md` | 人读进度、与目标距离、**已独立复现 idea 清单** | 提交 |
| 独立嵌套 git 仓库 | 每轮一次 commit | 隔离主仓库 |

**可重复执行**：`run_id` 固定；`INIT_SEED=0` 固定（预训练权重无 dropout）；数据顺序由 `REPL_ORDERS` 显式给出；`prepare.py`/`program.md`/`DESIGN_V4.md` 循环期只读；`--resume` / `--reset`。

**无人工干预**：`AUTO_PROCEED=true`，checkpoint 只写日志、不提问不等待；每轮自动完成 3 个独立复现。

## 6. 适配与局限

- 不使用 `val_bpb` 原义 / 300 秒墙钟 / `uv run` / 主仓库分支 / `git reset --hard`。
- 数据顺序是唯一变异轴；这测的是对数据排列的稳健性，不含模型初始化扰动（因预训练权重 + dropout=0，初始化扰动无意义）。
- 生成质量门为固定模板启发式，不检测"通顺但无关的替换答案"。
- 单 1B / TOFU forget10 / 60 steps / P3 单攻击强度；不得外推。
- 无跨模型评审后端，思路裁决为执行者 provisional。

## 7. 实跑结果（2026-11-24）

5 轮 / 1214 GPU 秒 / 0 crash / `BUDGET_EXHAUSTED` / `TARGET_MET=False`；**5 个 idea 的评估指标全部在 3 个独立数据顺序下被实验复现**（`replicated=True`），best 单指标 1.074645（R3）。

- R1（种子 idea 独立复现）：\(\bar\rho=0.3016\)、\(\operatorname{std}=0.1022\)、`good=False`，与 v3 `ROBUSTNESS.md` 一致 → 种子 idea 的"通过"只在固定协议下成立。
- 存在**不可能三角**：降 \(\eta\) 可把 \(\operatorname{std}\) 压到 0.0467/0.0459，但 \(\Delta F\) 塌缩到 0.02–0.41；调 \(\alpha\) 可把 \(\bar\rho\) 压到 0.2570，但 \(\operatorname{std}\) 回到 0.1019 且部分顺序 \(\Delta F\) 出带。三者不可兼得。
- 结论：对真实扰动稳健的"较优训练结果"在当前 lever 空间内尚未找到；如实停机，未放宽阈值。详见 `AUTORESEARCH_SUMMARY.md` 的"关键发现"节。
