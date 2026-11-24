# 自主训练–评估–思路优化闭环 v3：以「已复现的评估指标」驱动

**日期**：2026-11-23　**技能**：`idea-discovery` + `autoresearch-loop`
**前置**：v1（`auto_train_loop/`）达标结构性不可达；v2（`auto_train_loop_v2/`）两次 3-seed 确认失败（\(\operatorname{std}(\rho)=0.1022/0.1379\)），证明**单 seed 的"较优"不可复现**。
**本轮新增要求**：流程必须建立在「**通过实验复现评估指标的 idea**」之上。

## 0. 核心改动：复现从"事后门"变成"指标本身"

| | v2 | **v3** |
|---|---|---|
| 每轮评估 | 1 个 seed，初筛通过后另跑 2 seeds 确认 | **每个 idea 固定跑 3 seeds**，一次聚合 |
| 单一指标 | \(\rho\)（单 seed） | \(\bar\rho+2\operatorname{std}(\rho)+\) 各 seed 最坏约束惩罚 |
| 终止判据 | 单 seed 初筛 + 事后 std | \(\bar\rho\)、\(\operatorname{std}(\rho)\)、**所有 seed** 的 \(\Delta F/\Delta R/\)生成质量 |
| 数据顺序 | 随 seed 变化（数据顺序 + 初始化噪声混在一起） | **固定**（`DATA_ORDER_SEED=0`），seed 只控制初始化 |

因此 v3 里每个 idea 的评估指标**天然是被实验复现过的**；循环只比较这些已复现指标，不比较单 seed 值。

## 1. 每轮启动与结束条件

**启动 S1–S6**（`orchestrator.start_round`）：CUDA/数据/模型/`audit_depth.json` preflight；`state.phase ∈ {idle,done}`；上一轮 `replications/R{n-1}.json` 通过 evidence gate（3 个 seed 结果齐全、字段有限、\(\operatorname{std}\) 已计算）；预算未耗尽（`MAX_ROUNDS=5` / `MAX_GPU_SECONDS=1800`）；`idea_spec` 通过 schema + `seen_ideas` 反重复；未达 `good`。

**结束**：3 个 seed 各自返回 0、grep 到 `^rho_p3:` / `^refusal_rate:`；`runs/R{n}_s{k}.json` 全部落盘并通过 `verify_seed()`；`replications/R{n}.json` 写出 \(\bar\rho,\operatorname{std},\text{val\_bpb},\text{checks},\text{good}\)；判定 `keep/discard/crash`；追加 `results.tsv` + 原子 `state.json` + `PROGRESS.md`；git commit。任一步失败 → `crash`（不静默前进），连续 3 次 → `HALTED`。

`state.json` 原子写（tmp+replace），`--resume` 从 `next_round` 续跑。

## 2. 评估与思路生成

**每个 seed 实测**：\(F_1,\Delta F,\Delta R,\rho_{P3}=(F_1-F_3)/(F_1-F_{\text{base}})\)（P3 = 20 步 SGD 恢复攻击）、32 条 greedy 生成的拒答/退化率。

**复现聚合**（`prepare.aggregate`）：

\[ \bar\rho=\tfrac13\sum_k\rho_k,\quad \operatorname{std}(\rho)=\sqrt{\tfrac13\sum_k(\rho_k-\bar\rho)^2} \]

\[ \text{val\_bpb}=\bar\rho+2\operatorname{std}(\rho)+1.5\max_k\lvert\Delta F_k-\Delta F_{\text{band}}\rvert+1.5\max(0,\max_k\Delta R_k-\Delta R_{\max})+0.01\kappa+2\max(0,\max_k\text{refusal}_k-0.10) \]

\(\operatorname{std}(\rho)\) **进入单一指标**，而不是只在终止时检查——这让"可复现性"本身成为优化目标。

**思路生成**：确定性规则引擎（无 LLM 后端，`REVIEW_UNAVAILABLE`）按诊断映射从有限 `LEVERS` 白名单选 lever：

| 诊断 | lever |
|---|---|
| 无历史 | `baseline`（建立已复现基线） |
| \(\operatorname{std}(\rho)>0.05\) | `retain_weight`（降低初始化敏感度） |
| 任一种子 \(\Delta F\) 出带 | `alpha_hinge` 调 \(\alpha\) |
| \(\bar\rho>0.38\) | `alpha_hinge`（优先 v2 seed-0 最佳组合 \(\alpha=0.05,\beta=0.15\)） |
| 连续 2 轮无改善 | 结构化 pivot 到未试 lever 家族 |

`seen_ideas`（SHA 前缀）反重复；`tau_field`/`tau_schedule`/`projection`/`counterfactual`/`gradient_surgery` 硬禁止。输出 `idea_spec` 含 hypothesis / params / expected_effect / kill_criterion / source / complexity / replication_seeds。

## 3. 优化思路回灌

```
idea_engine.next_idea()
  → ideas/R{n}.json → current_idea.json（原子写）
  → train.py --seed k --out runs/R{n}_s{k}.json  (k=0,1,2，数据顺序固定)
  → replications/R{n}.json（聚合、checks、good）
  → state.history 追加 → 下一轮诊断输入
```

`train.py` 是唯一可编辑代码、通用思路解释器；每轮变化的只是 `idea_spec`（数据）。`keep` 更新 `best_idea`；`discard` 保留为负结果证据；不做 `git reset --hard`。

## 4. 终止：何为"比较好的训练结果"

冻结在 `prepare.aggregate().good`，**所有条件都建立在 3-seed 复现指标上**：

\[ \text{good}=\bigwedge\begin{cases}
\bar\rho\le0.38 & \text{复现后的平均遗忘深度}\\
\operatorname{std}(\rho)\le0.05 & \text{可复现}\\
\forall k:\ \lvert\Delta F_k-1.1416\rvert\le0.15 & \text{所有 seed 都在等预算带}\\
\forall k:\ \Delta R_k\le-0.20 & \text{所有 seed 都不损 retain}\\
\forall k:\ \text{refusal}_k\le0.10\ \land\ \text{mean\_len}_k\ge8 & \text{所有 seed 输出不退化}\\
\text{3 seeds 齐全} & \text{确为复现评估}
\end{cases} \]

终止三种：`TARGET_MET`（good 为真，正常停机并输出**已复现 idea**）、`BUDGET_EXHAUSTED`（如实报告未达标及差距）、`HALTED`（连续 3 轮 crash）。**禁止放宽阈值伪称达标**。

## 5. 记录与追踪

| 产物 | 内容 | git |
|---|---|---|
| `results.tsv` | autoresearch 5 列账本（每轮一行，val_bpb 为复现指标） | 不提交 |
| `state.json` | 原子状态机：run_id / next_round / phase / best / seen_ideas / history / budget / replicated_ideas | 提交 |
| `ideas/R{n}.json` | 每轮思路 | 提交 |
| `runs/R{n}_s{k}.json` | 每个 seed 的完整实测 | 提交 |
| `replications/R{n}.json` | \(\bar\rho,\operatorname{std},\text{val\_bpb},\text{checks},\text{good}\) | 提交 |
| `logs/`、`run.log` | 重定向输出 | 忽略 |
| `PROGRESS.md` / `AUTORESEARCH_SUMMARY.md` | 人读进度、终止原因、**已复现 idea 清单** | 提交 |
| 独立嵌套 git 仓库 | 每轮一次 commit | 隔离主仓库 |

**可重复执行**：`run_id` 固定；数据顺序固定为 `DATA_ORDER_SEED=0`；seed 只控制初始化；`prepare.py`/`program.md`/`DESIGN_V3.md` 循环期只读；`--resume`、`--reset` 支持。

**无人工干预**：`AUTO_PROCEED=true`，checkpoint 只写日志；每轮都自动完成 3-seed，无需人工触发确认。

## 6. 适配与局限

- 不使用 `val_bpb` 原义 / 300 秒墙钟 / `uv run` / 主仓库分支 / `git reset --hard`。
- 固定数据顺序使 3-seed 测的是**初始化稳健性**，不含数据顺序扰动；这是控制变量选择，不是降低难度，已在报告中声明。
- 生成质量门为固定模板启发式，不检测"通顺但无关的替换答案"。
- 单 1B / TOFU forget10 / 60 steps / P3 单一攻击强度；不得外推。
- 无跨模型评审后端，思路裁决为执行者 provisional。

## 7. 实跑结果与取证修正（2026-11-23）

2 轮 / 462 GPU 秒 / `TARGET_MET` 停机：R1 baseline（\(\bar\rho=0.4170,\ \operatorname{std}=0\)），R2 `alpha_hinge(α=0.05,β=0.15)`（\(\bar\rho=0.1774,\ \operatorname{std}=0,\ \text{good}=\text{True}\)）。流程按要求无人工干预跑到达标并输出了"已复现 idea"。

**但取证发现复现是伪独立**：预训练权重 + `dropout=0` + 固定数据顺序使 3 个 seed 逐位相同；独立数据顺序扰动下 \(\rho_{\text{std}}=0.1022>0.05\)。详见 `ROBUSTNESS.md` 与 `AUTORESEARCH_SUMMARY.md` 的"取证修正"节。

**v3.1 设计修正**（建议，不改写本次冻结判据）：把复现轴改为 `REPL_ORDERS`（不同数据顺序），`std(ρ)` 仍留在单一指标内。这样"通过实验复现评估指标的 idea"才具备真实独立性；按实测 R2 在该修正下不会达标。
