# 自主训练–评估–思路优化闭环设计

**日期**：2026-11-23　**技能**：`idea-discovery`（Problem Anchor / claim→gate / 反重复记忆 / evidence gate）+ `autoresearch-loop`（单文件可编辑 / 单一指标 / 固定预算 / commit-keep-discard 账本）
**目录**：`workspace/auto_train_loop/`　**状态**：设计冻结 + 已实跑验证

> **诚信前提**：本宿主无 `.aris/reviewer.json`，`codex`/`gemini`/`grok`/`knot` 全部缺失，也没有 LLM API。因此"模型自主生成优化思路"**不是**由 LLM 生成，而是由 `idea_engine.py` 的**确定性诊断规则 + 受控策略库**生成结构化 `idea_spec`；LLM 后端做成可插拔 adapter，当前状态 `REVIEW_UNAVAILABLE`。不得把规则引擎的输出描述为"模型在思考"。

## 0. Problem Anchor（冻结）

在 ASU 注意力温度自蒸馏 setting 下（TOFU forget10 / retain90，Llama-3.2-1B，全局 \(\tau=2.25\) 等预算），目标不是把 forget NLL 做大，而是**在固定平滑预算下降低 P3 可恢复性 \(\rho_{P3}\)**，同时不损 retain、不靠增大 \(\Delta F\) 稀释分母、不产生退化输出。任何一轮都不得改动这个锚。

## 1. 每轮的启动与结束条件

### 启动（`orchestrator.py::start_round`）

必须**全部**满足，否则 `BLOCKED` 并写日志，不进入训练：

| # | 条件 | 检查方式 |
|---|---|---|
| S1 | `preflight()` 通过：CUDA 可用、模型/数据/`audit_depth.json` 存在 | `prepare.preflight()` |
| S2 | 状态机处于 `idle`，且 `state.json` 的 `phase` 非 `training`/`evaluating` 残留 | 原子状态文件 |
| S3 | 上一轮 artifact 已落盘且通过 evidence gate（`rounds/R{n-1}.json` 含全部必需字段） | `prepare.verify_round()` |
| S4 | 预算未耗尽：`round < MAX_ROUNDS` 且累计 GPU 秒 < `MAX_GPU_SECONDS` | `state.budget` |
| S5 | 本轮 `idea_spec` 已生成、**通过反重复记忆**（`hash` 不在 `seen_ideas`）且 schema 校验通过 | `idea_engine` |
| S6 | 未达到终止门控 `is_good()` | `prepare.is_good()` |

S2/S3 提供**断点可恢复**：进程被杀后重启，从 `state.json` 读 `next_round` 与 `phase`，若上轮已完成则跳过，若中途失败则重跑该轮（幂等，输出路径固定）。

### 结束（`orchestrator.py::finish_round`）

一轮**只有**在下列全部完成后才标记 `done`：

1. `train.py` 返回 0 且 stdout 含 `^val_bpb:` / `^peak_vram_mb:` / `^rho_p3:`；
2. `rounds/R{n}.json` 写出完整指标（before/after/P3/delta/idea/anchor）；
3. `prepare.verify_round()` 校验字段齐全、数值有限；
4. 单一指标 `val_bpb` 抽取并判定 `keep` / `discard` / `crash`；
5. `results.tsv` 追加一行、`state.json` 原子更新、`PROGRESS.md` 重写；
6. `git commit`（独立嵌套仓库），`keep` 时更新 `last_keep` 并保存 checkpoint 指针（不复制 4.7GB 权重，只记路径）。

任一步失败 → `crash`，写日志，**不**静默进入下一轮；下一轮仍按 S3 检查，失败轮会重跑一次，二次失败才 `BLOCKED` 停机。

## 2. 如何评估当前结果并生成优化思路

### 2.1 评估（固定，不可在循环中改）

每轮训练后测四个量（协议见 `prepare.py`）：

\[ F_1=\mathcal{L}_{\text{NLL}}(M;\mathcal{D}_f),\quad \Delta F=F_1-F_0,\quad \Delta R=R_1-R_0,\quad \rho_{P3}=\frac{F_1-F_3}{F_1-F_{\text{base}}} \]

\(F_3\) 为 20 步 SGD (\(\eta=10^{-5}\)) 在 forget 子集上恢复后的 forget NLL；\(F_{\text{base}}=0.4621\) 从 `audit_depth.json` **读取**，不硬编码。

单一指标（越低越好，对应 autoresearch 的 `val_bpb`）：

\[ \text{val\_bpb}(c)=\rho_c+1.5\max(0,\Delta R_c-\Delta R_{\max})+1.5\lvert\Delta F_c-\Delta F_{\text{band}}\rvert+0.01\kappa_c \]

第二项惩罚 retain 损伤，第三项**对称**惩罚偏离等预算带（修正 Round 2 发现的"只罚忘得少、不罚忘得多"缺陷），第四项为复杂度。

### 2.2 思路生成（Idea Engine）

输入：本轮 metrics + 完整历史 `state.history` + `seen_ideas` + 已证伪方向。输出结构化 `idea_spec`：

```json
{
  "id": "idea_R3_hinge_beta0.20",
  "round": 3,
  "hypothesis": "上一轮 rho 高于目标；提高 hinge 强度可在 matched-dF 下进一步压低 P3 恢复",
  "lever": "hinge_beta",
  "params": {"hinge_beta": 0.20, "hinge_margin": 1.2},
  "expected_effect": "rho 下降，dF 保持在 band 内",
  "kill_criterion": "若 dF 偏离 band > 0.15 或 rho 未下降则 discard",
  "source": "deterministic_rule:rho_high",
  "complexity": 2,
  "budget": {"mean_tau": 2.25, "steps": 60}
}
```

生成逻辑（`idea_engine.py::next_idea`）——按优先级匹配**确定性诊断规则**：

| 诊断（基于上一轮 + 历史） | 选中的 lever | 动作 |
|---|---|---|
| `rho > RHO_TARGET` 且 `dF` 在 band 内 | `hinge_beta` | 提高/降低 hinge 强度 |
| `rho > RHO_TARGET` 且 `dF` 低于 band | `alpha` | 提高 \(\alpha\) 增强遗忘压力 |
| `dR > DR_MAX` | `alpha` ↓ / `retain_weight` ↑ | 先保 retain |
| `abs(dF - band) > 0.15` | `hinge_margin` | 把 \(\Delta F\) 拉回预算带 |
| 连续 2 轮 `val_bpb` 无改善 | **结构化 pivot** | 换一个未试过的 lever 家族 |
| 候选 lever 已在 `seen_ideas` | 跳过 | 反重复记忆（idea-discovery 纪律） |

**受控搜索空间**：`LEVERS` 是预声明的有限枚举（每个含取值域、步长、\(\kappa\)、kill criterion），Idea Engine 只能在其笛卡尔积内选择，不能发明新代码路径。这是"自主优化思路"在无可信 LLM 时的可执行定义；它保证可重复、可审计、不会静默扩 scope。

**可插拔 LLM adapter**：若将来配置 `IDEA_LLM_ENDPOINT` 或启用跨模型评审后端，`next_idea()` 会优先调用独立线程生成 `idea_spec`，但仍须通过同一 schema 校验、反重复检查与 `LEVERS` 白名单；未配置时返回 `source=deterministic_rule:*` 并在 `PROGRESS.md` 标注 `llm_backend=REVIEW_UNAVAILABLE`。

**已证伪方向（硬禁止）**：\(\tau\) 空间场 / 时间调度（Round 1 证伪）、forget-grad 子空间投影（Round 2 证伪）、V1 反事实配对（既有实测 KILL）。

## 3. 优化后的思路如何应用回下一轮训练

衔接是**数据驱动**而非改代码：

```
idea_engine.next_idea()
   → ideas/R{n}.json（schema 校验 + 反重复）
   → orchestrator 原子写入 current_idea.json
   → train.py 启动时读取 current_idea.json
   → 按 idea_spec.lever / params 动态配置 60-step 训练
   → rounds/R{n}.json 回写 idea_id 与实测
   → state.history 追加 → 下一轮诊断输入
```

`train.py` 是**通用思路解释器**（唯一可编辑代码），每轮变化的是 `idea_spec`（数据），不是训练代码。这比 Karpathy 原版"每轮改 `train.py`"更贴合"优化训练思路"的语义，同时保留"单一可执行入口"的纪律。所有 lever 在 `train.py` 里都有显式分支，未识别 lever 直接 `ValueError` 拒绝执行，杜绝静默忽略。

`keep` 的 idea 成为下一轮的**基线锚**（`state.best_idea`）；`discard` 的 idea 进入 `seen_ideas` 与 `PROGRESS.md` 的"已证伪"表，但**保留在账本里**（负结果也是证据）。

## 4. 终止判断：何为"比较好的训练结果"

**预先冻结、循环期间不可改**（写在 `prepare.py`，不在 `train.py` 里）：

\[ \text{is\_good}(m)=\bigwedge\begin{cases}
\rho_{P3}\le 0.30 & \text{深度目标}\\
\Delta F\ge 1.00 & \text{遗忘强度下限}\\
\Delta R\le -0.20 & \text{retain 不损}\\
\lvert\Delta F-\Delta F_{\text{band}}\rvert\le 0.15 & \text{等预算带，堵分母作弊}\\
\text{3 seeds: }\operatorname{std}(\rho)\le 0.05 & \text{统计稳健}\\
\text{refusal\_rate}\le 0.10 & \text{排除退化输出}
\end{cases} \]

\(\Delta F_{\text{band}}=1.1416\) 来自 Round 1 等预算 anchor 实测。

**门控顺序**：前四条为初筛，只有初筛通过才触发 3-seed 复跑与生成质量检查，避免每轮都付 3 倍算力。第 6 条需要生成式评估，当前环境未实现 → 记为 `generation_gate=NOT_IMPLEMENTED`，因此 **`is_good()` 在本轮结构性不可能返回 True**；这必须如实记录，不能靠放宽门控假装达标。

**终止条件只有两个**：

| 条件 | 结果 |
|---|---|
| `TARGET_MET`：`is_good()` 为真 | 正常停止，写 `AUTORESEARCH_SUMMARY.md` 与最佳 idea |
| `BUDGET_EXHAUSTED`：`MAX_ROUNDS` 或 `MAX_GPU_SECONDS` 到 | **如实报告"未达标"**，输出已达最佳值、差距、失败项，不伪称成功 |

另设 fail-safe：连续 3 轮 `crash` 或 evidence gate 连续 2 轮 `BLOCKED` → `HALTED`，写清原因。

## 5. 记录与追踪

| 产物 | 内容 | 是否提交 |
|---|---|---|
| `results.tsv` | autoresearch 5 列账本（commit / val_bpb / memory_gb / status / description） | **不提交**（技能硬约束） |
| `state.json` | 可恢复状态机：`run_id`、`next_round`、`phase`、`best_idea`、`seen_ideas`、`budget`、`history` | 提交（轻量） |
| `ideas/R{n}.json` | 每轮思路（hypothesis / lever / params / kill criterion / source） | 提交 |
| `rounds/R{n}.json` | 完整实测指标 + `idea_id` + anchor | 提交 |
| `logs/R{n}.log` | 每轮重定向输出，`run.log` 为最后一轮 | 忽略 |
| `PROGRESS.md` | 人读：轮次表、最佳值、已证伪、与目标的差距 | 提交 |
| `AUTORESEARCH_SUMMARY.md` | 停止时写：终止原因、最佳 idea、verdict | 提交 |
| 嵌套 git 仓库 | 每轮一次 commit，`keep` 链单调前进 | 隔离于主仓库 |

**可重复性**：`run_id` 由方向 slug + 日期派生；所有随机源固定（torch / cuda / random / numpy，seed 由 `state.seed` 决定）；数据顺序固定；`prepare.py`/`program.md`/`DESIGN.md` 循环期间只读；`state.json` 原子写（tmp + `os.replace`）；`— resume <run_id>` 从第一个非终态阶段恢复。

**无人工干预**：`AUTO_PROCEED=true`（取自 `idea-discovery`）——checkpoint 只写日志、不提问、不等待、不结束回合；每 10 轮追加进度摘要；`feishu` 通知未配置则静默跳过。

## 6. 与既有结论的一致性

- 单一指标的 \(|\Delta F-\Delta F_{\text{band}}|\) 对称项，直接修复 `loop2/AUDIT_FEEDBACK.md` 线索 D。
- 终止门控的 3-seed 与 refusal-rate 两条，对应线索 E（排除退化输出、补统计稳健）。
- 硬禁止列表对应 `STORY_DESIGN.md` §7/§8 的三个已证伪方向。
- 未启用跨模型评审，所有思路裁决为执行者 provisional；`generation_gate` 未实现，`TARGET_MET` 不可达时必须如实报告 `BUDGET_EXHAUSTED`。
