# 自主训练–评估–思路优化闭环 v2（可真正达标终止）

**日期**：2026-11-23　**技能**：`idea-discovery` + `autoresearch-loop`
**基线**：`workspace/auto_train_loop/`（v1，`BUDGET_EXHAUSTED`，best 0.4206）
**目录**：`workspace/auto_train_loop_v2/`

## 为什么需要 v2

v1 已实现"训练→评估→生成思路→回灌→循环→记录"，但 `generation_gate` 标为 `NOT_IMPLEMENTED` 且恒为 False，使 `is_good()` **结构性不可返回 True**，因此流程只能预算耗尽停止，不满足"直至达到预设较优效果"的要求。v2 补齐三个缺口：

| v1 缺口 | v2 修复 |
|---|---|
| `generation_gate` 未实现 → TARGET_MET 不可达 | 实跑 greedy 生成，计算拒答/退化率，写进每轮结果 |
| 3-seed 确认未实现 | 初筛通过后自动触发 3 个 seed 重跑，计算 \(\operatorname{std}(\rho)\) |
| lever 只有单变量，降 \(\rho\) 必然撑破 \(\Delta F\) 预算带 | 新增 `alpha_hinge` 组合 lever：降 \(\alpha\) 把 \(\Delta F\) 拉回带内、保留 hinge 的抗恢复效应 |

**诚信前提不变**：无 `.aris/reviewer.json`，`codex`/`gemini`/`grok`/`knot` 全缺失，无 LLM API。思路生成仍是**确定性规则引擎 + 有限策略白名单**，LLM 为可插拔 adapter、状态 `REVIEW_UNAVAILABLE`。

## 0. Problem Anchor（冻结）

TOFU forget10 / retain90，Llama-3.2-1B，全局 \(\tau=2.25\) 等预算，60 steps、\(\alpha=0.1\)、\(\eta=5\times10^{-6}\)。目标：在**不增大 \(\Delta F\)**、不损 retain、不产生退化输出的前提下降低 P3 可恢复性 \(\rho_{P3}\)。

## 1. 每轮启动与结束条件

**启动 S1–S6**（`orchestrator.start_round`，全部满足才进入训练）：

| # | 条件 |
|---|---|
| S1 | `preflight()`：CUDA、模型、数据、`audit_depth.json`、`common.py` |
| S2 | `state.phase ∈ {idle, done}`，无未完成的残留轮 |
| S3 | 上一轮 `rounds/R{n-1}.json` 通过 `verify_round()` evidence gate（字段齐全、数值有限） |
| S4 | 预算未耗尽：`rounds < MAX_ROUNDS=10` 且 `gpu_seconds < MAX_GPU_SECONDS=1200` |
| S5 | `idea_engine.next_idea()` 产出 schema 合法、不在 `seen_ideas` 的 `idea_spec` |
| S6 | 未达 `is_good()`（含 3-seed 确认结果） |

S2/S3 提供断点恢复；`state.json` 原子写（tmp + `replace`），`--resume <run_id>` 从 `next_round` 续跑。

**结束**（`finish_round`，六项全完成才标 `done`）：`train.py` 返回 0 且 grep 到 `^val_bpb:` / `^rho_p3:` / `^refusal_rate:`；`rounds/R{n}.json` 含 before/after/P3/generation/idea_id；`verify_round()` 通过；判定 `keep/discard/crash`；追加 `results.tsv` + 原子 `state.json` + `PROGRESS.md`；git commit。任一步失败 → `crash`，不静默进入下一轮，连续 3 次 → `HALTED`。

## 2. 评估与思路生成

### 2.1 固定评估协议

\[ F_1=\mathcal{L}_{\text{NLL}}(M;\mathcal{D}_f),\quad \Delta F=F_1-F_0,\quad \Delta R=R_1-R_0,\quad \rho_{P3}=\frac{F_1-F_3}{F_1-F_{\text{base}}} \]

\(F_3\)：20 步 SGD（\(\eta=10^{-5}\)）在 forget 子集上恢复后的 forget NLL；\(F_{\text{base}}=0.4621\) 从实测读取，不硬编码。

**生成质量门（v2 新增，实跑）**：对 forget 前 32 条做 greedy 生成（`max_new_tokens=64`），逐条判定退化：

\[ \text{degenerate}(y)=\mathbb{1}\big[|y|<3\ \lor\ \text{含拒答模板}\ \lor\ \text{unique}(y)/|y|<0.3\ \lor\ |y|>200\big] \]

\[ \text{refusal\_rate}=\frac{1}{32}\sum_i \text{degenerate}(y_i),\qquad \text{mean\_len}=\frac{1}{32}\sum_i |y_i| \]

拒答模板为固定字符串表（"I don't know" / "I cannot" / "I'm sorry" / "As an AI" / "no information"）。同时记录 `eos_rate`。这是启发式，不能替代人工判答，已在局限里声明。

**单一指标**（越低越好）：

\[ \text{val\_bpb}=\rho+1.5\max(0,\Delta R-\Delta R_{\max})+1.5\lvert\Delta F-\Delta F_{\text{band}}\rvert+0.01\kappa+2\max(0,\text{refusal\_rate}-0.10) \]

\(\Delta F_{\text{band}}=1.141628\)（v1/loop2 实测等预算 anchor）。

### 2.2 思路生成（Idea Engine）

输入：本轮 metrics + `state.history` + `seen_ideas` + 硬禁止列表；输出结构化 `idea_spec`（id / hypothesis / lever / params / expected_effect / kill_criterion / source / complexity / budget）。

**有限 lever 空间**（只能从中选，不能发明新代码路径）：

| lever | 取值 | \(\kappa\) | 用途 |
|---|---|---|---|
| `baseline` | 1 组 | 0 | 等预算锚 |
| `alpha_hinge` | \(\alpha\in\{0.05,0.07\}\times\beta\in\{0.05,0.10,0.15\}\) | 2 | **核心**：降 \(\alpha\) 抵消 hinge 造成的 \(\Delta F\) 膨胀 |
| `hinge_beta` | \(\{0.05,0.10,0.20\}\) | 2 | 单变量抗恢复 |
| `hinge_margin` | \(\{1.0,1.2,1.5\}\) | 2 | 调预算带 |
| `alpha` | \(\{0.07,0.15,0.20\}\) | 0 | 遗忘压力 |
| `retain_weight` | \(\{1.5,2.0\}\) | 0 | 保 retain |

**诊断规则**（确定性、按优先级）：

| 诊断 | lever |
|---|---|
| 无历史 | `baseline` |
| \(\rho>\rho_{\text{target}}\) 且 \(\Delta F\) 在带内 | `alpha_hinge`（优先取未试过的 \(\alpha\downarrow\) 组合） |
| \(\lvert\Delta F-\text{band}\rvert>0.15\) | `hinge_margin` / 更低的 `alpha_hinge` |
| \(\Delta R>-0.20\) | `retain_weight` |
| \(\rho>\rho_{\text{target}}\) 且 \(\Delta F<\text{band}\) | `alpha` |
| 连续 2 轮 `val_bpb` 无改善 | 结构化 pivot 到未试过 lever 家族 |
| 候选 hash 在 `seen_ideas` | 跳过（反重复记忆） |

**硬禁止**（前三轮已证伪）：`tau_field`、`tau_schedule`、`projection`、`counterfactual`、`gradient_surgery`。

**LLM adapter**：`IDEA_LLM_ENDPOINT` 未配置 → `REVIEW_UNAVAILABLE`；若配置则须经同一 schema 校验、`LEVERS` 白名单与反重复检查后才会被采用。

## 3. 优化思路如何回灌下一轮

```
idea_engine.next_idea()
  → ideas/R{n}.json（schema + 反重复校验）
  → current_idea.json（原子写）
  → train.py 读取并按 lever/params 配置训练 + 评估 + 生成质量
  → rounds/R{n}.json 回写 idea_id、实测、target_checks
  → state.history 追加 → 下一轮诊断输入
  → keep 成为 best_idea 基线锚；discard 保留为负结果证据
```

`train.py` 是**通用思路解释器**（唯一可编辑代码），每轮变化的是 `idea_spec`（数据）而非训练代码；未识别 lever 直接 `ValueError` 拒绝，杜绝静默忽略。`keep` 只更新 `last_keep` 指针，不做 `git reset --hard`（主工作区含用户产物）。

## 4. 终止：何为"比较好的训练结果"

冻结在 `prepare.py`，循环期间不可改。定义为**相对基线的帕累托改进 + 硬约束**，不是放宽容差：

\[ \text{is\_good}(m)=\bigwedge\begin{cases}
\rho\le 0.38 & \text{比基线 }0.4206\text{ 低至少 }0.04\\
\Delta F\in[1.00,1.30]\ \ (\lvert\Delta F-1.1416\rvert\le0.15) & \text{等预算带，堵分母作弊}\\
\Delta R\le-0.20 & \text{retain 不损}\\
\text{refusal\_rate}\le0.10\ \land\ \text{mean\_len}\ge8 & \text{输出不退化（v2 已实现）}\\
\text{3 seeds: }\operatorname{std}(\rho)\le0.05 & \text{统计稳健（v2 已实现）}
\end{cases} \]

**门控顺序**：前四条初筛通过才触发 3-seed 确认轮（seed 0/1/2 各跑一次，同配置），避免每轮付 3 倍算力。确认轮结果写入 `confirmations/`，\(\operatorname{std}(\rho)\) 达标才 `TARGET_MET`。

**终止只有三种**：`TARGET_MET`（正常达标停机）、`BUDGET_EXHAUSTED`（如实报告未达标 + 给出差距与最佳值）、`HALTED`（连续 3 轮 crash）。绝不通过放宽阈值伪称达标。

## 5. 记录与追踪

| 产物 | 内容 | git |
|---|---|---|
| `results.tsv` | autoresearch 5 列账本 | **不提交** |
| `state.json` | 原子状态机：run_id / next_round / phase / best / seen_ideas / history / budget / confirmations | 提交 |
| `ideas/R{n}.json` | 每轮思路（含 hypothesis、kill criterion、source） | 提交 |
| `rounds/R{n}.json` | 完整指标 + generation + target_checks + idea_id | 提交 |
| `confirmations/R{n}_seeds.json` | 3-seed 确认轮结果 | 提交 |
| `logs/R{n}.log`、`run.log` | 重定向输出 | 忽略 |
| `PROGRESS.md` | 人读轮次表、与目标距离、已证伪方向 | 提交 |
| `AUTORESEARCH_SUMMARY.md` | 终止原因、最佳 idea、verdict | 提交 |
| 独立嵌套 git 仓库 | 每轮一次 commit | 隔离主仓库 |

**可重复执行**：`run_id` 由方向 slug + 日期派生；所有随机源固定（torch/cuda/random/numpy，seed 来自 `state.seed`）；数据顺序固定；`prepare.py`/`program.md`/`DESIGN_V2.md` 循环期只读；`--resume` 从第一个非终态恢复；`--reset` 干净重跑。

**无人工干预**：`AUTO_PROCEED=true`（idea-discovery 规则）——checkpoint 只写日志、不提问、不等待、不结束回合；每 3 轮写进度摘要；Feishu 未配置则静默跳过。

## 6. 适配与局限

- 不使用 `val_bpb` 原义 / 300 秒墙钟 / `uv run` / 主仓库分支 / `git reset --hard`。
- 拒答判定为固定模板启发式，不等价于人工答案质量评估；不检测"换成无关但通顺的答案"。
- 单 1B 模型、TOFU forget10、answer-NLL 口径、P3 仅一种攻击强度与优化器；不得外推。
- 无跨模型评审后端，思路裁决为执行者 provisional。

## 7. 实跑结果（2026-11-23）

10 轮 / 1127 GPU 秒 / 0 crash / 2 次 3-seed 确认 / `BUDGET_EXHAUSTED`，best 单 seed `0.204737`（R4，未确认）。R4、R7 初筛全过并触发确认，但 \(\operatorname{std}(\rho)\) 分别为 0.1022、0.1379，均超过 0.05 → 未判 `TARGET_MET`。

**与 v1 的对比结论**：v1 是"结构上不可能达标"；v2 是"走到了达标门口，被统计稳健门控挡回"。后者证明终止判据现在是实证可判定的，同时揭示了本 setting 下单 seed 结果不可复现这一实质性科学问题。详见 `AUTORESEARCH_SUMMARY.md`。
