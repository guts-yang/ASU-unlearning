# RelResidual 验证与自主训练闭环 v6

**日期**：2026-11-24　**技能**：`/idea-discovery` + `/autoresearch-loop`
**目录**：`workspace/relresidual_loop/`

## 0. Problem Anchor 与诚信前提

**要验证的 idea**：用「替换主体后的反事实 Teacher 轨迹」代替 ASU 的注意力温度目标。同一关系问题构造 \(X^+\)（遗忘主体 \(s_f\)）与 \(X^-\)（仅换为保留主体 \(s_r\)，其他不变、token 等长）；冻结 Teacher 前向得各层 \(A^+\!/A^-\)；Student 接收 \(X^+\)，用 \(L_F\) 靠近 \(T_\ell=A_\ell(s_f,r)-\alpha B_\ell\)，用 DKT 式 \(L_{KL}\) 对齐反事实 Teacher 输出分布（共享同一输出前缀），再加 retain NLL \(L_R\)。只更新 Student，部署不需要 Teacher/donor/运行时 patch。

\[ B_\ell=[A_\ell(s_f,r)-A_\ell(s_r,r)]-[A_\ell(s_f,r_0)-A_\ell(s_r,r_0)],\qquad T_\ell=A_\ell(s_f,r)-\alpha B_\ell \]

\(r_0\) 取 **retain 集的关系模板**（明确不删除），套到 \(s_f/s_r\) 上；激活取 **query-last**；subject-last 仅作诊断对照，**不得据此误判信号不存在**（关系在主体之后时看不到）。

**反重复检查**：V1「反事实配对 + ΔA 正交/对齐」已被既有 stage0-14 H20 3-seed 实测 KILL 并写入 `research-wiki` 反重复记忆。RelResidual 提供新的可证伪点（跨关系差分去身份成分、位置对照），故按 `idea-discovery` 作为**有条件的新假设**小规模等预算验证，不复用旧结论直接判死，也不无视旧结论。

**Gate 0 实测（已跑）**：24/24 组 token 等长四元组；query-last \(\cos(\Delta A_r,\Delta A_{r_0})=0.2745<0.30\)、\(\|B\|/\|\Delta A_r\|=0.9776\)（\(B\approx\Delta A_r\)，扣减几乎无效）；分子层 L8/L12 cos=0.114/0.144，仅 L15=0.566；subject-last cos=0.235、ratio=1.600（仅诊断）。→ **原版 RelResidual 前提不成立，判 KILL**。

该 KILL **不终结流程**，而是作为诊断输入驱动思路收缩：只用 cos 高的层、调 \(\alpha_B\)、调 \(L_F\) 形式。所有收缩变体标注为「新条件性假设」，不得声称原 idea 通过。

## 1. 每轮启动与结束条件

**启动 S0–S6**：

| # | 条件 |
|---|---|
| S0 | `gate0.json` 存在且 schema 合法；记录其 `verdict` 与逐层 cos 作为诊断输入（不因 KILL 停机） |
| S1 | `preflight()`：CUDA、模型、forget10/retain90、`common.py`、`counterfactual.py` |
| S2 | `state.phase ∈ {idle, done}`，无残留未完成轮 |
| S3 | 上一轮 `replications/R{n}.json` 的 `order_results` 恰为 3 份且逐份通过 `verify_run()` |
| S4 | 预算未耗尽（`MAX_ROUNDS=4` / `MAX_GPU_SECONDS=1800`） |
| S5 | `idea_spec` 通过 schema + `seen_ideas` 反重复；lever 在 `LEVERS` 白名单内 |
| S6 | 未达冻结的 `good` 判据 |

**结束**：3 个独立数据顺序各自返回 0、grep 到 `^dF:` / `^donor_leak:` / `^premature:` / `^truth_leak:`；`runs/R{n}_o{k}.json` 落盘且 `verify_run()` 通过；`replications/R{n}.json` 聚合出 \(\overline{\Delta F},\operatorname{std}(\Delta F)\)、最坏 donor/premature/truth、`val_bpb`、`checks`、`good`；判定 `keep/discard/crash`；追加 `results.tsv` + 原子 `state.json` + `PROGRESS.md`；git commit。任一步失败 → `crash`（不静默前进），连续 3 次 → `HALTED`。

`state.json` 原子写（tmp + replace），`--resume` 从 `next_round` 续跑；数据顺序由 `REPL_ORDERS=(0,1,2)` 显式给出。

## 2. 评估与思路生成

**每个 order 实测**（`train.py`，60 steps，与 ASU 基线等预算）：

- forget / retain 的 answer NLL → \(\Delta F,\Delta R\)；
- 32 条 greedy 生成 → **donor_leak**（生成答案中出现 \(s_r\) 的特征词）、**premature**（空/<4 词/纯拒答）、**truth_leak**（原答案关键事实词仍被生成）；
- 同时记录 P3 恢复率 \(\rho_{P3}\) 作为诊断（不计入冻结判据）。

这三个生成指标直接对应用户点名的两个伪成功模式：**最终要验证的是事实遗忘，不是改答 donor 或提前结束**。

**独立复现聚合**（`prepare.aggregate`）：

\[ \text{val\_bpb}=\max(0,\ \Delta F_{\min}-\overline{\Delta F})+2\max(0,\Delta R_{\max}^{\text{worst}}-\Delta R_{\max})+3\max(0,\text{donor}-0.10)+3\max(0,\text{prem}-0.10)+1.5\max(0,\text{truth}-0.30)+2\operatorname{std}(\Delta F)+0.01\kappa \]

\(\Delta F_{\min}=0.7998\) 是 **ASU global τ=2 的实测等预算基线**——新 idea 不得弱于已有 ref（这是 V1 全系被 KILL 的核心原因，必须写进指标）。

**思路生成**（确定性规则引擎；无 LLM 后端 → `REVIEW_UNAVAILABLE`）：输入 = Gate 0 逐层 cos + 本轮聚合指标 + `state.history`；输出 `idea_spec`（hypothesis / lever / params / kill_criterion / source / complexity / replication_orders）。

| 诊断 | lever | 依据 |
|---|---|---|
| 无历史 + Gate0 KILL | `layer_select`（仅 L15） | Gate 0 只有 L15 cos=0.566 显示跨关系共有成分 |
| \(\Delta F\) 低于基线 | `alpha_B` {0.5, 1.0, 2.0} | 调节有限位移强度 |
| donor_leak 超标 | `kl_weight` {0.05, 0.2} | 削弱反事实 Teacher 输出对齐，减少 donor 内容注入 |
| truth_leak 超标 | `lf_form` {mse, cosine} | 换激活目标形式 |
| 全项过 | `layer_weight`（cos 加权多尺度） | 保守探针 |
| 连续 2 轮无改善 | 结构化 pivot 到未试家族 | 防停滞 |

`seen_ideas` SHA 反重复；硬禁止把 \(\tau\) 空间/时间场、子空间投影等已证伪方向混入。

## 3. 优化思路如何回灌

```
gate0.json（诊断输入）
  → idea_engine.next_idea(state)
  → ideas/R{n}.json → current_idea.json（原子写）
  → train.py --data-order k --out runs/R{n}_o{k}.json   (k=0,1,2)
  → replications/R{n}.json（独立复现聚合）
  → state.history → 下一轮诊断输入
```

`train.py` 是**唯一可编辑代码**、通用思路解释器：按 `layer_select / alpha_B / kl_weight / lf_form` 显式分支构造 \(B_\ell,T_\ell\) 与 \(L_F,L_{KL},L_R\)，未识别 lever 直接 `ValueError`。Teacher 激活/输出目标按固定 seed=0 的配对预计算并缓存，保证 3 个 order 之间只有**训练数据顺序**不同。`keep` 更新 `best_idea`；`discard` 保留为负结果证据；不做 `git reset --hard`。

## 4. 终止：何为"比较好的训练结果"

冻结在 `prepare.aggregate().good`（基于 3 个独立数据顺序）：

\[ \text{good}=\bigwedge\begin{cases}
\overline{\Delta F}\ge0.7998\ \text{且}\ \operatorname{std}(\Delta F)\le0.08 & \text{真遗忘且不弱于 ASU 基线、可复现}\\
\forall o:\ \Delta R_o\le0.05 & \text{retain 不损}\\
\forall o:\ \text{donor\_leak}_o\le0.10 & \text{不是改答 donor}\\
\forall o:\ \text{premature}_o\le0.10 & \text{不是提前结束}\\
\forall o:\ \text{truth\_leak}_o\le0.30 & \text{原事实确未输出}\\
\text{3 orders 齐全} & \text{确为独立复现}
\end{cases} \]

终止三种：`TARGET_MET`（正常停机并输出**已独立复现的 idea**）、`BUDGET_EXHAUSTED`（如实报告未达标、差距与最佳 idea）、`HALTED`（连续 3 轮 crash）。**阈值全程冻结，禁止放宽伪称达标**。

## 5. 记录与追踪

| 产物 | 内容 | git |
|---|---|---|
| `gate0.json` | 前置假设检验逐层 cos / ratio / verdict | 提交 |
| `results.tsv` | autoresearch 5 列（每轮一行，val_bpb 为独立复现指标） | **不提交** |
| `state.json` | 原子状态机：run_id / next_round / phase / best / seen_ideas / history / budget / replicated_ideas / gate0 | 提交 |
| `ideas/R{n}.json` | 每轮思路（hypothesis、kill criterion、source） | 提交 |
| `runs/R{n}_o{k}.json` | 每个独立数据顺序的完整实测 | 提交 |
| `replications/R{n}.json` | 聚合指标、逐项 checks、good | 提交 |
| `logs/`、`run.log` | 重定向输出 | 忽略 |
| `PROGRESS.md` / `AUTORESEARCH_SUMMARY.md` | 人读进度、与目标距离、**已独立复现 idea 清单** | 提交 |
| 独立嵌套 git 仓库 | 每轮一次 commit | 隔离主仓库 |

**可重复执行**：`run_id` 固定、`INIT_SEED=0`、配对构造固定、`REPL_ORDERS` 显式、`prepare.py`/`program.md`/`DESIGN.md`/`gate0.py` 循环期只读；`--resume`/`--reset`。**无人工干预**：`AUTO_PROCEED=true`，checkpoint 只写日志、不提问不等待；3-order 复现自动完成。

## 6. 适配与局限

- 不使用 `val_bpb` 原义 / 300 秒墙钟 / `uv run` / 主仓库分支 / `git reset --hard`。
- \(r_0\) 为 retain 模板套到 \(s_f\)，是反事实问题、无标注答案；只需 Teacher 激活，不需要答案。
- donor / truth leak 为词面启发式，不等价于语义级答案判定；`s_r` 常见词可能造成误报，已在结果里附样本。
- 单 1B / TOFU forget10 / 60 steps / answer-NLL；不得外推到 7B / MUSE / WMDP / MMLU。
- 无跨模型评审后端，思路与裁决均为执行者 provisional。
