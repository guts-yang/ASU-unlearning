# program.md — 复现验证研究组织代码（固定，不修改）

> 本文件是 `/autoresearch-loop` 的 `program.md` 等价物，描述本次任务的组织约束。
> 与标准 autoresearch 的差异（如实声明）：本任务目标是**复现验证**，不是调参优化。
> 因此不启用 `NEVER STOP` 无限循环、不修改 `train.py` 做架构搜索、不以 `val_bpb` 为指标、
> 不创建 `autoresearch/<tag>` 分支、不做 `git reset --hard` 回滚（工作区含用户既有产物）。
> 仅沿用 autoresearch 的**单一账本 `results.tsv` + commit/metric/status 记录纪律**与
> `run.log` 重定向纪律；`/idea-discovery` 沿用其证据链、来源标注与渐进式披露纪律。

## 目标

对 `workspace/idea-stage/IDEA_REPORT_V4.md` 中所有**已通过测试（PASS）的方法**逐一独立复现，
再串联跑通完整流程，验证连通性与正确性，给出整体验证结论。

## 固定文件（循环期间只读）

- `prepare.py`：环境、数据/模型路径、指标抽取、`git` 与显存采样。
- `program.md`：本文件。
- 上游 `workspace/auto_experiments/*.py`：被验证对象，不修改。

## 唯一可编辑代码

- `train.py`：验证编排器（本次不迭代修改，只执行一轮固定预算验证）。

## 单一指标（适配）

标准 autoresearch 用 `val_bpb`（越低越好）。本次用等价的 **`repro_shortfall`**（越低越好）：

\[ \text{repro\_shortfall} = \max(0,\ \text{阈值} - \text{观测量}) + \text{辅助约束违反量} \]

`0.000000` 表示该单元完全通过判定标准；>0 表示未达标。写入 `results.tsv` 的 `val_bpb` 列。

## 固定预算与停止条件

- 每个单元超时 900 秒；单次训练仍为 60 steps（与原实验一致），并非 300 秒墙钟预算。
- `max_experiments` = 独立单元数 + 串联步数；跑完即停止，写 `AUTORESEARCH_SUMMARY.md`。
- 不使用 `uv run`（本机依赖已在系统 Python 中）；统一用 `sys.executable`。

## 环境

- 1×NVIDIA H20，CUDA 版 PyTorch 2.5.1+cu124。
- 数据/模型：`/jizhicfs/chenyliao`；TOFU forget10/retain90/holdout10。
- 基线 checkpoint：`workspace/auto_experiments/checkpoints/tau_global_s0`（ASU global τ=2）。

## 验证顺序、预期输入输出与判定标准

独立阶段（每个单元单独起进程，输出写到 `independent/`，不依赖其他单元产物）：

| 顺序 | ID | 方法（脚本） | 预期输入 | 预期输出 | 判定标准 |
|---|---|---|---|---|---|
| 1 | U1 | `gate_sensitemp.py`（τ 敏感度诊断，PASS） | 基座模型 + forget10 | `gate_sensitemp.json`：`gate.verdict`, `layer_cv`, `head_cv_mean`, `top_layers` | `verdict=PASS` 且 `layer_cv>=0.25` |
| 2 | U2 | `train_tau.py --schedule global`（ASU 基线，PASS） | 基座模型 + forget10/retain90 | `train_tau_global.json`：`before/after.forget/retain` | `dF>=0.30` 且 `dR<=0.05` |
| 3 | U3 | `train_tau.py --schedule low2high`（τ 课程，PASS） | 同 U2 | `train_low2high.json` | `dF>=0.90` 且 `dR<=0.05` |
| 4 | U4 | `gate_subspace.py`（forget-grad 子空间，PASS） | ASU checkpoint + forget/retain | `gate_subspace.json`：`mean_selectivity` | `verdict=PASS` 且 `mean_selectivity>=0.05` |
| 5 | U5 | `gate_deepcompress.py`（深层 ΔA 方向 gate，PASS） | 基座模型 + 反事实配对 | `gate_deepcompress.json`：`layers`, `gate` | `verdict=PASS` 且 `len(layers)>=4` |
| 6 | U6 | `audit_depth.py`（P1/P2/P3/INT8 审计，PASS） | base + ASU checkpoint + GA checkpoint | `audit_depth.json`：`models.asu.depth_score` | `verdict=PASS` 且 `asu.mean_recovery>=0.30` |
| 7 | U7 | `run_all.summarize()`（报告生成，PASS） | `results/*.json` 全部结果 | `IDEA_REPORT_V4.md` | 文件存在且含 7 个 idea 的决策行 |

串联阶段（`chain/`，按依赖顺序，前一步输出即后一步输入）：

| 顺序 | 步骤 | 依赖输入 | 输出 | 判定标准 |
|---|---|---|---|---|
| C1 | gate_sensitemp | 无 | `01_gate_sensitemp.json` | 同 U1 |
| C2 | train global + `--save` | 无 | `02_train_global.json` + 新 checkpoint | 同 U2，且 checkpoint 落盘 |
| C3 | gate_subspace | C2 checkpoint | `03_gate_subspace.json` | 同 U4，且 checkpoint 存在 |
| C4 | train high2low（对照） | 无 | `04_train_high2low.json` | 进程成功 + JSON 可解析 |
| C5 | train low2high | 无 | `05_train_low2high.json` | 同 U3 |
| C6 | train global τ=2.607（等平均对照） | 无 | `06_train_equal.json` | 进程成功 + JSON 可解析 |
| C7 | train sens | C1 敏感度矩阵 | `07_train_sens.json` | 进程成功 + JSON 可解析 |
| C8 | train sens_shuffled（打乱场对照） | C1 敏感度矩阵 | `08_train_shuffled.json` | 进程成功 + JSON 可解析 |
| C9 | gate_deepcompress | 无 | `09_gate_deepcompress.json` | 同 U5 |
| C10 | audit_depth | C2 checkpoint | `10_audit_depth.json` | 同 U6，且 `models.asu` 存在 |
| C11 | 汇总生成 + 决策一致性比对 | C1–C10 全部结果 | `IDEA_REPORT_V4_CHAIN.md` | 决策表与规范版 `IDEA_REPORT_V4.md` 完全一致；独立/串联数值字段在 1e-3 容差内一致（忽略 checkpoint 路径与 run name 元数据） |

## 不计入「已通过测试的方法」但需在串联中出现的对照

- `train high2low`（KILL，弱于 global）、`train global τ=2.607`（等预算对照）、
  `train sens`（PARTIAL）、`train sens_shuffled`（打乱场对照）。
- `R2-C relearning-aware meta 目标`：**尚未实现**，因此不可复现，标注 `NOT_APPLICABLE`，不算失败也不算通过。

## 整体验证通过条件

1. 独立阶段 7/7 单元 `keep`；
2. 串联阶段 11/11 步通过且所有依赖检查通过；
3. 串联报告与规范版 V4 的决策表完全一致；
4. 无 `crash`，`results.tsv` 中所有 `val_bpb=0.000000`。

任一条不满足 → `OVERALL: FAIL`，并在 `AUTORESEARCH_SUMMARY.md` 列出具体失败项。

## 诚实性约束

- 跨模型评审后端仍不可用，本验证只检验**可复现性/连通性**，不检验新颖性或学术正确性。
- 所有数字来自实际运行，不写死期望值；日志保留在 `logs/`。
- `results.tsv` 保持 untracked，不提交。
