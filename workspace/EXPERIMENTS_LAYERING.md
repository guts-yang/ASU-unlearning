# 六个实验目录的分层索引（精简分层处理）

**日期**：2026-11-27　**范围**：`workspace/auto_train_loop{, _v2, _v3, _v4, _v5}` + `workspace/auto_experiments`
**本文件的作用**：把 6 个目录的内容按价值密度分层，让后续查阅不必读全文；并提供可按需触发的物理精简命令。**本轮未删除、未移动、未修改任何原有文件。**

## 0. 为什么这样分层（关键事实）

- **体量极小**：五个 loop 合计约 400KB，`auto_experiments` 2.3MB；共 194 个文件全部已被 git 跟踪，不是临时垃圾。
- **无逐字节冗余**：用 md5 比对五个版本的所有非 ignore 文件，**没有任何一对跨版本文件相同**。每版脚本都承载了该轮的实质改动（`v1` 退化路径拒绝 → `v2` 3-seed → `v3` 复现指标 → `v4` REPL_ORDERS → `v5` EMA），是方法论演进证据，删旧版 = 丢历史。
- **互引路径**：`DESIGN_V2` 引 v1、`DESIGN_V3` 引 v1/v2、`DESIGN_V4/V5` 引 `v3/replications/R2.json`，`ROBUSTNESS.md`/`COMBINATION_PROBE.md` 被多份文档引用。**目录名不可随意改**。
- 因此本索引采取"逻辑分层 + 可选物理精简"，不做破坏性删除。

## 1. 三层总表

| 层 | 定义 | 内容 | 处置 |
|---|---|---|---|
| **L1 热层（必留，信息核心）** | 结论、判定、可复现账本 | 全部 `.md` 文档；所有 `state.json`；所有 `program.md`；`gate_*.json`、`audit_depth*.json`、`diagnostic_methods_self_test.json`、`full_repro_bench.json`；所有 `summary.json`；`screening/`、`nonclaim/`、`formal_training/` 下全部小 JSON；`full_repro_v1/` 下判定 JSON | 原样保留 |
| **L2 温层（保留，重建脚本）** | 产生结论的代码与取证程序 | 每个 loop 的 `orchestrator.py`、`idea_engine.py`、`train.py`、`prepare.py`；`v3/robustness_check.py`、`v3/train_loop_snapshot.py`、`v5/combination_probe.py`；`auto_experiments/*.py` | 原样保留（各版独有，勿合并删） |
| **L3 冷层（可压缩候选）** | 由 L1 摘要覆盖、访问价值低的逐 step 原始日志 | `auto_experiments/results/cf04_300/*/CF-04.json` 中 >50KB 的逐 step 大文件（约 16 个，~1.2MB）；其 delta 已完整落在 `cf04_300/summary.json` | 仅在确认后按第 4 节归档 |

## 2. 各目录一句话角色

- `auto_train_loop` (v1)：退化路径拒绝——"靠忘得更狠换低 rho"被正确 discard，`BUDGET_EXHAUSTED`。
- `auto_train_loop_v2`：3-seed 确认上线——揭穿单 seed 最优是随机性（R4 std=0.1022）。
- `auto_train_loop_v3`：报 `TARGET_MET=True` 又被 `ROBUSTNESS.md` 自己证伪（伪独立，std=0 只是确定性重跑）——**全链诚信核心，绝不可删**。
- `auto_train_loop_v4`：独立数据顺序复现轴 + "不可能三角"确立（`BUDGET_EXHAUSTED`）。
- `auto_train_loop_v5`：EMA 稳定化逼近三角但未打破 + `COMBINATION_PROBE.md` 组合取证。
- `auto_experiments`：真实 1B/TOFU 实验库——3 gate + audit_depth 全 PASS、CF-04/v2 双探针 pass、full_repro 14 方法；**全部 `claim_level=false`**。

## 3. 查阅默认值

- 问"实验结论/是否通过" → 读各 `AUTORESEARCH_SUMMARY.md` + `PROGRESS.md` + `auto_experiments/results/*/summary.json` + gate JSON + `formal_training/`、`screening/`（已在 L1）。
- 问"怎么复现/代码怎么 evolution" → 才读 L2 脚本，且要按版本读，勿合并。
- 问"逐 step 训练曲线" → 才碰 L3 大 JSON；其余情况 `summary.json` 的 delta 已够。

## 4. 可选的物理精简（三档，均需你确认后执行）

> 所有删除都可从 git 恢复（`git checkout HEAD -- <path>`）；执行前请确认工作区干净。本索引**不替你执行**。

**档 1 · 零删除（推荐）**：只保留本索引。收益是"以后不用读全文"，成本 = 一个新文件。适合当前（总量 <3MB，无空间压力）。

**档 2 · 仅压缩 L3 逐 step 大日志**（保留 summary，可恢复）：
```
cd /Users/chenyliao/Repositories/Machine_Unlearning/ASU-unlearning
find workspace/auto_experiments/results/cf04_300 -name CF-04.json -size +50k
# 核对上一步列出的文件后，确认再删；summary.json 仍保留 delta/超参/路径
# 归档而非直删（可恢复）：
mkdir -p workspace/auto_experiments/_archive/cf04_300_raw
find workspace/auto_experiments/results/cf04_300 -name CF-04.json -size +50k -exec mv {} workspace/auto_experiments/_archive/cf04_300_raw/ \;
```
代价：逐 step loss 曲线需从 git 或 archive 取回；summary.json 与 gate 结论不受影响。

**档 3 · 不推荐**：删旧版 loop 脚本。五个版本无冗余、互引路径多，删了会破坏方法论链与文档引用，收益（约 300KB）远小于信息损失。除非做正式归档（整体移入 `archive/` 并重写引用），否则不要做。

## 5. 安全边界

- 所有结果 `claim_level=false`、`gate_status=NOT_APPLIED`：L1 里的"通过"是诊断/gate/筛选级判定，不是投稿级结论；精简不应改变这一标注。
- 本轮未改任何 git 跟踪文件、未删数据、未 push；仅新增本索引文件。
