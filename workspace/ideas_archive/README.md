# ideas 归档索引

**归档时间**：2026-11-24  **方式**：`copy2` 快照  **文件数**：299  **总字节**：1,002,909

> 原文件**全部保留在原路径**，未移动、未删除、未改动；本目录是副本汇总。
> 目的是避免破坏脚本中的相对路径引用（例如 `HERE.parent / "idea-stage"`）与各 `loop/` 下的独立嵌套 git 仓库。
> 未触碰 `.codebuddy/`、`.git/`、`__pycache__/`、`checkpoints/`。

## 目录索引

| 目录 | 文件数 | 内容 |
|---|---|---|
| `01_reports/` | 11 | V1/V2/V3/V4 idea 报告、叙事报告与 HTML 阅读视图 |
| `02_idea_cards/` | 8 | research-wiki 的 idea 卡片与反重复记忆 log |
| `03_design_contracts/` | 17 | 各版 DESIGN/program、研究契约、故事设计、实验计划/追踪/结果 |
| `04_loop_ideas/` | 30 | 五个自循环每轮生成的 idea_spec（hypothesis / lever / params / kill criterion / source） |
| `05_results_ledgers/` | 134 | state.json、results.tsv、PROGRESS、AUTORESEARCH_SUMMARY、runs/、replications/、rounds/、gate 实测 JSON |
| `06_audit_forensics/` | 37 | anti-autoresearch 两个 PAPER_DIR 的 claims/REPORT/findings/coverage，以及 ROBUSTNESS / COMBINATION_PROBE / MATCHED_DF_CONTROL / AUDIT_FEEDBACK / AUDIT_COMPARISON |
| `07_verification/` | 33 | 独立 7/7 + 串联 11/11 复现验证的日志、账本与结论 |
| `08_code_snapshot/` | 29 | 产生上述结果的最小代码快照（train.py / prepare.py / idea_engine.py / orchestrator.py 等） |

每类下按**来源目录名**再分子目录（如 `04_loop_ideas/auto_train_loop_v4/R1.json`），避免同名文件互相覆盖。
`MANIFEST.json` 逐文件记录 `source / archived / sha256 / size_bytes / category`，可用于完整性校验。

## 当前结论（归档时的状态）

**判定口径**：以 v4/v5 冻结的 `prepare.aggregate().good` 为准，评估指标必须在 **3 个独立数据顺序**上复现：
$\bar\rho\le0.38$、$\operatorname{std}(\rho)\le0.05$、所有 order $|\Delta F-1.1416|\le0.15$、$\Delta R\le-0.20$、拒答率 $\le0.10$。

- **当前所有候选 `good=False`，无一达标**。v4（5 轮）+ v5（5 轮）+ v5 事后组合探针（3 个）共 13 个候选，评估指标全部经过独立实验复现，`BUDGET_EXHAUSTED` 如实停机，阈值全程未放宽。
- 13/13 个候选都卡在 `all_df_in_band`；低 $\barho$ 与低 std 已被 EMA 同时解决（v5 R3：0.1412 / 0.0354），但 $\Delta F$ 塌缩出带。
- 最优候选是事后探针 `EMA0.95 + β=0.20`（val_bpb 0.5685，$\barho=0.1260$、std=0.0263，仅差 `worst_df=0.2532>0.15`）；循环内最优是 v5 R2（EMA 0.90，0.8213）。
- **v3 的 `TARGET_MET`（R2 `good=True`）已撤回**：3 个 seed 逐位相同（预训练权重 + `dropout=0` + 固定数据顺序），属确定性重跑而非独立复现；独立数据顺序扰动后 $\operatorname{std}=0.1022>0.05$。
- idea 层面：R3-2 深度审计是唯一稳固的正面贡献；R2-B τ 课程被等预算对照推翻；R2-C meta 目标未实现；R2-A/R3-1/R3-3 降级；V1 全系淘汰。
- 流程侧：复现验证独立 7/7、串联 11/11 `PASS`；五个自循环共 30 轮 0 crash，`--resume` 行为正确，跨循环 $\rho$ 逐位一致。
- 诚信取证：v1/v2 论文目录 verdict 均为 **`REVIEW_UNAVAILABLE`（L2）**；10 个语义/代码级维度全部 `review_unavailable`，结构性阻断无罪判定，0 finding 不等于通过。

## 口径提醒

v1/v2/v3 的 `val_bpb` 不含 $2\operatorname{std}(\rho)$ 项，v4/v5/探针含该项，**跨版本绝对值不可直接比较**；`05_results_ledgers/` 内各版本账本请按版本分别阅读。

## 完整性校验

```bash
cd /usr/local/workspace/ASU-unlearning-Auto
python3 - <<'PY'
import json, hashlib
from pathlib import Path
m = json.loads(Path("workspace/ideas_archive/MANIFEST.json").read_text())
bad = [f["archived"] for f in m["files"]
       if hashlib.sha256(Path(f["archived"]).read_bytes()).hexdigest() != f["sha256"]]
print("files:", m["file_count"], "mismatch:", len(bad), bad[:5])
PY
```

## 未纳入归档的内容（仍在原路径）

| 内容 | 原因 |
|---|---|
| `workspace/auto_experiments/checkpoints/` | 模型权重，体积大且非 idea 文档 |
| 各 `logs/*.log`、`driver_console.log`、`run.log` | 原始运行日志；摘要已在 `VERIFICATION_LOG.md` / `AUTORESEARCH_SUMMARY.md` |
| 各 `loop/.git/` | 独立嵌套 git 仓库元数据（commit 链仍可在原路径查看） |
| `current_idea.json`、`loop_state*.json` | 运行时可变状态，非归档对象 |
| `__pycache__/`、`.aris/` | 缓存与轨迹，非 idea 产物 |

> 归档不改动任何原始结论文件；`MANIFEST.json` 只列被复制的源文件，本 README 为索引本身。
