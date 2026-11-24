# program.md — 自主训练闭环（固定，循环中不可修改）

完整设计见 `DESIGN.md`。本文件只固定执行纪律。

1. **只读**：`prepare.py`（含 `is_good` 终止门控、`val_bpb` 单一指标、预算常量）、`program.md`、`DESIGN.md`、`idea_engine.py` 的 `LEVERS` 白名单、上游 `common.py` 与全部既有结果。
2. **唯一可编辑代码**：`train.py`（通用思路解释器）。每轮变化的是 `current_idea.json`（数据），不是训练代码。
3. **固定预算**：60 steps、\(\tau=2.25\)、\(\alpha=0.1\)、\(\eta=5\times10^{-6}\)、seed 0、P3 20 步 SGD。
4. **单一指标**：`val_bpb`（`prepare.val_bpb`），越低越好。
5. **终止**：`TARGET_MET`（`prepare.is_good()`）或 `BUDGET_EXHAUSTED`（`MAX_ROUNDS=8` / `MAX_GPU_SECONDS=900`）。连续 3 轮 crash → `HALTED`。预算耗尽必须如实报告"未达标"。
6. **AUTO_PROCEED=true**：checkpoint 只写日志，不提问、不等待、不结束回合；无人工干预。
7. **日志**：`python train.py > logs/R{n}.log 2>&1`，只 grep `^val_bpb:` / `^peak_vram_mb:` / `^rho_p3:`。
8. **账本**：`results.tsv` 5 列、保持 untracked；`state.json` 原子写、可 resume；每轮一次 git commit（独立嵌套仓库）。
9. **禁止 `git reset --hard`**：主工作区含用户产物；`discard` 只更新 `last_keep` 指针，不改文件。
10. **不使用** `val_bpb` 原义 / 300 秒墙钟 / `uv run` / 主仓库分支。无跨模型评审后端，思路生成走确定性规则，标注 `llm_backend=REVIEW_UNAVAILABLE`。
