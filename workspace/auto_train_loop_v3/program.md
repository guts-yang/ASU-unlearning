# program.md — 自主训练闭环 v3（固定，循环中不可修改）

完整设计见 `DESIGN_V3.md`。本文件只固定执行纪律。

1. **只读**：`prepare.py`（含复现聚合 `aggregate`、终止判据、预算常量）、`program.md`、`DESIGN_V3.md`、`idea_engine.py` 的 `LEVERS` 白名单、上游 `common.py` 与既有结果。
2. **唯一可编辑代码**：`train.py`（通用思路解释器，一次调用 = 一个 seed）。每轮变化的是 `current_idea.json`（数据）。
3. **固定预算**：60 steps、\(\tau=2.25\)、\(\eta=5\times10^{-6}\)、P3 20 步 SGD、32 条 greedy 生成；**每个 idea 固定 3 seeds**（0/1/2），数据顺序固定为 `DATA_ORDER_SEED=0`。
4. **单一指标**：`val_bpb = rho_mean + 2*rho_std + 最坏约束惩罚`，越低越好；复现性在指标内部。
5. **终止**：`TARGET_MET`（`aggregate().good`，全部条件基于 3-seed 复现）/ `BUDGET_EXHAUSTED`（`MAX_ROUNDS=5` / `MAX_GPU_SECONDS=1800`）/ `HALTED`（连续 3 轮 crash）。预算耗尽必须如实报告未达标并列出差距，**禁止放宽阈值**。
6. **AUTO_PROCEED=true**：不提问、不等待、不结束回合；3-seed 自动完成，无需人工确认。
7. **日志**：`python train.py --seed k --out ... > logs/R{n}_s{k}.log 2>&1`，只 grep `^rho_p3:` / `^refusal_rate:` / `^val_bpb:`。
8. **账本**：`results.tsv` 5 列、未跟踪；`state.json` 原子写、可 `--resume`；每轮一次 git commit（独立嵌套仓库）。
9. **禁止 `git reset --hard`**：主工作区含用户产物；`discard` 只更新 `last_keep` 指针。
10. **不使用** `val_bpb` 原义 / 300 秒墙钟 / `uv run` / 主仓库分支。无跨模型评审后端，思路生成走确定性规则，标注 `llm_backend=REVIEW_UNAVAILABLE`。
