# program.md — 自主训练闭环 v5（固定，循环中不可修改）

完整设计见 `DESIGN_V5.md`。本文件只固定执行纪律。

1. **只读**：`prepare.py`（含独立复现聚合 `aggregate`、终止判据、预算常量、`SEED_IDEA`）、`program.md`、`DESIGN_V5.md`、`idea_engine.py` 的 `LEVERS` 白名单、上游 `common.py` 与既有结果。
2. **唯一可编辑代码**：`train.py`（通用思路解释器，一次调用 = 一个独立数据顺序），含 `ema` / `tail_avg` / `retain_mix` / `none` 四个显式稳定化分支。每轮变化的是 `current_idea.json`（数据）。
3. **起点**：`SEED_IDEA` = v4 R3（v4 中已独立复现的最佳 idea，`alpha=0.04, β=0.15`）；R1 重新验证，不直接继承结论。
4. **固定预算**：60 steps、\(\tau=2.25\)、P3 20 步 SGD、32 条 greedy 生成；**每个 idea 固定 3 个独立数据顺序**（0/1/2），`INIT_SEED=0`。
5. **单一指标**：`val_bpb = rho_mean + 2*rho_std + 最坏约束惩罚`，与 v4 完全一致，越低越好；独立复现性在指标内部。
6. **终止**：`TARGET_MET`（`aggregate().good`，全部基于 3 个独立顺序）/ `BUDGET_EXHAUSTED`（`MAX_ROUNDS=5` / `MAX_GPU_SECONDS=1800`）/ `HALTED`（连续 3 轮 crash）。预算耗尽必须如实报告未达标及差距，**禁止放宽阈值**。
7. **AUTO_PROCEED=true**：不提问、不等待、不结束回合；3-order 复现自动完成。
8. **日志**：`python train.py --data-order k --out ... > logs/R{n}_o{k}.log 2>&1`，只 grep `^rho_p3:` / `^refusal_rate:` / `^val_bpb:`。
9. **账本**：`results.tsv` 5 列、未跟踪；`state.json` 原子写、可 `--resume`；每轮一次 git commit（独立嵌套仓库）。
10. **禁止 `git reset --hard`**；不使用 `val_bpb` 原义 / 300 秒墙钟 / `uv run` / 主仓库分支。无跨模型评审后端，思路生成走确定性规则，标注 `llm_backend=REVIEW_UNAVAILABLE`。
