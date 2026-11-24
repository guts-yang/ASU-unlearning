# program.md — RelResidual 验证闭环 v6（固定，循环中不可修改）

完整设计见 `DESIGN.md`。本文件只固定执行纪律。

1. **只读**：`prepare.py`（配对构造、评估协议、`aggregate` 终止判据、预算常量）、`program.md`、`DESIGN.md`、`gate0.py`（前置假设协议）、`gate0.json`、`idea_engine.py` 的 `LEVERS` 白名单、 upstream `common.py` / `counterfactual.py` 与既有结果。
2. **唯一可编辑代码**：`train.py`（通用思路解释器，一次调用 = 一个独立数据顺序），含 `layer_select / alpha_B / kl_weight / lf_form` 显式分支。每轮变化的是 `current_idea.json`（数据）。
3. **S0 诊断输入**：`gate0.json` 的 verdict 与逐层 cos 必须读入；`verdict=KILL` 只标注「原版前提不成立」，不停机，而是驱动层收缩思路。
4. **固定预算**：60 steps（与 ASU global τ=2 基线等预算）、\(\eta=5\times10^{-6}\)、24 组 token 等长四元组、32 条 greedy 生成；**每个 idea 固定 3 个独立数据顺序**（0/1/2），`INIT_SEED=0`，配对构造固定、仅训练采样顺序随 order 变化。
5. **单一指标**：`val_bpb = max(0, 0.7998 - mean_dF) + 最坏约束惩罚 + 2*std(dF)`，越低越好；**不得弱于 ASU 基线**是硬约束。
6. **终止**：`TARGET_MET`（`aggregate().good`，全部基于 3 个独立顺序）/ `BUDGET_EXHAUSTED`（`MAX_ROUNDS=4` / `MAX_GPU_SECONDS=1800`）/ `HALTED`（连续 3 轮 crash）。预算耗尽必须如实报告未达标及差距，**禁止放宽阈值**。
7. **AUTO_PROCEED=true**：不提问、不等待、不结束回合；3-order 复现自动完成。
8. **日志**：`python train.py --data-order k --out ... > logs/R{n}_o{k}.log 2>&1`，只 grep `^dF:` / `^donor_leak:` / `^premature:` / `^truth_leak:` / `^val_bpb:`。
9. **账本**：`results.tsv` 5 列、未跟踪；`state.json` 原子写、可 `--resume`；每轮一次 git commit（独立嵌套仓库）。
10. **禁止 `git reset --hard`**；不使用 `val_bpb` 原义 / 300 秒墙钟 / `uv run` / 主仓库分支。无跨模型评审后端，思路生成走确定性规则，标注 `llm_backend=REVIEW_UNAVAILABLE`。
