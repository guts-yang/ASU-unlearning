# AUDIT_FEEDBACK — 待核实线索（不是已证实造假）

**来源**：`paper_under_audit/REPORT.md`（deterministic-only，L2，`REVIEW_UNAVAILABLE`）
**日期**：2026-11-23　**性质**：detect-only 决策支持，`human_review_required = true`

## 审计结论

- `overall_verdict = REVIEW_UNAVAILABLE`；`observability_level = 2`；`adjudicator = deterministic-rules-v2`；`anchoring_verified = true`。
- 四个确定性检查器（numeric / presentation / stat / AI-style）各产出 0 条 finding；10 个 verdict-bearing 维度全部 `review_unavailable`，因此**按规则阻断"无罪"判定**，不能声称 `CLEAN_GIVEN_EVIDENCE`，更不能声称"论文诚实"。
- Anti 不做 L3 复现，不验证 `run.log` 是否真由对应 commit 产生；也不做语义级一致性判断。

## 人工需要核实的两处（由执行者自审发现，非审计工具产出）

| # | 位置 | 逐字 span | 建议核实点 | 状态 |
|---|---|---|---|---|
| 1 | `submission.tex` §Abstract | "improves forgetting by 0.1491 NLL over the global tau=2 baseline" | 用本循环 anchor/round 数字重算：`0.9491 - 0.7998 = 0.1493`，与 0.1491 相差 0.0002；更重要的是 §Results 已写明等预算对照 `dF=1.1416` 高于 `0.9491`，摘要的"improves"表述与结果表**语义冲突**。摘要应改写为"在与 τ=2 的非等预算比较中观察到 +0.1493，但等预算下不成立"。 | 待核实；**不要**在取证阶段改动被审文件 |
| 2 | `submission.tex` §Method / §Results | "The baseline val_bpb is 0.000000 and the best val_bpb is 0.000000." | 两句重复且 "best" 与 anchor 不可区分；`results/experiments.csv` 中 6 个 candidate 全为 discard，应明确写出"0 candidates kept"，否则读者会误读为有改进被保留。 | 待核实 |

## 交回 autoresearch-loop 的可操作线索

- **线索 A（实验层）**：第 3 幕已被等预算对照推翻，下一轮 `train.py` 不应再尝试 τ 的空间/时间分配；应转向 `STORY_DESIGN.md` 第 5 幕（抗恢复目标 / bottom-PC），并且**每个新候选必须先声明其等预算 anchor**。
- **线索 B（写作层）**：`submission.tex` 摘要与结果表的冲突应在下一版修订；这是表述问题，不是数字造假。
- **线索 C（覆盖层）**：启用任一异族评审后端后重跑完整模式，才能把 `consistency-audit` / `baseline-comparison-audit` 从 `review_unavailable` 升级为 completed；当前 verdict 结构性不可 acquit。

> 不要把上述 finding 当作已证实的学术不端；不要把 `REVIEW_UNAVAILABLE` 当作通过。

## 追加线索（Round 2 结束后）

| # | 类型 | 内容 | 状态 |
|---|---|---|---|
| D | **指标层（重要）** | `loop2/prepare.py::compute_regret` 第三项只惩罚「忘得更少」，未惩罚「忘得更多」；且 \(0.02\kappa\) 复杂度惩罚把唯一有效候选 B5 误判为 `discard`。matched-ΔF 对照 B6 证明 B5 的 \(\rho\) 下降不是分母伪影（0.4085 vs 0.5909）。下一版需对称指标或 ΔF 预算带约束。 | 待核实；prepare.py 是固定文件，本轮未改 |
| E | 实验层 | B5 的 hinge 可能只造成退化输出（拒答/乱码），P3 后 NLL 仍高 ≠ 知识擦除。必须补生成质量、答案分布、拒答率、3-seed、多种攻击强度。 | 待核实，不得写成主张 |
| F | 覆盖层 | 子空间路线 B1–B4 的 \(\rho\) 与 anchor 差在 ±0.003 内，bottom/top/随机不可区分；第 5 幕的 RepSelect/SAGO 迁移价值在等预算下不成立。 | 已观测 |
