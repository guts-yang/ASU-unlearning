# Initial Experiment Results — 自动 idea 验证

**日期**: 2026-11-23
**算力**: 1×NVIDIA H20，TOFU-1B，数据/模型 `/jizhicfs/chenyliao`
**代码**: `workspace/auto_experiments/`
**原始 JSON**: `workspace/auto_experiments/results/`

## Gate 结果

| Idea | Gate | 关键数字 | 自动决策 |
|---|---|---|---|
| V1 全系（反事实配对/ΔA 正交/梯度手术） | KILL | stage0-14 既有实测全部弱于 ASU ref | ELIMINATED |
| R2-A ASU-Selective | PASS | mean_selectivity=0.531（v_proj/down_proj）；但新颖性与 RepSelect/ADU/ALMPU 冲突 | REVISE |
| R2-B τ 课程 | PASS | dF: low2high=+0.949 > global=+0.800，dR=-0.256 | PROMOTE |
| R2-C 抗恢复目标 | PASS | ASU-u P3 gain ratio=0.371，mean_recovery=0.448 | PROMOTE |
| R3-1 SensiTemp | PARTIAL | layer CV=1.11，head CV=9.22；dF: sens=1.747 vs 等平均τ=1.792 vs 打乱场=1.517 | REVISE |
| R3-2 深度审计 | PASS | ASU-u P1 forget=0.960，P2 gain=0.261，P3 gain=0.185，Q shift 已记录 | PROMOTE |
| R3-3 浅-深解耦 | PASS→REVISE | L12 suppress dF=+0.133/dR=+0.099；严格阈值通过数=0 | REVISE |

## 训练对照（60 steps, alpha=0.1, lr=5e-6, seed 0）

| 配置 | forget 0.5025→ | retain 0.4651→ |
|---|---|---|
| global τ=2 | 1.3023 | 0.1695 |
| global τ=2.607（等平均） | 2.2940 | 0.1609 |
| high2low 3→1.5 | 1.0646 | 0.1451 |
| low2high 1.5→3 | 1.4516 | 0.2089 |
| SensiTemp field | 2.2495 | 0.1654 |
| SensiTemp field shuffled | 2.0191 | 0.1598 |

## 结论
- 主线：R2-B、R2-C、R3-2；降级：R2-A（新颖性冲突）、R3-1、R3-3；淘汰：V1 全系。
- SensiTemp 的收益可由等平均 τ 全局对照解释，不能主张结构化温度分配带来帕累托改进。
- 所有裁决为执行者 provisional（跨模型评审后端不可用），未启用 `experiment-audit` / `result-to-claim`。
