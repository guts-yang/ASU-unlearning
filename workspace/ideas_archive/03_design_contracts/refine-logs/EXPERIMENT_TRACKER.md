# Experiment Tracker — V4 自动 idea 验证

状态：gate 阶段已完成（H20 / TOFU-1B / 2026-11-23）。详细结果见 `EXPERIMENT_RESULTS.md`。

| Run | Idea | 配置 | Seed | 状态 | 关键结果 |
|---|---|---|---|---|---|
| G0 | V1 全系 | stage0-14 既有实测复核 | 0/1/2 | DONE | KILL，全部弱于 ASU ref |
| G1 | R3-1 | τ 敏感度诊断（层/头） | 0 | DONE | layer CV=1.11，head CV=9.22，PASS |
| G2 | R2-A | ASU-Mu forget-grad SVD 选择性 | 0 | DONE | mean_selectivity=0.531，机制 PASS；新颖性冲突 → REVISE |
| G3 | R2-B | low2high / high2low / global 等步数训练 | 0 | DONE | low2high dF=+0.949 最优，PASS |
| G4 | R3-3 | 深层 ΔA suppress/align gate | 0 | DONE | 边缘信号，严格阈值 REVISE |
| G5 | R3-2 | P1/P2/P3/INT8 深度审计 | 0 | DONE | ASU-u mean_recovery=0.448，PASS |
| G6 | R3-1 | 等平均 τ 与打乱场对照 | 0 | DONE | 结构化分配无净收益，REVISE |
| M1 | R2-A | 完整两段式 + P1/P2/P3 | — | BLOCKED | 需先完成与 RepSelect/ADU/ALMPU 划界 |
| M2 | R2-C | relearning-aware meta 目标 | — | NOT STARTED | 下一步主线 |
| M3 | R2-B | 3-seed 重复确认 | — | NOT STARTED | 主线增量验证 |
| M4 | R3-2 | 统一审计扩展到更多 checkpoint | — | NOT STARTED | 基准完善 |

原始产物：`workspace/auto_experiments/results/*.json`；日志：`workspace/auto_experiments/logs/`。
