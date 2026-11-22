# Experiment Tracker — Δ-Gated Progressive Orthogonal Alignment

状态：plan-only。所有 run 待 GPU 环境启动后填写。

| Run | Block | 配置 | Seed | 状态 | 关键结果 | 备注 |
|---|---|---|---|---|---|---|
| A1 | A gate | ΔA probe (事实 vs 主体) | 3 | not started | — | kill switch gate |
| A2 | A gate | wrong-donor ΔA 对照 | 3 | not started | — | 验特异性 |
| B1 | B gate | 渐进 vs 静态全层 vs 反向 vs 随机 | 3 | not started | — | 等预算 |
| C1 | C | 真实/wrong/shuffled/noise/matched-retain donor | 3 | not started | — | 5 对照 |
| C2 | C | 答案分解多指标 | 3 | not started | — | 防 donor 替换 |
| D1 | D | relearning/probe/quant/relation-swap | 3 | not started | — | ≥2 类抗恢复 |
| D2 | D(optional) | UNDO unlearn-and-distill | 3 | not started | — | 抗恢复增强 |

本轮 idea-discovery 未运行任何实验（本机无 CUDA），此表为后续 `/experiment-bridge` / `/run-experiment` 的占位。
