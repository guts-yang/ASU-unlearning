# Experiment Tracker — Plan-level Candidate Screening

| Run | Candidate | Priority | Mode | Status | Result File |
|---|---|---|---|---|---|
| R000 | sanity | 0 | 5-step global tau | PASSED | results/screening/sanity.json |
| R001 | CF-01 约束反演目标 | rank 1 / 8.0 | operator review + adjacent ablation | NOT_IMPLEMENTABLE | results/screening/CF-01.json |
| R002 | CF-03 因果门控 activation gate | rank 2 / 7.8 | 20-step sens vs sens_shuffled | PENDING | results/screening/CF-03.json |
| R003 | CF-02 三前向角色分解 | rank 3 / 7.6 | operator review | NOT_IMPLEMENTABLE | results/screening/CF-02.json |
| R004 | CF-04 事实身份双探针解耦 | rank 4 / 7.4 | existing pooled probe | PENDING | results/screening/CF-04.json |
| R005 | CF-06 LRM 推理轨迹 deltaA | rank 5 / 7.4 | track review | PENDING | results/screening/CF-06.json |
| R006 | CF-05 多 donor 边际化 | rank 6 / 7.4 | operator review | PENDING | results/screening/CF-05.json |
| R007 | CF-08 零空间抗恢复投影 | rank 7 / 7.4 | existing depth audit | PENDING | results/screening/CF-08.json |
| R008 | CF-07 读出空间 deltaA readout | rank 8 / 7.2 | operator review | PENDING | results/screening/CF-07.json |
| R009 | CF-09 结构化对照族残差化 | rank 9 / 7.2 | adjacent ablation review | PENDING | results/screening/CF-09.json |
| R010 | CF-10 纯度驱动渐进 schedule | rank 10 / 7.2 | 20-step high2low vs global | PENDING | results/screening/CF-10.json |
| R011 | R2-C 抗恢复即目标 | PROMOTE | operator review + audit | PENDING | results/screening/R2-C.json |
| R012 | R3-2 ASU 遗忘深度审计基准 | PROMOTE | existing P1/P2/P3/INT8 evaluation | PENDING | results/screening/R3-2.json |
| R013 | R3-3 浅深解耦双阶段遗忘 | REVISE / ablation only | existing V4/stage9 evaluation | PENDING | results/screening/R3-3.json |

## Verdict vocabulary
- `POSITIVE_SCREEN`: diagnostic signal meets stated criterion.
- `NEGATIVE`: criterion not met.
- `INCONCLUSIVE`: required arms/metrics missing.
- `NOT_IMPLEMENTABLE`: core operator absent.
- `NOT_APPLICABLE`: requires another track.
