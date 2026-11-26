# Experiment Tracker — Plan-level Candidate Screening

| Run | Candidate | Priority | Milestone | Status | Result File | Notes |
|---|---|---:|---|---|---|---|
| R000 | sanity | 0 | M0 | PASSED | results/screening/sanity.json | 5-step global tau on frozen TOFU-1B |
| R001 | CF-01 约束反演目标 | 1 | M1 | NOT_IMPLEMENTABLE | results/screening/CF-01.json | constrained inverse target |
| R002 | CF-03 因果门控 | 2 | M2 | INCONCLUSIVE | results/screening/CF-03.json | causal activation gate |
| R003 | CF-02 三前向角色分解 | 3 | M3 | NOT_IMPLEMENTABLE | results/screening/CF-02.json | three-role decomposition |
| R004 | CF-04 事实身份双探针 | 4 | M4 | INCONCLUSIVE | results/screening/CF-04.json | fact/identity probes |
| R005 | CF-06 LRM 推理轨迹 | 5 | M5 | NOT_APPLICABLE | results/screening/CF-06.json | NOT_APPLICABLE on LLM track |
| R006 | CF-05 多 donor 边际化 | 6 | M6 | NOT_IMPLEMENTABLE | results/screening/CF-05.json | multi-donor marginalization |
| R007 | CF-08 零空间抗恢复投影 | 7 | M7 | INCONCLUSIVE | results/screening/CF-08.json | nullspace anti-recovery |
| R008 | CF-07 读出空间 deltaA | 8 | M8 | NOT_IMPLEMENTABLE | results/screening/CF-07.json | readout-space deltaA |
| R009 | CF-09 结构化对照族残差化 | 9 | M9 | NOT_IMPLEMENTABLE | results/screening/CF-09.json | structured control residual |
| R010 | CF-10 纯度驱动 schedule | 10 | M10 | NEGATIVE | results/screening/CF-10.json | purity-driven schedule |
| R011 | R2-C 抗恢复即目标 | 11 | M11 | NOT_IMPLEMENTABLE | results/screening/R2-C.json | relearning-aware objective |
| R012 | R3-2 深度审计基准 | 12 | M12 | POSITIVE_SCREEN | results/screening/R3-2.json | depth audit benchmark |
| R013 | R3-3 浅深解耦 | 13 | M13 | NEGATIVE | results/screening/R3-3.json | ablation only |

## Verdict vocabulary
- `POSITIVE_SCREEN`: first-pass signal meets the candidate's stated success criterion.
- `NEGATIVE`: stated success criterion not met by existing/new evidence.
- `INCONCLUSIVE`: required arms or metrics missing.
- `NOT_IMPLEMENTABLE`: core operator cannot be expressed by the frozen baseline harness.
- `NOT_APPLICABLE`: requires another track (CF-06).
