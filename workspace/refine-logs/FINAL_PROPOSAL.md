# Final Proposal — Plan-level Candidate Screening on Frozen ASU TOFU-1B

> 2026-11-26 W1→W1.5 handoff。覆盖 `idea-stage/candidates/` 下 13 个计划级候选。
> 状态：bounded screening proposal；不是论文投稿主张，不声称 faithful reproduction。

## Problem Anchor
在冻结基线 ASU/open-unlearning TOFU `forget10/retain90/holdout10` + `Llama-3.2-1B-Instruct` 上，按候选文件显式优先级逐个执行最小判别实验，判断每个计划级候选是否有正向、可审计的首轮信号。统计单位、指标口径和算力预算在开跑前冻结，不因结果回改。

## Method Thesis
以 `workspace/auto_experiments/` 为代码基线，优先复用已生成的 gate/恢复审计结果；只对缺失 arms 补跑短程训练。每个候选必须记录：实现状态、对照臂、forget/retain 指标、恢复或特异性证据、判定。机制无法在本 LLM 基线实现时标 `NOT_APPLICABLE`/`NOT_IMPLEMENTABLE`，不伪造复现。

## Candidate Set and Order
1. CF-01 约束反演目标（rank 1, 8.0）
2. CF-03 因果门控 activation gate（rank 2, 7.8）
3. CF-02 三前向角色分解（rank 3, 7.6）
4. CF-04 事实身份双探针解耦（rank 4, 7.4）
5. CF-06 LRM 推理轨迹 deltaA（rank 5, 7.4；本线 NOT_APPLICABLE）
6. CF-05 多 donor 边际化（rank 6, 7.4）
7. CF-08 零空间抗恢复投影（rank 7, 7.4）
8. CF-07 读出空间 deltaA readout（rank 8, 7.2）
9. CF-09 结构化对照族残差化（rank 9, 7.2）
10. CF-10 纯度驱动渐进 schedule（rank 10, 7.2）
11. R2-C 抗恢复即目标（PROMOTE）
12. R3-2 ASU 遗忘深度审计基准（PROMOTE）
13. R3-3 浅深解耦双阶段遗忘（REVISE，仅消融）

## Bounded Claims
- 首轮只能声称“某计划级候选在冻结协议上通过/未通过判别 screening”，不能声称论文级复现、novelty 或 submission-ready。
- 正向信号定义：在所声明的成功判据上，候选臂相对对照臂同时满足 forget 目标与 retain 约束；恢复类候选还需至少两类独立恢复证据。
- 所有新结果必须进入 `EXPERIMENT_TRACKER.md` 与 JSON 文件，随后才允许 `/auto-review-loop → /experiment-audit → /result-to-claim`。

## Kill Switches
- 实现需要跨轨道数据/模型（CF-06）→ `NOT_APPLICABLE`，转 LRM track。
- 现有代码无法表达候选定义的核心算子 → `NOT_IMPLEMENTABLE`，不降级成无关超参扫描后仍称复现。
- 单 seed 正向但 3-seed 确认失败 → `NEGATIVE`/`INCONCLUSIVE`。
- 任一候选超过 2x 预算 → 记录 timeout 并进入下一候选。
