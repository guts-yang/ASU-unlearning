# Experiment Plan — Δ-Gated Progressive Orthogonal Alignment

> claim-driven 实验路线。状态：plan-only，未实现/未训练/无新结果。预算与系数实验前冻结。
> 基座：ASU-unlearning（LLaMA-2 架构 token 粒度，16 decoder layers，TOFU 数据集）。执行环境需 A800 GPU（本机无 CUDA）。

## Claims → Evidence 映射
| Claim | 判别实验 | 通过标准 |
|---|---|---|
| C1 ΔA 编码事实方向 | A1 probe + A2 wrong-donor 对照 | 事实 probe AUC 显著 > 主体 probe；wrong-donor 子空间无同等效果 |
| C3 渐进优于静态 | B1 等预算 schedule 对照 | 渐进在 forget-retain Pareto 上统计显著优于静态全层/反向/随机 |
| C2 正交抑制有效 | C1 配对特异性 + 多指标 | 真实 donor 优于 4 类对照；donor 答案不上升 |
| C4 深度遗忘 | D1 抗恢复 suite | ≥2 类抗恢复证据改善；无 hook 仍有效 |

## Run Order & Budget（估计，需 GPU 复核）
- **Block A（gate, 最先）**：A1 ΔA probe（4 层 × 2 关系族 × 3 seed）；A2 wrong-donor 对照。轻量，<1 GPU·h。
- **Block B（gate）**：B1 四 schedule × 3 seed，等预算。~4 GPU·h。
- **Block C**：C1 五对照配对特异性；C2 答案分解（原事实/donor/错答/拒答/长度/EOS）。~2 GPU·h。
- **Block D**：D1 relearning 曲线 + probe/patching + INT4/INT8 量化 + relation-swap probe；可选 UNDO 蒸馏增强。~2 GPU·h。
- 合计 ~8 GPU·h，与技能 MAX_TOTAL_GPU_HOURS=8 对齐。

## Freeze Before Running
学习率、λ_F/λ_KL/β、层抽点 P、ΔA 秩 k、Λ(k) 序列与 ρ_ℓ(k)、η、生成配置、seed 集、tokenizer/chat template 一致性。冻结后不得根据结果回改。

## Metrics
- 遗忘：TOFU forget QA（FQ）、verbatim（若适用）、probing、MIA。
- 保留：retain QA、MMLU、GSM8K、retain NLL。
- 特异性：donor 答案率、其他错答率、拒答率、平均长度、EOS 率。
- 抗恢复：relearning 曲线斜率、small-mask/quantization 恢复率、relation-swap 偏移。
- 统计单位：作者/实体簇；簇内多 prompt/donor/seed 为重复测量；报告置信区间，禁止单 seed 结论。

## Ablations（由 /ablation-planner 细化）
层集合（全 16 / 稀疏 4 层 / 反向 / 随机）、ΔA 秩 k、损失开关（仅 L_F / 仅 L_KL / 组合）、对比锚（真实 donor / wrong donor / noise）、正交投影开关、UNDO 蒸馏开关。

## Stop / Kill Rules
A1 或 B1 任一失败按研究契约 kill switch 收缩；不复用历史 pilot 作为 confirmation；任一结果写入 paper 前须过 /result-to-claim 与 /paper-claim-audit。

## Evidence & Audit
实验日志 → `EXPERIMENT_LOG.md`（本轮未创建，由 experiment-bridge 生成）；一致性检查 → /experiment-audit；部署级审计 → /paper-claim-audit + /citation-audit。
