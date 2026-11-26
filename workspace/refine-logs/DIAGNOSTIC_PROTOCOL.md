# Non-Claim Diagnostic Protocol

- claim_level: `false`
- evidence_gate: `NOT_APPLIED`
- allowed outputs: diagnostic JSON, tracker/log updates, non-claim report
- forbidden: `accepted`/`provisional` state, `/auto-review-loop`, `/experiment-audit`, `/result-to-claim`, paper claims, novelty claims
- baseline: frozen ASU/open-unlearning TOFU `forget10/retain90/holdout10`, `Llama-3.2-1B-Instruct`
- environment: `/jizhicfs/chenyliao/llm-unlearning/envs/open-unlearning/bin/python`
- GPU: one H20; serial execution only

## Method Order and Diagnostic Standard

| # | Method | Diagnostic mode | Primary diagnostic | Negative / stop criterion |
|---:|---|---|---|---|
| 1 | CF-01 约束反演目标 | operator review + adjacent ablation | forget NLL, retain NLL, stage13 decouple | operator absent or no retain-safe gain → NOT_IMPLEMENTABLE |
| 2 | CF-03 因果门控 | fresh 20-step train: `sens` vs `sens_shuffled` | forget NLL, retain NLL | causal arm not better on both → INCONCLUSIVE/NEGATIVE |
| 3 | CF-02 三前向角色分解 | operator review | retain floor / donor negative constraint | missing operator → NOT_IMPLEMENTABLE |
| 4 | CF-04 事实身份双探针 | existing pooled probe | AUC_f, AUC_id, retain-orthogonal | missing dual probe → INCONCLUSIVE |
| 5 | CF-06 LRM 推理轨迹 | track review | LRM/CoT requirement | LLM-only baseline → NOT_APPLICABLE |
| 6 | CF-05 多 donor 边际化 | operator review | K-donor shared/private AUC | missing K-donor operator → NOT_IMPLEMENTABLE |
| 7 | CF-08 零空间抗恢复投影 | existing depth audit | P2/P3/INT8 recovery, retain NLL | missing projected checkpoint/P3 → INCONCLUSIVE |
| 8 | CF-07 读出空间 deltaA | operator review | rotation invariance, Jacobian | no preregistered Jacobian → NOT_IMPLEMENTABLE |
| 9 | CF-09 结构化对照族残差化 | adjacent ablation review | residual discrimination, EOS/donor rate | operator absent or all gates false → NOT_IMPLEMENTABLE/NEGATIVE |
| 10 | CF-10 纯度驱动 schedule | fresh 20-step train: `high2low` vs `global` | forget NLL, retain NLL | schedule not better on both → NEGATIVE |
| 11 | R2-C 抗恢复即目标 | operator review + audit | recovery gains, retain NLL | minimax trainer absent → NOT_IMPLEMENTABLE |
| 12 | R3-2 深度审计基准 | existing P1/P2/P3/INT8 evaluation | mean_recovery, method separation | Probe missing or no separation → INCONCLUSIVE/POSITIVE_SCREEN |
| 13 | R3-3 浅深解耦 | existing V4/stage9 | dF, dR, strict threshold | dF<0.15 or dR>0.05 → NEGATIVE |

## Result Schema
Every `results/screening/<ID>.json` must contain:
`candidate`, `claim_level: false`, `gate_status: "NOT_APPLIED"`, `diagnostic_mode`, `training_executed`, `evaluation_executed`, `verdict`, `metrics`, `evidence`, `limitations`, `timestamp`.

## Commit and Push Format
After each method:

```text
commit: diag(<ID>): non-claim <short-verdict>
files:
  workspace/auto_experiments/results/screening/<ID>.json
  workspace/refine-logs/EXPERIMENT_TRACKER.md
  workspace/refine-logs/EXPERIMENT_LOG.md
  workspace/refine-logs/DIAGNOSTIC_REPORT.md
push: origin Auto/20261122
```

No force push. No token in commit messages, files, or remotes.
