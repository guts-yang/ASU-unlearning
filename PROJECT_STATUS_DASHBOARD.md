# LLM-Unlearning Reproduction — Project Status Dashboard

Last updated: 2026-11-27

## 0. Repository state

| Item | Value |
|---|---|
| Project root | `/usr/local/workspace/tracks/llm-unlearning/ASU-unlearning` |
| Branch | `Auto/20261122` |
| Local HEAD | `194286e` |
| Remote HEAD | `194286e` |
| claim_level | `false` |
| evidence_gate | `NOT_APPLIED` |
| Current mode | Non-claim diagnostic |
| Active plan | `cf04-official-budget-compare` |
| Plan status | Ready; not started because GPU queue has pending watchers |

## 1. One-line status

`CF-04 v2g` is the strongest retain-safe candidate; `ASU` remains the stronger standard unlearning baseline; `R3-2` is the best audit component. No method is claim-ready.

## 2. Method performance leaderboard

Budgets differ. Do not compare `dF` across rows without checking the budget column.

| Rank | Method / config | Budget | dF forget | dR retain | dH holdout | retain-safe | Diagnostic verdict | Main note |
|---:|---|---|---:|---:|---:|---|---|---|
| 1 | **CF-04 v2g** | 300 steps, 3 seeds | **+2.3084 ± 0.0915** | **+0.0180 ± 0.0031** | +0.1754 ± 0.0131 | yes | Best candidate | High forget with retain-safe; holdout still weak |
| 2 | CF-04 v2g formal | 10 epochs / 4500 steps | **+8.0920** | **-0.4561** | +0.9656 | yes | High-forget diagnostic | Forget/retain excellent, but holdout severely degraded |
| 3 | ASU global | 300 steps, 3 seeds | +3.1780 ± 0.1492 | +0.1730 ± 0.0069 | **+0.0340 ± 0.0298** | no | Strong baseline | Strongest forget and best holdout, but retain unsafe |
| 4 | CF-04 c11 | 300 steps, 3 seeds | +0.5380 ± 0.0427 | +0.0187 ± 0.0023 | +0.0754 ± 0.0385 | yes | Conservative retain-safe | Safe but much weaker forget |
| 5 | R3-2 audit | audit only | — | — | — | — | Best evaluation tool | P1/P2/P3/INT8 + Probe; not an unlearning method |
| 6 | CF-07 formal | 10 epochs | +0.0266 | -0.4577 | +0.9713 | yes | Analysis / regularizer | Clean representation property, negligible forget |
| 7 | CF-08 | 100 steps | +0.9352 | +0.3150 | -0.0154 | no | Negative | Forget signal but retain damage |
| 8 | CF-10 | 100 steps | +0.4917 | +0.1785 | -0.0266 | no | Negative | Retain damage |
| 9 | CF-01 / CF-09 | 100 steps | +0.2111 | +0.1951 | -0.0007 | no | Negative | Weak forget, retain damage; CF-09 may share implementation path |
| 10 | CF-04 old | 100 steps | +1.0072 | +0.3475 | +0.1126 | no | Negative | Old probe loss, retain unsafe |
| 11 | CF-03 / CF-05 | analysis | 0 | 0 | 0 | — | Inconclusive | Core analysis only, not trainable method |
| 12 | CF-06 | N/A | — | — | — | — | Not applicable | Requires LRM/CoT track |
| 13 | CF-02 / R2-C / R3-3 | 100 steps | +4.0572 / +100.9142 / +96.1653 | +3.7442 / +0.4449 / +2.6453 | +2.2565 / +0.0390 / +6.5238 | no | Negative / diverged | Not suitable without stabilization |

## 3. Current conclusions

| Question | Answer |
|---|---|
| Best trainable candidate | `CF-04 v2g` |
| Best standard baseline | `ASU 300-step` |
| Best audit/evaluation component | `R3-2` |
| Methods ready for claim | None |
| Main unresolved issue for CF-04 v2 | Holdout degradation: 300-step `dH≈0.175`; formal `dH≈0.966` |
| Main unresolved issue for ASU | Retain unsafe: `dR≈0.173` at 300 steps |
| Antirecovery status | ASU has R3-2 audit; CF-04 v2 checkpoint has not been audited |
| Official ASU-budget comparison | Not yet run |

## 4. Budget definitions

| Term | Meaning |
|---|---|
| batch | Number of samples processed in one forward/backward |
| step | One optimizer parameter update |
| epoch | One full pass over the anchor dataset |
| anchor=forget | Epoch length determined by `forget10=400`; retain is sampled |
| anchor=retain | Epoch length determined by `retain90=3600`; forget is sampled |

ASU official configuration:

```text
10 epochs
anchor = forget10 = 400
per_device_train_batch_size = 8
gradient_accumulation_steps = 4
effective_batch = 32（docs口径）
steps_per_epoch = ceil(400/32) = 13
total_steps ≈ 130
learning_rate = 1e-5
hardware = 2×L40 + ZeRO-3
```

Our CF-04 formal 10-epoch is different:

```text
anchor = retain90 = 3600
batch = 8
steps_per_epoch = 450
total_steps = 4500
```

Therefore CF-04 formal 10-epoch is not directly comparable with ASU official 10-epoch.

## 5. Pending work / next actions

| Priority | Item | Status | Reason |
|---:|---|---|---|
| P0 | Wait for GPU watchers | Blocked | DISCO fix watcher, R2MU full watcher, result watcher still queued |
| P1 | ASU vs CF-04 v2g under official ASU budget | Planned, not started | Need fair budget comparison: 10 epochs, anchor=forget, ~130 steps, 3 seeds |
| P2 | CF-04 v3 holdout fix | Planned | Reduce `dH` while preserving `dF≥2.0` and `dR≤0.05` |
| P3 | R3-2 audit of CF-04 v2 checkpoint | Not started | Required to assess antirecovery |
| P4 | ASU same-10-epoch control | Not started | Needed before claiming CF-04 formal superiority |

## 6. Key artifact paths

```text
workspace/refine-logs/UNIFIED_EVALUATION.md
workspace/refine-logs/CF04_V2_OPTIMIZATION_REPORT.md
workspace/refine-logs/CF04_OPTIMIZATION_REPORT.md
workspace/refine-logs/FULL_TRAINING_RESULTS.md
workspace/refine-logs/NONCLAIM_METHOD_RESULTS.md
workspace/auto_experiments/results/cf04_v2_300/
workspace/auto_experiments/results/formal_training/
workspace/auto_experiments/results/full_repro_v1/
```

## 7. Handoff constraints

- This dashboard is a non-claim progress summary.
- It does not invoke the ARIS evidence gate.
- It must not be cited as paper-level reproduction, novelty, or accepted result.
- Any future claim requires full-budget matched comparison, antirecovery audit, and cross-model review.
