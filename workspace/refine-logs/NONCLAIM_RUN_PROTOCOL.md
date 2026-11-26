# Non-Claim Execution Protocol (Core Algorithms v2)

- claim_level: `false`
- evidence_gate: `NOT_APPLIED`
- run_kind: `non-claim diagnostic`
- baseline: frozen ASU/open-unlearning TOFU `forget10/retain90/holdout10`, `Llama-3.2-1B-Instruct`
- device: one H20 per invocation; serial
- code: `workspace/experiments/diagnostic_methods/`
- result root: `workspace/auto_experiments/results/nonclaim/`
- forbidden: `accepted`/`provisional`, `/auto-review-loop`, `/experiment-audit`, `/result-to-claim`, novelty/citation/reproduction claims
- commit format: `diag(<ID>): non-claim <VERDICT>`; push `origin Auto/20261122`

## Run Order, Budget, Metrics

| # | Method | Steps | Primary diagnostic | Pass / negative |
|---:|---|---:|---|---|
| 1 | CF-01 | 10 | forget/retain NLL; constrained ΔA diagnostics | retain-safe forget gain |
| 2 | CF-03 | 0 | patch NLL effects, null vs real patch | real effect separated from null |
| 3 | CF-02 | 3 | role losses; forget/retain NLL | retain floor preserved |
| 4 | CF-04 | probe 200 | AUC_fact, AUC_identity, margin | margin ≥ 0.10 |
| 5 | CF-06 | 0 | LRM applicability | NOT_APPLICABLE on LLM |
| 6 | CF-05 | 0 | shared/private fraction, donor consistency | shared fraction high + consistency high |
| 7 | CF-08 | 3 | nullspace removed fraction; forget/retain NLL | retain-safe forget gain |
| 8 | CF-07 | 0 | rotation invariance, readout delta | invariance holds |
| 9 | CF-09 | 3 | residual fraction; forget/retain NLL | residual retains signal + no retain collapse |
| 10 | CF-10 | 3 | purity schedule; forget/retain NLL | schedule better on both |
| 11 | R2-C | 2 | post-attack forget NLL, retain NLL | recovery-aware gain |
| 12 | R3-2 | 0 | P1/P2/P3/INT8, mean_recovery | method separation + shallow detection |
| 13 | R3-3 | 3 | two-stage schedule; forget/retain NLL | dF/dR direction |

## Fixed result schema
`candidate, claim_level:false, gate_status:"NOT_APPLIED", run_kind, before, after, delta, training_executed, evaluation_executed, metrics/modules, verdict, limitations`.

## Verdict vocabulary
`DIAGNOSTIC_POSITIVE`, `DIAGNOSTIC_NEGATIVE`, `DIAGNOSTIC_INCONCLUSIVE`, `NOT_APPLICABLE`.
