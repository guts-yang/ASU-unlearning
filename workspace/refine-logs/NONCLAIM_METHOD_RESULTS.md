# Non-Claim Method Results (Core Algorithms v2)

- claim_level: `false`
- evidence_gate: `NOT_APPLIED`
- baseline: ASU/open-unlearning TOFU `forget10/retain90/holdout10`, `Llama-3.2-1B-Instruct`
- status: serial diagnostic run

| # | Method | Verdict | forget Δ | retain Δ | holdout Δ | Key diagnostic |
|---:|---|---|---:|---:|---:|---|
| 1 | CF-01 | DIAGNOSTIC_NEGATIVE | +0.1054 | +0.1220 | — | constrained ΔA; dF/dR 判据 |
| 2 | CF-03 | DIAGNOSTIC_INCONCLUSIVE | +0.0000 | +0.0000 | +0.0000 | patch effects vs shuffled null |
| 3 | CF-02 | DIAGNOSTIC_NEGATIVE | +1.7697 | +1.7578 | — | role decomposition; retain floor |
| 4 | CF-04 | DIAGNOSTIC_POSITIVE | +0.0000 | +0.0000 | +0.0000 | AUC_fact=1.000; AUC_id=0.764; margin=0.236 |
| 5 | CF-06 | NOT_APPLICABLE | — | — | — | LRM/CoT required |
| 6 | CF-05 | DIAGNOSTIC_INCONCLUSIVE | +0.0000 | +0.0000 | — | shared=0.667; donor cosine=0.232 |
| 7 | CF-08 | DIAGNOSTIC_NEGATIVE | +0.2840 | +0.2238 | — | nullspace removed fraction; dF/dR |
| 8 | CF-07 | DIAGNOSTIC_POSITIVE | +0.0000 | +0.0000 | — | rotation change=1.26e-6; cosine=1.000 |
| 9 | CF-09 | DIAGNOSTIC_NEGATIVE | +0.0122 | -0.0019 | — | residual fraction; dF/dR |
| 10 | CF-10 | DIAGNOSTIC_NEGATIVE | +0.5226 | +0.5586 | — | purity schedule; dF/dR |
| 11 | R2-C | DIAGNOSTIC_NEGATIVE | +0.4930 | +0.2471 | — | inner attack + minimax; dF/dR |
| 12 | R3-2 | DIAGNOSTIC_POSITIVE | — | — | — | ASU mean_recovery=0.448; GA=0.379 |
| 13 | R3-3 | DIAGNOSTIC_NEGATIVE | +0.4705 | +0.2539 | — | two-stage; dF/dR |

## Constraint
This is a non-claim diagnostic run. It does not invoke the ARIS evidence gate, does not write accepted/provisional state, and cannot be cited as faithful reproduction or paper evidence.
