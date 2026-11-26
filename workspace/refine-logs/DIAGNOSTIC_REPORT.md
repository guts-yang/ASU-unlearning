# Non-Claim Diagnostic Report

- claim_level: `false`
- evidence_gate: `NOT_APPLIED`
- started_at: `2026-11-26`
- baseline: ASU/open-unlearning TOFU `forget10/retain90/holdout10`, `Llama-3.2-1B-Instruct`
- scope: 13 candidate methods, serial execution, diagnostic only

## Method Results

| # | Method | Diagnostic mode | Verdict | Key result |
|---:|---|---|---|---|
| 0 | sanity | 5-step global tau | PASSED | forget 0.5025→2.0562; retain 0.4651→1.4441 |
| 1 | CF-01 | operator review | PENDING | — |
| 2 | CF-03 | 20-step sens vs shuffled | PENDING | — |
| 3 | CF-02 | operator review | PENDING | — |
| 4 | CF-04 | existing probe | PENDING | — |
| 5 | CF-06 | track review | PENDING | — |
| 6 | CF-05 | operator review | PENDING | — |
| 7 | CF-08 | existing audit | PENDING | — |
| 8 | CF-07 | operator review | PENDING | — |
| 9 | CF-09 | adjacent ablation | PENDING | — |
| 10 | CF-10 | 20-step high2low vs global | PENDING | — |
| 11 | R2-C | operator review | PENDING | — |
| 12 | R3-2 | existing recovery audit | PENDING | — |
| 13 | R3-3 | existing V4/stage9 | PENDING | — |

## Constraints

This report is not a claim artifact. It does not invoke the ARIS evidence gate, does not record `accepted`/`provisional`, and must not be cited as reproduction or novelty evidence.
