# Non-Claim Method Results (Core Algorithms v2)

- claim_level: `false`
- evidence_gate: `NOT_APPLIED`
- baseline: ASU/open-unlearning TOFU `forget10/retain90/holdout10`, `Llama-3.2-1B-Instruct`
- status: serial diagnostic run

| # | Method | Steps | Verdict | forget Δ | retain Δ | Key diagnostic |
|---:|---|---:|---|---:|---:|---|
| 1 | CF-01 | 10 | DIAGNOSTIC_NEGATIVE | +0.1054 | +0.1220 | constrained ΔA + 10-step train |
| 2 | CF-03 | 0 | PENDING | — | — | patch effect vs null |
| 3 | CF-02 | 3 | PENDING | — | — | role decomposition + 3-step train |
| 4 | CF-04 | 0 | PENDING | — | — | dual probe AUC |
| 5 | CF-06 | 0 | PENDING | — | — | LRM applicability |
| 6 | CF-05 | 0 | PENDING | — | — | multi-donor SVD |
| 7 | CF-08 | 3 | PENDING | — | — | nullspace + 3-step train |
| 8 | CF-07 | 0 | PENDING | — | — | readout rotation |
| 9 | CF-09 | 3 | PENDING | — | — | control residual + 3-step train |
| 10 | CF-10 | 3 | PENDING | — | — | purity schedule + 3-step train |
| 11 | R2-C | 2 | PENDING | — | — | inner attack + minimax |
| 12 | R3-2 | 0 | PENDING | — | — | existing depth audit |
| 13 | R3-3 | 3 | PENDING | — | — | two-stage schedule + 3-step train |

## Constraint
This is a non-claim diagnostic run. It does not invoke the ARIS evidence gate, does not write accepted/provisional state, and cannot be cited as faithful reproduction or paper evidence.
