# Non-Claim Method Results (Core Algorithms v2)

- claim_level: `false`
- evidence_gate: `NOT_APPLIED`
- baseline: ASU/open-unlearning TOFU `forget10/retain90/holdout10`, `Llama-3.2-1B-Instruct`
- status: serial diagnostic run

| # | Method | Steps | Verdict | forget Δ | retain Δ | Key diagnostic |
|---:|---|---:|---|---:|---:|---|
| 1 | CF-01 | 10 | DIAGNOSTIC_NEGATIVE | +0.1054 | +0.1220 | constrained ΔA + 10-step train |
| 2 | CF-03 | 0 | DIAGNOSTIC_INCONCLUSIVE | +0.0000 | +0.0000 | patch effect vs null |
| 3 | CF-02 | 3 | DIAGNOSTIC_NEGATIVE | +1.7697 | +1.7578 | role decomposition + 3-step train |
| 4 | CF-04 | 0 | DIAGNOSTIC_NEGATIVE | +0.0000 | +0.0000 | dual probe AUC |
| 5 | CF-06 | 0 | NOT_APPLICABLE | — | — | LRM applicability |
| 6 | CF-05 | 0 | PENDING | — | — | multi-donor SVD |
| 7 | CF-08 | 3 | DIAGNOSTIC_NEGATIVE | +0.2182 | +0.1791 | nullspace + 3-step train |
| 8 | CF-07 | 0 | DIAGNOSTIC_POSITIVE | +0.0000 | +0.0000 | readout rotation |
| 9 | CF-09 | 3 | DIAGNOSTIC_NEGATIVE | -0.0284 | -0.0420 | control residual + 3-step train |
| 10 | CF-10 | 3 | DIAGNOSTIC_NEGATIVE | +0.4253 | +0.4550 | purity schedule + 3-step train |
| 11 | R2-C | 2 | DIAGNOSTIC_NEGATIVE | +0.2858 | +0.1114 | inner attack + minimax |
| 12 | R3-2 | 0 | PENDING | — | — | existing depth audit |
| 13 | R3-3 | 3 | DIAGNOSTIC_NEGATIVE | +0.4210 | +0.2343 | two-stage schedule + 3-step train |

## Constraint
This is a non-claim diagnostic run. It does not invoke the ARIS evidence gate, does not write accepted/provisional state, and cannot be cited as faithful reproduction or paper evidence.
