# Unified Evaluation of Implemented Methods

- claim_level: `false`
- evidence_gate: `NOT_APPLIED`
- baseline: ASU/open-unlearning TOFU `forget10/retain90/holdout10`, `Llama-3.2-1B-Instruct`

## A. Common 100-step Full-Data Diagnostic

| Method | Budget | dF forget | dR retain | dHoldout | Diagnostic | Verdict |
|---|---:|---:|---:|---:|---|---|
| ASU_BASE | 100 steps | — | — | — | 0.4832 / 0.4769 | baseline |
| CF-01 | 100 steps | +0.2111 | +0.1951 | -0.0007 | constrained ΔA | NEGATIVE |
| CF-02 | 100 steps | +4.0572 | +3.7442 | +2.2565 | role decomposition | NEGATIVE-COLLAPSE |
| CF-03 | 100 steps | +0.0000 | +0.0000 | +0.0000 | patch effects only | INCONCLUSIVE |
| CF-04 | 100 steps | +1.0072 | +0.3475 | +0.1126 | dual probe + 100-step | NEGATIVE |
| CF-05 | 100 steps | +0.0000 | +0.0000 | +0.0000 | multi-donor SVD only | INCONCLUSIVE |
| CF-06 | 0 steps | — | — | — | LRM/CoT required | NOT_APPLICABLE |
| CF-07 | 100 steps | +1.5686 | +0.8580 | +0.1396 | readout-space + 100-step | NEGATIVE |
| CF-08 | 100 steps | +0.9352 | +0.3150 | -0.0154 | nullspace + 100-step | NEGATIVE |
| CF-09 | 100 steps | +0.2111 | +0.1951 | -0.0007 | control residual + 100-step | NEGATIVE |
| CF-10 | 100 steps | +0.4917 | +0.1785 | -0.0266 | purity schedule + 100-step | NEGATIVE |
| R2-C | 100 steps | +100.9142 | +0.4449 | +0.0390 | inner attack + minimax | NEGATIVE-DIVERGED |
| R3-2 | 0 steps | — | — | — | P1/P2/P3/INT8 audit | MIXED |
| R3-3 | 100 steps | +96.1653 | +2.6453 | +6.5238 | two-stage | NEGATIVE-DIVERGED |

## B. Formal 10-Epoch Full-Data Training

| Method | Epochs | dF forget | dR retain | dHoldout | Verdict | Checkpoint |
|---|---:|---:|---:|---:|---|---|
| CF-04 | 10 | +0.6294 | -0.4609 | +0.8917 | DIAGNOSTIC_POSITIVE | `/jizhicfs/chenyliao/llm-unlearning/checkpoints/full_cf04_formal` |
| CF-07 | 10 | +0.0266 | -0.4577 | +0.9713 | DIAGNOSTIC_NEGATIVE | `/jizhicfs/chenyliao/llm-unlearning/checkpoints/full_cf07_formal` |

## C. R3-2 Audit

| Checkpoint / Model | P1 forget | P2 gain | P3 gain | INT8 shift | mean_recovery |
|---|---:|---:|---:|---:|---:|
| base | 0.4832 | -0.1203 | +0.0601 | +0.1932 | — |
| asu | 2.8927 | +0.2450 | +0.1990 | +0.0284 | 0.0921 |
| grad_ascent | 0.9431 | +0.0729 | +0.0738 | +0.1675 | 0.1595 |

100-step audit gate: `{'shallow_forgetting_detected': False, 'verdict': 'KILL'}`

3-seed cross-check (base/GradAscent/NPO):

| Model | mean_recovery | probe AUC pre | probe AUC post |
|---|---:|---:|---:|
| base | — | 0.9817 | 0.9817 |
| ga | 0.3772 | 0.9861 | 0.9871 |
| npo | 0.3815 | 0.9866 | 0.9866 |

3-seed gate: `{'probe_channel_present': True, 'shallow_forgetting_detected': True, 'verdict': 'PASS'}`

## D. Conclusions

- `CF-04` is the only method with a positive formal 10-epoch full-data run: dF `+0.6294`, dR `-0.4609`; holdout NLL increased `+0.8917`, so over-forgetting on unseen facts is the main risk.
- `CF-07` validates a clean readout-space property but fails as an unlearning method in formal 10 epochs: dF `+0.0266` is negligible.
- `R3-2` is the strongest benchmark/audit component, not a unlearning method. The 100-step ASU audit is `KILL` (mean_recovery `0.092`), while the GA/NPO 3-seed cross-check is `PASS`; the result is checkpoint/budget dependent.
- `CF-01`, `CF-08`, `CF-10` show non-trivial forget signals but fail the retain constraint; they are candidates for retain-constrained variants.
- `CF-02`, `R2-C`, `R3-3` diverge or collapse; not suitable for full training without stabilization.
- `CF-03`, `CF-05` are analysis-only and currently inconclusive; `CF-06` belongs to the LRM track.

## E. Applicability

- Retain-sensitive unlearning: `CF-04` (with holdout guard).
- Robustness benchmark: `R3-2`.
- Representation analysis / regularizer: `CF-07`.
- Component ideas needing retain constraints: `CF-01`, `CF-08`, `CF-10`.
- Not ready: `CF-02`, `R2-C`, `R3-3`.
- Other track: `CF-06`.

## Constraint
No ARIS claim gate, review, audit, or claim artifact was produced. This table is diagnostic evidence, not a paper-level leaderboard.
