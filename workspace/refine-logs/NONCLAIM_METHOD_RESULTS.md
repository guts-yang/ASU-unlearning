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
| 4 | CF-04 | 3 | DIAGNOSTIC_NEGATIVE | +0.2599 | +0.2181 | dual probe AUC + 3-step train |
| 5 | CF-06 | 0 | NOT_APPLICABLE | — | — | LRM applicability |
| 6 | CF-05 | 0 | DIAGNOSTIC_INCONCLUSIVE | +0.0000 | +0.0000 | multi-donor SVD |
| 7 | CF-08 | 3 | DIAGNOSTIC_NEGATIVE | +0.2840 | +0.2238 | nullspace + 3-step train |
| 8 | CF-07 | 3 | DIAGNOSTIC_NEGATIVE | +0.8458 | +0.8376 | readout rotation + 3-step train |
| 9 | CF-09 | 3 | DIAGNOSTIC_NEGATIVE | +0.0122 | -0.0019 | control residual + 3-step train |
| 10 | CF-10 | 3 | DIAGNOSTIC_NEGATIVE | +0.5226 | +0.5586 | purity schedule + 3-step train |
| 11 | R2-C | 2 | DIAGNOSTIC_NEGATIVE | +0.4930 | +0.2471 | inner attack + minimax |
| 12 | R3-2 | 3-seed | DIAGNOSTIC_POSITIVE | — | — | probe + 3-seed full audit |
| 13 | R3-3 | 3 | DIAGNOSTIC_NEGATIVE | +0.4705 | +0.2539 | two-stage schedule + 3-step train |

## R3-2 Probe + 3-seed Full Audit

Output: `workspace/auto_experiments/results/audit_depth_probe_3seed.json`

| Model | P1 forget | P2 gain | P3 gain | mean_recovery | probe AUC pre | probe AUC post |
|---|---:|---:|---:|---:|---:|---:|
| base | 0.4621 | -0.0158 | +0.0660 | — | 0.982 | 0.982 |
| GradAscent | 0.9335 | +0.2607 | +0.0949 | 0.3772 ± 0.0018 | 0.986 | 0.987 |
| NPO | 0.9323 | +0.2611 | +0.0977 | 0.3815 ± 0.0019 | 0.987 | 0.987 |

- Seeds: `0,1,2`; P3 uses seeded shuffled relearning.
- Gate: `PASS`; `probe_channel_present=true`.
- Probe AUC is near-saturated for base and unlearned models, so the probe channel is implemented but **not discriminative** in its current form.
- ASU 1B checkpoint is unavailable locally; `asu-forget05-asu-gd` is 7B/vocab-32000 and is rejected by the architecture guard. Historical ASU numbers remain in `audit_depth.json`.

## CF-04 / CF-07 Update

- `CF-04`: probe-guided loss implemented and trained 3 steps. Mechanism check stayed positive (`AUC_fact=1.000`, `AUC_identity=0.764`, margin `+0.236`), but training gave `dF=+0.2599`, `dR=+0.2181`, so retain safety failed.
- `CF-07`: readout-space loss implemented and trained 3 steps. Rotation invariance stayed positive (`1.26e-6`, cosine `1.000`), but training gave `dF=+0.8458`, `dR=+0.8376`, so retain safety failed.

## Full-Data 1-Epoch Training

See `FULL_TRAINING_RESULTS.md`.

| Method | forget Δ | retain Δ | Verdict |
|---|---:|---:|---|
| CF-04 | +0.1650 | -0.3048 | `DIAGNOSTIC_POSITIVE` |
| CF-07 | -0.1230 | -0.3069 | `DIAGNOSTIC_NEGATIVE` |

## Constraint
This is a non-claim diagnostic run. It does not invoke the ARIS evidence gate, does not write accepted/provisional state, and cannot be cited as faithful reproduction or paper evidence.
