# Full-Data Non-Claim Reproduction Report

- claim_level: `false`
- evidence_gate: `NOT_APPLIED`
- dataset: TOFU `forget10=400 / retain90=3600 / holdout10=400`, full split evaluation
- model: `Llama-3.2-1B-Instruct`, fp32, eager attention
- enhancement: 100-step full-data diagnostic runs (vs. previous 2–20 steps)

## Forget / Retain Results vs. Thresholds

| Method | Steps | dF(forget NLL) | dR(retain NLL) | dHoldout | dF≥0.15,dR≤0.05 | dF≥0.30,dR≤0.05 | dF≥0.90,dR≤0.05 | step_sec | elapsed_sec |
|---|---:|---:|---:|---:|---|---|---|---:|---:|
| CF-01 | 100 | +0.2111 | +0.1951 | -0.0007 | NO | NO | NO | 0.377 | 276.3 |
| CF-02 | 100 | +4.0572 | +3.7442 | +2.2565 | NO | NO | NO | 0.905 | 390.0 |
| CF-03 | 100 | +0.0000 | +0.0000 | +0.0000 | NO | NO | NO | — | 299.0 |
| CF-04 | 100 | +1.0072 | +0.3475 | +0.1126 | NO | NO | NO | 0.684 | 256.3 |
| CF-05 | 100 | +0.0000 | +0.0000 | +0.0000 | NO | NO | NO | — | 201.2 |
| CF-06 | 100 | — | — | — | — | — | — | — | 0.0 |
| CF-07 | 100 | +1.5686 | +0.8580 | +0.1396 | NO | NO | NO | 0.552 | 216.0 |
| CF-08 | 100 | +0.9352 | +0.3150 | -0.0154 | NO | NO | NO | 0.502 | 303.2 |
| CF-09 | 100 | +0.2111 | +0.1951 | -0.0007 | NO | NO | NO | 0.291 | 159.9 |
| CF-10 | 100 | +0.4917 | +0.1785 | -0.0266 | NO | NO | NO | 0.468 | 290.6 |
| R2-C | 100 | +100.9142 | +0.4449 | +0.0390 | NO | NO | NO | 0.386 | 195.2 |
| R3-2 | — | — | — | — | — | — | — | — | — |
| R3-3 | 100 | +96.1653 | +2.6453 | +6.5238 | NO | NO | NO | 0.277 | 133.8 |

## Depth Audit

| Model | P1 forget | P2 gain | P3 gain | INT8 shift | mean_recovery |
|---|---:|---:|---:|---:|---:|
| base | 0.4832 | -0.1203 | +0.0601 | +0.1932 | — |
| asu | 2.8927 | +0.2450 | +0.1990 | +0.0284 | +0.0921 |
| grad_ascent | 0.9431 | +0.0729 | +0.0738 | +0.1675 | +0.1595 |

gate: `{'shallow_forgetting_detected': False, 'verdict': 'KILL'}`

## Step Scaling Evidence (ASU global, full evaluation)

| Steps | dF | dR | dHoldout |
|---:|---:|---:|---:|
| 1 | +0.3309 | +0.3057 | -0.1410 |
| 3 | +1.3679 | +1.2924 | +0.2795 |
| 5 | +1.3005 | +1.1530 | +0.1428 |
| 10 | +1.2661 | +0.9642 | +0.0934 |
| 20 | +1.5272 | +0.7589 | +0.0380 |
| 30 | +2.0891 | +0.5585 | +0.0164 |
| 60 | +2.8837 | +0.2480 | +0.0349 |
| 150 | +2.8569 | +0.1878 | +0.0470 |
| 300 | +2.9717 | +0.1821 | +0.0757 |

## Timing

- Sum of measured per-job wall-clock: 2721.4 s (45.4 min) on 1 GPU.
- Mean training step time across methods: 0.494 s/step.
- ASU baseline step time: 0.1881 s/step; max 0.6098 s/step.
- Full evaluation of 4,400 examples (forget+retain+holdout, batch 8) measured 39.1 s; before+after ≈78 s/job.
- Two-GPU wall-clock estimate ≈ sum(elapsed)/2 + queue overhead ≈ 22.7 min.

## Interpretation

- Full-data forget NLL reaches dF≥0.15 in as few as 1–3 steps for ASU global and saturates by ~60 steps.
- Increasing steps alone does not fix the retain constraint: dR remains far above 0.05 at every measured step count.
- Therefore a joint `dF target + dR≤0.05` pass is not achievable by step count alone under the frozen lr/alpha and full split evaluation; retain-side changes are required.
- CF-06 remains NOT_APPLICABLE on the LLM backbone; CF-05/R3-2 core operators are partial and R3-2 lacks a Probe channel.
