# CF-04 300-Step Optimization and ASU Comparison

- claim_level: `false`
- evidence_gate: `NOT_APPLIED`
- model: `Llama-3.2-1B-Instruct`, fp32, eager attention
- data: TOFU `forget10=400 / retain90=3600 / holdout10=400`
- hardware: one NVIDIA H20, single GPU
- budget: 300 optimizer steps, batch size 4, full-split evaluation at batch 8
- seeds: sweep at seed 0; final candidates at seeds 0,1,2
- code: `workspace/experiments/diagnostic_methods/{run_one_method.py,cf04_dual_probes.py}`

## Metric Definitions

- `dF = after_forget_NLL - before_forget_NLL`; higher is better (more forgetting).
- `dR = after_retain_NLL - before_retain_NLL`; lower is better; retain-safe target `dR <= 0.05`.
- `dH = after_holdout_NLL - before_holdout_NLL`; lower is better (less collateral forgetting).
- `retain-safe pass`: `dF >= 0.30` and `dR <= 0.05`.
- `strong pass`: `dF >= 0.90` and `dR <= 0.05`.
- convergence speed: mean training step time and total wall-clock.
- stability: mean and population std over 3 seeds.

## Hyperparameter Sweep (300 steps, seed 0)

| Config | LR | identity w | probe retain w | retain NLL w | dF | dR | dH | Retain-safe |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 1e-5 | 0.5 | 0.5 | 0.5 | +1.0395 | +0.2908 | +0.0155 | no |
| c2 | 5e-6 | 1.0 | 1.0 | 2.0 | +0.4373 | -0.0361 | +0.0584 | yes |
| c3 | 2e-6 | 1.0 | 1.0 | 3.0 | +0.0826 | -0.2184 | +0.2055 | no (dF low) |
| c4 | 5e-6 | 2.0 | 0.5 | 3.0 | +0.3926 | -0.0370 | +0.1158 | yes |
| c5 | 1e-5 | 1.0 | 1.0 | 2.0 | +0.9564 | +0.2595 | +0.0156 | no |
| c6 | 8e-6 | 1.0 | 1.5 | 3.0 | +0.6662 | +0.1442 | +0.0329 | no |
| c7 | 5e-6 | 1.5 | 1.5 | 3.0 | +0.4199 | -0.0358 | +0.0669 | yes |
| c8 | 6e-6 | 1.5 | 2.0 | 4.0 | +0.5139 | +0.0401 | +0.0450 | yes |
| c9 | 7e-6 | 2.0 | 2.0 | 4.0 | +0.7449 | +0.0910 | +0.0517 | no |
| c10 | 6.5e-6 | 2.0 | 2.0 | 4.5 | +0.6015 | +0.0566 | +0.0380 | borderline |
| c11 | 6e-6 | 2.0 | 2.5 | 5.0 | +0.5360 | +0.0204 | +0.0481 | yes |
| c12 | 6e-6 | 2.0 | 3.0 | 6.0 | +0.5281 | +0.0212 | +0.0355 | yes |
| c13 | 7e-6 | 2.5 | 3.0 | 6.0 | +0.7511 | +0.0880 | +0.0127 | no |

The best retain-safe configuration is `c11` (LR `6e-6`, identity w `2.0`, probe-retain w `2.5`, retain-NLL w `5.0`). `c12` is a close second with slightly lower dF and slightly lower dH.

## 3-Seed Stability

| Method | dF mean±std | dR mean±std | dH mean±std | final forget | final retain | final holdout |
|---|---:|---:|---:|---:|---:|---:|
| CF-04 c11 | `+0.5380 ± 0.0427` | `+0.0187 ± 0.0023` | `+0.0754 ± 0.0385` | 1.0211 | 0.4956 | 2.0632 |
| CF-04 c12 | `+0.5085 ± 0.0145` | `+0.0243 ± 0.0090` | `+0.0688 ± 0.0393` | — | — | — |
| ASU global 300-step | `+3.1780 ± 0.1492` | `+0.1730 ± 0.0069` | `+0.0340 ± 0.0298` | 3.6589 | 0.6496 | 2.0218 |

## Final Comparison: CF-04 c11 vs ASU

| Dimension | CF-04 c11 | ASU 300-step | Winner |
|---|---:|---:|---|
| forget NLL after | 1.0211 | **3.6589** | ASU (stronger forgetting) |
| dF | +0.5380 | **+3.1780** | ASU |
| retain NLL after | **0.4956** | 0.6496 | CF-04 |
| dR | **+0.0187** | +0.1730 | CF-04 |
| dH | +0.0754 | **+0.0340** | ASU |
| step time | 0.334 s | **0.298 s** | ASU |
| dF stability | 0.043 std | 0.149 std | CF-04 (relative: both stable) |
| retain-safe | **yes** | no | CF-04 |

## Conclusion

- CF-04 can be made retain-safe at 300 steps, but only at a large forgetting cost: best safe dF is `+0.5380`, versus ASU `+3.1780`.
- The baseline CF-04 has strong forgetting (`dF=+1.0395`) but fails retain safety (`dR=+0.2908`).
- No tested CF-04 configuration reaches `dF >= 0.90` while keeping `dR <= 0.05`.
- CF-04 is therefore a retain-sensitive conservative method, not an ASU replacement.
- ASU remains the stronger unlearning baseline; CF-04 is preferable only when retain preservation is the dominant constraint.

## Independent Review

Hy4 review was attempted twice with 4096 and 16384 token budgets. It returned only `reasoning_content` and empty final `content`, so the review is recorded as `REVIEW_UNAVAILABLE`; no score or verdict is fabricated.

## Artifacts

```text
workspace/auto_experiments/results/cf04_300/<config>/CF-04.json
workspace/auto_experiments/results/full_repro_sweep/train_global_s300_seed{0,1,2}.json
workspace/review-stage/AUTO_REVIEW.md
workspace/review-stage/REVIEW_STATE.json
```
