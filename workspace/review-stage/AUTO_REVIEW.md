# CF-04 v2 Optimization Review

- claim_level: `false`
- evidence_gate: `NOT_APPLIED`
- reviewer_backend: `manual` / `hy4`
- review_status: `REVIEW_UNAVAILABLE`
- reason: a genuine Hy4 review call returned only `reasoning_content` with empty final `content` at an 8192 token budget. A forced-format probe that echoed a caller-supplied score is not a review and is not used. No score is fabricated.

## Artifacts Reviewed

- `refine-logs/CF04_V2_OPTIMIZATION_REPORT.md`
- `auto_experiments/results/cf04_v2_300/`
- `auto_experiments/results/formal_training/CF-04_v2_formal.json`
- `auto_experiments/results/full_repro_sweep/train_global_s300_seed{0,1,2}.json`
- `experiments/diagnostic_methods/cf04_dual_probes.py`
- `experiments/diagnostic_methods/run_one_method.py`

## Experimental Summary

- CF-04 v2g 300-step 3-seed: dF +2.3084 ± 0.0915, dR +0.0180 ± 0.0031, dH +0.1754 ± 0.0131.
- CF-04 v2g 10-epoch: dF +8.0920, dR -0.4561, dH +0.9656.
- ASU 300-step 3-seed: dF +3.1780 ± 0.1492, dR +0.1730 ± 0.0069, dH +0.0340 ± 0.0298.
- CF-04 v2 is retain-safe and closes most of the 300-step forget gap; ASU remains more holdout-stable.

## Gate Status

`REVIEW_UNAVAILABLE` is terminal for this run. No claim is made.
