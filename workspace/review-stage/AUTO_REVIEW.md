# CF-04 Optimization Review

- claim_level: `false`
- evidence_gate: `NOT_APPLIED`
- reviewer_backend: `manual` / `hy4`
- review_status: `REVIEW_UNAVAILABLE`
- reason: Hy4 returned only `reasoning_content` and empty final `content` after 4096 and 16384 token budgets; no valid Score/Verdict was produced. No score is fabricated.

## Artifacts Reviewed

- `refine-logs/CF04_OPTIMIZATION_REPORT.md`
- `auto_experiments/results/cf04_300/`
- `auto_experiments/results/full_repro_sweep/train_global_s300_seed{0,1,2}.json`
- `experiments/diagnostic_methods/cf04_dual_probes.py`
- `experiments/diagnostic_methods/run_one_method.py`

## Experimental Summary

- CF-04 baseline 300-step: dF +1.0395, dR +0.2908, dH +0.0155 (retain-unsafe).
- Best retain-safe CF-04 c11: 3-seed dF +0.5380 ± 0.0427, dR +0.0187 ± 0.0023, dH +0.0754 ± 0.0385.
- ASU 300-step: 3-seed dF +3.1780 ± 0.1492, dR +0.1730 ± 0.0069, dH +0.0340 ± 0.0298.
- CF-04 c11 is retain-safe but forgets much less; ASU forgets much more but is retain-unsafe.

## Gate Status

`REVIEW_UNAVAILABLE` is terminal for this run. The experiment result remains diagnostic and non-claim.
