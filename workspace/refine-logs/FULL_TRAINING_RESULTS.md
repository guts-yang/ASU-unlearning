# Full-Data Non-Claim Training Results

- claim_level: `false`
- evidence_gate: `NOT_APPLIED`
- run_kind: full-data non-claim training
- baseline: ASU/open-unlearning TOFU `forget10/retain90/holdout10`, `Llama-3.2-1B-Instruct`
- hardware: one NVIDIA H20
- data: 400 forget / 3600 retain / 400 holdout
- schedule: 1 epoch, batch 2, grad accumulation 8, single GPU
- evaluation: full forget/retain/holdout NLL + 8-sample generation behavior

## Runs

| Method | LR | retain weight | Early stopped | forget Δ | retain Δ | holdout Δ | Checkpoint |
|---|---:|---:|---|---:|---:|---:|---|
| CF-04 probe-guided | 5e-6 | 3.0 | no | **+0.1650** | **-0.3048** | +0.1850 | `/jizhicfs/chenyliao/llm-unlearning/checkpoints/full_cf04_e1` |
| CF-07 readout-space | 2e-6 | 5.0 | no | -0.1230 | -0.3069 | +0.2228 | `/jizhicfs/chenyliao/llm-unlearning/checkpoints/full_cf07_e1` |

## CF-04 — DIAGNOSTIC_POSITIVE

- Before: forget `0.4832`, retain `0.4769`, holdout `1.9878`
- After: forget `0.6482`, retain `0.1721`, holdout `2.1728`
- Forget NLL increased `+0.1650` while retain NLL decreased `-0.3048`
- Completed all 1800 steps; no early stop
- Probe mechanism remained valid (`AUC_fact=1.000`, `AUC_identity=0.764`, margin `+0.236`)
- This is the first full-data run in this track with both a forget signal and retain improvement

## CF-07 — DIAGNOSTIC_NEGATIVE

- Before: forget `0.4832`, retain `0.4769`, holdout `1.9878`
- After: forget `0.3602`, retain `0.1699`, holdout `2.2106`
- Forget NLL decreased `-0.1230` (wrong direction for unlearning)
- Retain NLL improved `-0.3069`, but there is no forget signal
- Rotation invariance was validated earlier, but the readout-space loss did not produce actual forgetting

## Interpretation

`CF-04` is now the only candidate with a full-data diagnostic signal that is retain-safe in this run. `CF-07` fails the primary forgetting criterion despite its clean representation property.

## Limitations

- 1 epoch, single GPU, 1 seed; not the paper's 10-epoch multi-GPU protocol
- NLL and 8-sample generation behavior only; official TOFU metrics are not yet computed
- No ARIS claim gate, review, audit, or claim artifact
