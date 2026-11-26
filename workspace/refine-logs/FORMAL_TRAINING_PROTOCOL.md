# Formal Multi-Epoch Training Protocol

- claim_level: `false`
- evidence_gate: `NOT_APPLIED`
- formal_epochs: `10`
- source: `open-unlearning/configs/experiment/unlearn/tofu/default.yaml` → `num_train_epochs: 10`
- baseline data: TOFU `forget10` (400), `retain90` (3600), `holdout10` (400)
- model: `Llama-3.2-1B-Instruct`
- device: single H20, float32, eager attention

## Uniform Epoch Budget

All epoch-trainable candidates use exactly the same 10-epoch budget:

```text
CF-01, CF-02, CF-04, CF-07, CF-08, CF-09, CF-10, R2-C, R3-3
```

Analysis/audit-only candidates do not have an epoch loop and keep their established protocols:

```text
CF-03 patch-effect analysis
CF-05 multi-donor SVD
CF-06 LRM/CoT (NOT_APPLICABLE on this track)
R3-2 P1/P2/P3/INT8 + Probe audit
```

## Formal Run Command

```bash
cd workspace/experiments
python -m diagnostic_methods.run_one_method \
  --method CF-04 \
  --formal \
  --batch-size 2 \
  --grad-accum 8 \
  --lr 5e-6 \
  --seed 0
```

Formal mode:

- ignores `--steps`
- uses `FORMAL_EPOCHS = 10`
- disables retain-guard early stopping
- visits every retain90 record each epoch
- logs one `epoch_summary` row per epoch
- writes final full forget/retain/holdout NLL and behavior stats

## Debug Mode

Short 1–3 step runs are retained only for code debugging:

```bash
python -m diagnostic_methods.run_one_method --method CF-04 --steps 3
```

The old 1-epoch full-data runner is retained as debug only:

```bash
python -m diagnostic_methods.run_full_training --method CF-04 --mode debug
```

## Reproducibility

Each formal result records:

- git SHA
- method, epochs, batch size, grad accumulation, LR, retain weight
- seed and per-epoch seeded data order
- epoch-level logs and final metrics
- checkpoint path

No formal run may use early stopping; if a run is interrupted, it is incomplete and must not be reported as a formal result.
