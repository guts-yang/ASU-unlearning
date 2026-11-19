# ASU × LeakRU minimal sanity check

Orchestration lives here; runtime is `/usr/local/open-unlearning` (LeakRU plugin + Hydra).

## Quick start

```bash
# build data (CPU ok)
python workspace/scripts/build_tofu_leakru.py

# preflight only (no GPU, no training)
bash workspace/scripts/prepare_asu.sh

# later, with GPU: train ASU only
# see workspace/notes/ready_asu.md
```

## Layout

| Path | Role |
|------|------|
| `src/open_unlearning_plugins/asu/` | ASU trainer + attention-τ helper (symlinked into OU) |
| `configs/experiment/unlearn/{asu,ga,npo}_tofu05.yaml` | Train configs |
| `configs/experiment/eval/leakru_tofu05_*.yaml` | LeakRU eval |
| `data/leakru/tofu_forget05.jsonl` | forget05-adapted LeakRU protocol |
| `scripts/run_sanity.sh` | End-to-end |
| `results/` | FQ/RR table + H1/H2 verdict |
