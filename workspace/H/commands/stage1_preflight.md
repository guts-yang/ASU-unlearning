# Stage 1 preflight record

This record contains only completed non-training checks. It is not an experiment log.

## Executed checks

```text
python3 --version
python3 - <<'PY'
import importlib.metadata
for package in ('torch', 'transformers', 'datasets', 'accelerate', 'deepspeed', 'hydra-core'):
    try:
        print(f'{package}=={importlib.metadata.version(package)}')
    except importlib.metadata.PackageNotFoundError:
        print(f'{package}=NOT_INSTALLED')
PY
```

Observed:

```text
Python 3.9.6
torch==2.8.0
transformers=NOT_INSTALLED
datasets=NOT_INSTALLED
accelerate=NOT_INSTALLED
deepspeed=NOT_INSTALLED
hydra-core=NOT_INSTALLED
```

GPU discovery attempted:

```text
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
```

Observed: `nvidia-smi: command not found`. This does not prove that no GPU exists; it proves only that the active shell cannot query one through this utility.

## Required commands after inputs are supplied

The following is a template, not authorized for execution until the fields in `configs/global_asu_forget05.template.yaml` are frozen:

```bash
cd Right-to-be-forgotten
export TASK_LIST=1
CUDA_VISIBLE_DEVICES=<train_gpus> torchrun --nproc_per_node=<n_train_gpus> \
  forget.py --config-name=tofu.yaml \
  split=forget05 forget_loss=ASU+GD attention_temp=2.3 layers_id=null \
  forget_coeff=0.1 regularization_coeff=1.0 lr=1e-5 num_epochs=5 \
  mask=true fix_ref_model=true seed=<seed> \
  model_path=<checkpoint_path> data_path=<tofu_path> save_root=<immutable_run_root>

CUDA_VISIBLE_DEVICES=<eval_gpu> torchrun --nproc_per_node=1 \
  eval.py --config-name=tofu.yaml \
  split=forget05 forget_loss=ASU+GD attention_temp=2.3 layers_id=null \
  forget_coeff=0.1 regularization_coeff=1.0 lr=1e-5 num_epochs=5 \
  mask=true fix_ref_model=true seed=<seed> \
  model_path=<checkpoint_path> data_path=<tofu_path> save_root=<immutable_run_root>
```

Before a real invocation, record the exact resolved Hydra config, commit, hashes, hardware and start timestamp in the run directory. Do not substitute historical CSV values for missing execution output.
