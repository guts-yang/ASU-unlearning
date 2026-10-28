# Stage 1 preflight record

This record contains only the non-GPU identity check completed at 2026-10-28T16:28:00+08:00. It is not an experiment log. No package was installed, and no training or smoke test was started.

Repository at check time: branch `fuxian`, commit `47ba51c88d11689ab51faa60beb9143961a29993`.

## Executed checks

```text
/root/miniconda3/bin/python3 --version
```

Observed: `Python 3.12.3`.

Package inventory against `Right-to-be-forgotten/requirements.txt`:

```text
torch==2.8.0+cu128
transformers==4.46.3
datasets==5.0.1
accelerate==1.14.0
deepspeed==0.19.3
hydra-core==1.3.4
jsonlines=NOT_INSTALLED
nltk==3.10.0
numpy==2.3.2
omegaconf==2.3.1
peft==0.20.0
PyYAML==6.0.2
rouge_score==0.1.2
scikit-learn==1.9.0
scipy==1.18.0
sentence_transformers=NOT_INSTALLED
tqdm==4.70.0
bitsandbytes==0.50.0
```

`requirements.txt` pins `transformers==4.43.0`. The installed version does not match that pin. `metrics.py` imports `jsonlines` and `sentence_transformers`; both are absent. They were not installed.

Data identity for `split=forget05` and `TASK_LIST=1`:

- `forget05.json` has 2000 rows, 200 per `task_id` 1 through 10. Task 1 is the forget set: 200 rows. SHA-256 `a8c207a905af73ea4797b1af92a14cf458f42df567c941f7284f87d930096f4b`.
- `forget.py` resolves the retain filename with `100 - min(10 * 5, 90)`, which is `retain50.json`, not `retain95.json`. That file has 2000 rows and SHA-256 `6dcc828b669c09b1a537776ef7ac80d7e51a9e2d76af5c827543c3c01086625e`.
- Rows in `forget05.json` with `task_id != 1` add 1800 retain rows. Question overlap with `retain50.json` is 0. Effective retain size is 3800.
- `retain95.json` is not in `Right-to-be-forgotten/data/tofu`.
- Evaluation retain file `retain_perturbed.json` has 400 rows. SHA-256 `fc69f33bad70d3dca65920bdb54380039e3e8872dd16bbae20c1a839b99533fb`.

Full file hashes are in `workspace/H/derived/stage1_data_identity.json`.

Checkpoint:

- Path: `/root/autodl-tmp/models/tofu_ft_llama2-7b`
- Snapshot: `8fa500e8f345f1dd9cfe95bb4689878c944c9cbd`
- Three safetensors shards were readable. Their SHA-256 values are `403a43e265a6fbb8a59e9b21813a837758a59b7cb4bd399ef7f5af6664749ada`, `c65fe3be1daab5d6d5b934c322235a31ae8356dd7f444d0926ecc655ca2fc902`, and `3b3fb2e1a8265cfc66d3ef5c99cc5a1c4e9a2b1f9f0d91b038aaf36e12b51ae3`.
- The model was not loaded.

The rest of the environment record is `workspace/H/derived/stage1_checkpoint_environment.json`.

GPU discovery:

```text
command -v nvidia-smi
nvidia-smi -L
ls /dev/nvidia*
```

Observed: `/usr/bin/nvidia-smi` exists and `nvidia-smi -L` exited 0 with no device lines. `/dev/nvidia*` does not exist. No GPU is discoverable in this shell.

## Required commands after the remaining inputs are frozen

`configs/global_asu_forget05.template.yaml` now has data paths, data hashes, and the checkpoint snapshot. `seeds` and `gpu_budget_hours` are still null, so the commands below are not authorized.

`Right-to-be-forgotten/config/tofu.yaml` defaults to `attention_temp: 2.0` and `layers_id: []`. An empty list does not mean every layer. The overrides below are required; otherwise no layer receives the temperature.

```bash
cd Right-to-be-forgotten
export TASK_LIST=1
CUDA_VISIBLE_DEVICES=<train_gpus> torchrun --nproc_per_node=<n_train_gpus> \
  forget.py --config-name=tofu.yaml \
  split=forget05 forget_loss=ASU+GD attention_temp=2.3 layers_id=null \
  forget_coeff=0.1 regularization_coeff=1.0 lr=1e-5 num_epochs=5 \
  mask=true fix_ref_model=true seed=<seed> \
  model_path=/root/autodl-tmp/models/tofu_ft_llama2-7b \
  data_path=/usr/local/ASU-unlearning/Right-to-be-forgotten/data/tofu \
  save_root=<immutable_run_root>

CUDA_VISIBLE_DEVICES=<eval_gpu> torchrun --nproc_per_node=1 \
  eval.py --config-name=tofu.yaml \
  split=forget05 forget_loss=ASU+GD attention_temp=2.3 layers_id=null \
  forget_coeff=0.1 regularization_coeff=1.0 lr=1e-5 num_epochs=5 \
  mask=true fix_ref_model=true seed=<seed> \
  model_path=/root/autodl-tmp/models/tofu_ft_llama2-7b \
  data_path=/usr/local/ASU-unlearning/Right-to-be-forgotten/data/tofu \
  save_root=<immutable_run_root>
```

Before a real invocation, record the exact resolved Hydra config, commit, hashes, hardware and start timestamp in the run directory. Do not substitute historical CSV values for missing execution output.
