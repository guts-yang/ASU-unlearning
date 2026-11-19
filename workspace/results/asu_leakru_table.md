# ASU × LeakRU Sanity Results

FQ reported in percentage points of `1 - Acc(θu)/Acc(θp)`. RR in percent.

| Method | FQ_SH | FQ_HS | FQ_MP | FQ_LL | FQ_CC | FQ_DS | FQ_MT | RR_FocusOnKey | RR_Probab |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ASU | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| GA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| NPO | NA | NA | NA | NA | NA | NA | NA | NA | NA |

## Paths
- θ_p: `/usr/local/open-unlearning/workspace/saves/eval/leakru_tofu05_theta_p/LeakRU_EVAL.json`
- ASU: `/usr/local/open-unlearning/workspace/saves/eval/leakru_tofu05_asu/LeakRU_EVAL.json`
- GA: `/usr/local/open-unlearning/workspace/saves/eval/leakru_tofu05_ga/LeakRU_EVAL.json`
- NPO: `/usr/local/open-unlearning/workspace/saves/eval/leakru_tofu05_npo/LeakRU_EVAL.json`

## Run meta
```json
{
  "seed": 1001,
  "forget_split": "forget05",
  "retain_split": "retain95",
  "model_theta_p": "open-unlearning/tofu_Llama-2-7b-chat-hf_full",
  "asu_tau": 2.3,
  "asu_alpha": 0.1,
  "epochs": 5,
  "lr": 1e-05,
  "train": 0,
  "reuse_asu": 1,
  "asu_ckpt_reuse": "/root/autodl-tmp/ASU-unlearning/checkpoints/tofu_replay/llama2-7b/forget05/ASU+GD/seed_1001/epoch5_1e-05_FixRefTrue_maskTrue_Fcoeff0.1_Rcoeff1.0_Temp2.3_LayersNone_beta0.1/1/unlearn_times_1/checkpoint-last",
  "data": "/usr/local/ASU-unlearning/workspace/data/leakru/tofu_forget05.jsonl",
  "attacks": [
    "probab",
    "focus_on_key"
  ],
  "gpu_status": "blocked",
  "gpu_detail": "torch.cuda.is_available() is False (device open may return EPERM). Scripts/data ready."
}
```

## Missing evals
- /usr/local/open-unlearning/workspace/saves/eval/leakru_tofu05_asu/LeakRU_EVAL.json
- /usr/local/open-unlearning/workspace/saves/eval/leakru_tofu05_ga/LeakRU_EVAL.json
- /usr/local/open-unlearning/workspace/saves/eval/leakru_tofu05_npo/LeakRU_EVAL.json
