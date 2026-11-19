# ASU × LeakRU Sanity Results

FQ reported in percentage points of `1 - Acc(θu)/Acc(θp)`. RR in percent.

| Method | FQ_SH | FQ_HS | FQ_MP | FQ_LL | FQ_CC | FQ_DS | FQ_MT | RR_FocusOnKey | RR_Probab |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ASU | 56.2 | 50.0 | NA | 100.0 | 40.0 | 25.0 | 71.4 | 32.4 | 42.6 |
| GA | 55.7 | 0.0 | NA | 50.0 | 20.0 | 25.0 | 57.1 | 43.5 | 56.5 |
| NPO | 40.7 | 0.0 | NA | 50.0 | 0.0 | 25.0 | 57.1 | 53.7 | 75.5 |

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
  "theta_p_tofu_forget05_rouge": 0.9625446736275544,
  "theta_p_tofu_exact_memorization": 0.9987873476743698,
  "acc_sh_theta_p": 0.97,
  "asu_tau": 2.3,
  "asu_alpha": 0.1,
  "asu_retain": "NLL",
  "epochs": 5,
  "lr": 1e-05,
  "asu_steps": 60,
  "asu_epoch": 4.64,
  "asu_train_loss": 3.661335245768229,
  "asu_ckpt": "/root/autodl-tmp/saves/workspace/unlearn/asu_tofu05_seed1001",
  "ga_definition": "OpenUnlearning GradDiff: forget gradient ascent + retain NLL",
  "ga_ckpt": "/usr/local/open-unlearning/workspace/saves/unlearn/ga_tofu05_seed1001",
  "ga_candidate": "default",
  "ga_lr": 1e-05,
  "ga_fq_sh": 0.5567010309278351,
  "npo_ckpt": "/usr/local/open-unlearning/workspace/saves/unlearn/npo_tofu05_seed1001",
  "npo_candidate": "default",
  "npo_beta": 0.1,
  "npo_lr": 1e-05,
  "npo_fq_sh": 0.4072164948453608,
  "asu_fq_sh": 0.5618556701030928,
  "controls_matched": false,
  "fq_match_rule": "select checkpoint with FQ_SH closest to ASU; sweep only if default gap > 10pp; do not use DS/MT/RR",
  "data": "/usr/local/ASU-unlearning/workspace/data/leakru/tofu_forget05.jsonl",
  "attacks": [
    "probab",
    "focus_on_key"
  ],
  "quantization": false,
  "probab": {
    "temperature": 0.8,
    "top_p": 0.95,
    "trials": 5
  },
  "focus_on_key_repeats": 3,
  "gpu_status": "ok",
  "gpu_detail": "2x A800 80GB"
}
```
