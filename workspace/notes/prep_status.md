# ASU × LeakRU sanity — prep status

- Seed: **1001**
- Split: TOFU `forget05` / `retain95`
- θ_p: `open-unlearning/tofu_Llama-2-7b-chat-hf_full`
- ASU: τ=2.3, α=0.1, 5 epochs, lr=1e-5（训练配置已就绪；默认评测复用 Rtbf ASU+GD ckpt）
- Data: `workspace/data/leakru/tofu_forget05.jsonl`（260 rows: 200 SH + 60 MH）
- Attacks: Probab + FocusOnKey（Quantization disabled）

## GPU

As of last check: driver node has `/dev/nvidiactl` but **no GPU device** (`torch.cuda.is_available()==False`).  
Train/eval blocked until a GPU is attached. Re-run:

```bash
cd /usr/local/ASU-unlearning
TRAIN=0 REUSE_ASU=1 bash workspace/scripts/run_sanity.sh
# or full train:
TRAIN=1 REUSE_ASU=0 bash workspace/scripts/run_sanity.sh
```

## Reuse ASU ckpt

`/root/autodl-tmp/ASU-unlearning/checkpoints/tofu_replay/llama2-7b/forget05/ASU+GD/seed_1001/epoch5_1e-05_FixRefTrue_maskTrue_Fcoeff0.1_Rcoeff1.0_Temp2.3_LayersNone_beta0.1/1/unlearn_times_1/checkpoint-last`
