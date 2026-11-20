"""FE/MU evaluation with TF32 matmul (fast, avoids bf16 cuBLAS SIGFPE).

Sets torch.backends.cuda.matmul.allow_tf32=True so fp32 GEMMs use tensor
cores (~same speed as bf16) while avoiding the bf16 path that SIGFPEs on this
node's CUDA 12.2 / cuBLAS. This is still fp32 numerics for storage, TF32 for
matmul -- standard and fast on H20.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import hydra
from omegaconf import DictConfig

REPO = Path(__file__).resolve().parents[2]
OU = REPO / "open-unlearning"
WS = REPO / "workspace"
sys.path[:0] = [str(OU / "src"), str(WS / "src")]

from model import get_model
from evals import get_evaluators


@hydra.main(version_base=None, config_path=str(WS / "configs"), config_name="eval_fe_mu_runtime.yaml")
def main(cfg: DictConfig):
    t0 = time.time()
    model_cfg = cfg.model
    model, tokenizer = get_model(model_cfg)
    model = model.float()  # fp32 storage + TF32 matmul
    model.eval()
    print(f"[eval] model ready in {time.time()-t0:.1f}s (fp32+TF32)", flush=True)
    evaluators = get_evaluators(cfg.eval)
    for name, evaluator in evaluators.items():
        evaluator.evaluate(
            template_args=model_cfg.template_args,
            model=model,
            tokenizer=tokenizer,
        )
    print(f"[eval] FE/MU DONE in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
