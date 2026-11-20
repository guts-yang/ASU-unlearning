"""FE/MU evaluation with the student forced to float32.

The GPU node's CUDA 12.2 / cuBLAS triggers a SIGFPE in bf16 GEMM inside
model.generate() during ROUGE scoring (retain_Q_A_ROUGE). Running generation
in float32 avoids it. 1B model in fp32 is ~5GB, well within the 96GB H20.

This reproduces exactly the FE (forget_quality / KS-test) and MU
(model_utility / 9-metric harmonic mean) definitions from open-unlearning.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

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
    model_cfg = cfg.model
    model, tokenizer = get_model(model_cfg)
    # Force fp32 to avoid the bf16 cuBLAS SIGFPE during generation.
    model = model.float()
    model.eval()
    evaluators = get_evaluators(cfg.eval)
    for name, evaluator in evaluators.items():
        evaluator.evaluate(
            template_args=model_cfg.template_args,
            model=model,
            tokenizer=tokenizer,
        )
    print("FE/MU FP32 EVAL DONE")


if __name__ == "__main__":
    main()
