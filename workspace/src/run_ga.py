"""Compatibility entry for GradAscent on transformers 4.51.

Patches the upstream GradAscent.compute_loss to accept ``num_items_in_batch``
(new signature in transformers>=4.46) without modifying the upstream repo.
Then mirrors train.py with workspace configs on the Hydra search path.
"""
from __future__ import annotations

import os
import sys


def _bootstrap() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", ".."))
    ws_src = os.path.join(repo, "workspace", "src")
    ou_src = os.path.join(repo, "open-unlearning", "src")
    ws_configs = os.path.join(repo, "workspace", "configs")
    for p in (ws_src, ou_src):
        if p not in sys.path:
            sys.path.insert(0, p)
    if not any(a.startswith("hydra.searchpath=") for a in sys.argv):
        sys.argv.append(f"hydra.searchpath=[file://{os.path.join(repo, 'open-unlearning', 'configs')}]")
    # ensure workspace configs remain reachable too
    sys.argv.append(f"hydra.searchpath=[file://{ws_configs},file://{os.path.join(repo, 'open-unlearning', 'configs')}]")


_bootstrap()

import hydra
from omegaconf import DictConfig

from data import get_data, get_collators
from evals import get_evaluators
from model import get_model
from trainer import TRAINER_REGISTRY, load_trainer
from trainer.unlearn.grad_ascent import GradAscent
from trainer.utils import seed_everything

# Patch compute_loss for transformers>=4.46 signature.
_orig_compute_loss = GradAscent.compute_loss


def _patched_compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    return _orig_compute_loss(self, model, inputs, return_outputs=return_outputs)


GradAscent.compute_loss = _patched_compute_loss
TRAINER_REGISTRY["GradAscent"] = GradAscent


@hydra.main(version_base=None, config_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs"),
            config_name="unlearn.yaml")
def main(cfg: DictConfig):
    seed_everything(cfg.trainer.args.seed)
    mode = cfg.get("mode", "unlearn")
    model, tokenizer = get_model(cfg.model)
    data = get_data(cfg.data, mode=mode, tokenizer=tokenizer, template_args=cfg.model.template_args)
    collator = get_collators(cfg.collator, tokenizer=tokenizer)
    evaluators = None
    if cfg.get("eval", None):
        evaluators = get_evaluators(eval_cfgs=cfg.eval, template_args=cfg.model.template_args,
                                    model=model, tokenizer=tokenizer)
    trainer, trainer_args = load_trainer(
        trainer_cfg=cfg.trainer, model=model,
        train_dataset=data.get("train", None), eval_dataset=data.get("eval", None),
        tokenizer=tokenizer, data_collator=collator,
        evaluators=evaluators, template_args=cfg.model.template_args,
    )
    if trainer_args.do_train:
        trainer.train()
        trainer.save_state()
        trainer.save_model(trainer_args.output_dir)
    if trainer_args.do_eval:
        trainer.evaluate(metric_key_prefix="eval")


if __name__ == "__main__":
    main()
