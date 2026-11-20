"""Self-contained QKSpec entry point (kept entirely under ``workspace/``).

It mirrors ``open-unlearning/src/train.py`` but:

* resolves the primary Hydra config from ``workspace/configs``;
* adds ``open-unlearning/configs`` as an extra Hydra search path so the base
  config groups (model/data/eval/collator/hydra/paths) remain available;
* registers the QKSpec trainer handler at runtime (the upstream
  ``trainer/__init__.py`` is intentionally left untouched).

Usage (from anywhere)::

    python workspace/src/run_qkspec.py --config-name=unlearn.yaml \
        experiment=unlearn/tofu/qkspec.yaml trainer=QKSpec task_name=...
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import hydra
from omegaconf import DictConfig


def _bootstrap_paths() -> None:
    """Make both the QKSpec package and the upstream framework importable."""
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", ".."))
    ws_src = os.path.join(repo_root, "workspace", "src")
    ou_src = os.path.join(repo_root, "open-unlearning", "src")
    for path in (ws_src, ou_src):
        if path not in sys.path:
            sys.path.insert(0, path)

    # Inject the upstream config dir as an extra Hydra search path unless the
    # caller already specified one. This lets trainer=QKSpec / experiment=
    # unlearn/tofu/qkspec.yaml resolve from workspace/configs while model/,
    # data/, eval/ ... resolve from open-unlearning/configs.
    if not any(arg.startswith("hydra.searchpath=") for arg in sys.argv):
        ou_configs = os.path.join(repo_root, "open-unlearning", "configs")
        sys.argv.append(f"hydra.searchpath=[file://{ou_configs}]")


_bootstrap_paths()

from data import get_collators, get_data  # noqa: E402
from evals import get_evaluators  # noqa: E402
from model import get_model  # noqa: E402
from qkspec.trainer import QKSpecTrainer  # noqa: E402
from trainer import TRAINER_REGISTRY, load_trainer  # noqa: E402
from trainer.utils import seed_everything  # noqa: E402


# Register the handler at runtime; upstream trainer/__init__.py is untouched.
TRAINER_REGISTRY.setdefault("QKSpec", QKSpecTrainer)
TRAINER_REGISTRY.setdefault("QKSpecTrainer", QKSpecTrainer)


@hydra.main(
    version_base=None,
    config_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs"),
    config_name="unlearn.yaml",
)
def main(cfg: DictConfig):
    seed_everything(cfg.trainer.args.seed)
    mode = cfg.get("mode", "unlearn")
    model_cfg = cfg.model
    template_args = model_cfg.template_args
    assert model_cfg is not None, "Invalid model yaml passed in config."
    model, tokenizer = get_model(model_cfg)

    data_cfg = cfg.data
    data = get_data(data_cfg, mode=mode, tokenizer=tokenizer, template_args=template_args)

    collator = get_collators(cfg.collator, tokenizer=tokenizer)

    trainer_cfg = cfg.trainer
    assert trainer_cfg is not None, ValueError("Please set trainer")

    evaluators = None
    eval_cfgs = cfg.get("eval", None)
    if eval_cfgs:
        evaluators = get_evaluators(
            eval_cfgs=eval_cfgs,
            template_args=template_args,
            model=model,
            tokenizer=tokenizer,
        )

    trainer, trainer_args = load_trainer(
        trainer_cfg=trainer_cfg,
        model=model,
        train_dataset=data.get("train", None),
        eval_dataset=data.get("eval", None),
        tokenizer=tokenizer,
        data_collator=collator,
        evaluators=evaluators,
        template_args=template_args,
    )

    if trainer_args.do_train:
        trainer.train()
        trainer.save_state()
        trainer.save_model(trainer_args.output_dir)

    if trainer_args.do_eval:
        trainer.evaluate(metric_key_prefix="eval")


if __name__ == "__main__":
    main()
