"""QKSpec trainer wrapper.

QKSpec is a one-shot, gradient-free weight edit.  We override ``train()`` so
that instead of entering the HF training loop we:

1. build independent D_F / D_R loaders from the ``ForgetRetainDataset``,
2. run :func:`qkspec.run_qkspec_edit` (stage A -> stage B -> fold -> controls
   -> full D_F/D_R eval),
3. write the structured report.

Model saving and the external TOFU evaluator still run from the launcher
(``workspace/src/run_qkspec.py``, mirroring upstream ``src/train.py``) after
``train()`` returns, exactly as for other trainers.  No ``compute_loss`` /
``training_step`` is needed.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from qkspec import QKSpecConfig, run_qkspec_edit
from qkspec.report import build_report, write_report
from trainer.unlearn.base import UnlearnTrainer

logger = logging.getLogger(__name__)


def _extract_forget_retain(train_dataset):
    """Return the two source datasets from a ForgetRetainDataset (or raise)."""
    if train_dataset is None:
        raise ValueError("QKSpec requires a train_dataset (ForgetRetainDataset)")
    forget = getattr(train_dataset, "forget", None)
    retain = getattr(train_dataset, "retain", None)
    if forget is None or retain is None:
        raise ValueError(
            "QKSpec requires a ForgetRetainDataset exposing .forget and .retain; "
            "got " + type(train_dataset).__name__
        )
    return forget, retain


class QKSpecTrainer(UnlearnTrainer):
    """Thin Trainer wrapper that performs a single QKSpec weight edit."""

    def __init__(self, *args, qkspec: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.qkspec_cfg = QKSpecConfig.from_any(qkspec)
        self.qkspec_result = None

    def _build_loader(self, dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
        collator = self.data_collator
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collator,
            num_workers=0,
        )

    def train(self, *args, **kwargs):  # type: ignore[override]
        cfg = self.qkspec_cfg
        model = self.model
        device = model.device if hasattr(model, "device") else next(model.parameters()).device

        # Discovery requires eager attention (flash/sdpa do not return weights).
        attn_impl = getattr(model.config, "_attn_implementation", None)
        if attn_impl not in (None, "eager"):
            raise RuntimeError(
                f"QKSpec discovery requires attn_implementation='eager', "
                f"but the model was loaded with '{attn_impl}'. Set "
                "model.model_args.attn_implementation: eager in the experiment config."
            )

        forget_ds, retain_ds = _extract_forget_retain(self.train_dataset)
        forget_loader = self._build_loader(forget_ds, cfg.discovery_batch_size, shuffle=False)
        retain_loader = self._build_loader(retain_ds, cfg.discovery_batch_size, shuffle=False)

        model.eval()
        result = run_qkspec_edit(
            model=model,
            forget_loader=forget_loader,
            retain_loader=retain_loader,
            cfg=cfg,
            output_dir=getattr(self.args, "output_dir", None),
        )
        self.qkspec_result = result

        report = build_report(
            cfg,
            result,
            forget_metrics=result.forget_metrics,
            retain_metrics=result.retain_metrics,
        )
        out_dir = getattr(self.args, "output_dir", None)
        path = write_report(report, out_dir)
        result.report_path = path
        if path:
            logger.info("QKSpec report written to %s", path)
        logger.info(
            "QKSpec edit complete: %d directions, update_norm=%.6f",
            len(result.selected),
            result.update_norm,
        )

        # Return an object with a state_dict-like attribute is not required;
        # HF Trainer.save_state() is called by train.py after train() returns.
        return result
