"""Launch Right-to-be-forgotten/forget.py without editing that tree.

transformers 4.46 calls Trainer.compute_loss(..., num_items_in_batch=...).
CustomTrainerForgetting.compute_loss does not accept that keyword, so the
first step raises TypeError. This adapter accepts and ignores the extra
argument. The loss expression is unchanged.
"""
import os
import runpy
import sys
from pathlib import Path

ROOT = Path("/usr/local/ASU-unlearning/Right-to-be-forgotten")
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import trainer.trainer as trainer_mod

_orig = trainer_mod.CustomTrainerForgetting.compute_loss


def _compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
    del num_items_in_batch, kwargs
    return _orig(self, model, inputs, return_outputs=return_outputs)


trainer_mod.CustomTrainerForgetting.compute_loss = _compute_loss

sys.argv[0] = str(ROOT / "forget.py")
runpy.run_path(str(ROOT / "forget.py"), run_name="__main__")
