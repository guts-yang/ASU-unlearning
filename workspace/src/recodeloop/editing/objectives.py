"""Local editing objective: residual recovery plus retain constraint."""

from __future__ import annotations

import torch


def editing_objective(
    reader_loss: torch.Tensor,
    retain_nll: torch.Tensor,
    lambda_ret: float,
) -> torch.Tensor:
    """Minimize L_rec + lambda_ret * L_ret.

    With a CE reader, A_r = -CE, so L_rec = -A_r = CE. The reader is frozen
    during editing; gradients flow only into the attached low-rank factors.
    """
    return reader_loss + float(lambda_ret) * retain_nll
