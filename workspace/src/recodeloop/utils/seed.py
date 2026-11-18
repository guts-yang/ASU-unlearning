"""Seed control with an explicit record of every actual seed used."""

from __future__ import annotations

import random

import numpy as np
import torch

from ..types import SeedRecord


def set_global_seed(seed: int) -> SeedRecord:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cuda_seeds = tuple(torch.cuda.initial_seed() for _ in range(torch.cuda.device_count())) \
        if torch.cuda.is_available() else ()
    return SeedRecord(
        seed=seed,
        python_random=random.getstate()[1][0],
        numpy_random=int(np.random.get_state()[1][0]),
        torch_cpu=int(torch.initial_seed()),
        torch_cuda=cuda_seeds,
    )
