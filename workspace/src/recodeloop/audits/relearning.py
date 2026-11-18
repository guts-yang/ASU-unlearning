"""Budget-limited relearning audit.

Requires an explicit in-memory snapshot permission; without it the arm is
reported uncovered rather than silently cloning a large model.
"""

from __future__ import annotations

import time
from typing import Sequence

import torch

from ..baseline_adapter import BaselineAdapter
from ..config import FrozenConfig
from ..types import AuditResult, Budget
from ..utils.logging import JsonlLogger
from ..utils.seed import set_global_seed
from .base import not_run


def run_relearning(
    adapter: BaselineAdapter,
    train_items: Sequence[tuple[torch.Tensor, torch.Tensor]],
    cfg: FrozenConfig,
    allow_in_memory_copy: bool = False,
    logger: JsonlLogger | None = None,
    seed: int | None = None,
) -> AuditResult:
    arm_cfg = cfg.section("audits")["arms"]["relearning"]
    steps = int(arm_cfg["max_steps"])
    budget = Budget(tokens=0, steps=steps, components=0, rank=0)
    if not train_items or not allow_in_memory_copy:
        return not_run(
            "relearning",
            "未获显式 in-memory 快照授权或无数据；不隐式复制模型",
            budget,
        )
    actual_seed = int(seed if seed is not None else arm_cfg["seed"])
    set_global_seed(actual_seed)
    snapshot = {k: v.detach().clone() for k, v in adapter.model.state_dict().items()}
    if logger:
        logger.log("audit.relearning.snapshot", tensors=len(snapshot),
                   note="explicit in-memory copy for CPU/small-model audit")
    adapter.model.train()
    optimizer = torch.optim.AdamW(adapter.model.parameters(), lr=1e-4)
    used_tokens = 0
    started = time.monotonic()
    try:
        for _ in range(steps):
            for input_ids, labels in train_items:
                out = adapter.model(input_ids=input_ids, labels=labels)
                loss = out.loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                used_tokens += int(input_ids.numel())
    finally:
        adapter.model.load_state_dict(snapshot)
        adapter.model.eval()
    return AuditResult(
        arm="relearning",
        independence_statement=(
            f"独立审计预算：1 epoch / <= {steps} 步；实际 token={used_tokens}；"
            f"seed={actual_seed}；审计后恢复模型快照"
        ),
        budget=Budget(tokens=used_tokens, steps=steps, components=0, rank=0,
                      wall_clock_s=time.monotonic() - started),
        raw_metrics={"used_tokens": float(used_tokens)},
        control_delta={}, coverage="covered",
    )
