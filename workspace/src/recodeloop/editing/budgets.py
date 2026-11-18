"""Budget accounting for matched controls and editor steps."""

from __future__ import annotations

import time
from dataclasses import dataclass

from ..types import Budget


@dataclass
class BudgetTracker:
    budget: Budget
    start: float = 0.0

    def __post_init__(self) -> None:
        if self.start == 0.0:
            self.start = time.monotonic()

    def finish(self, tokens: int, steps: int) -> Budget:
        return Budget(
            tokens=tokens,
            steps=steps,
            components=self.budget.components,
            rank=self.budget.rank,
            wall_clock_s=time.monotonic() - self.start,
        )

    def assert_within(self, actual: Budget) -> None:
        if actual.tokens > self.budget.tokens:
            raise ValueError(f"token budget exceeded: {actual.tokens} > {self.budget.tokens}")
        if actual.steps > self.budget.steps:
            raise ValueError(f"step budget exceeded: {actual.steps} > {self.budget.steps}")
