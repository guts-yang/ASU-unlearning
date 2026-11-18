"""Audit arm protocol and shared helpers."""

from __future__ import annotations

from typing import Protocol

from ..types import AuditResult, Budget, uncovered_audit


class AuditArm(Protocol):
    name: str

    def run(self) -> AuditResult: ...


def not_run(arm: str, reason: str, budget: Budget | None = None) -> AuditResult:
    return uncovered_audit(arm, reason, budget)
