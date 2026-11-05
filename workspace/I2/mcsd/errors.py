"""Errors for the candidate localization path. They are not E0 verdicts."""

from __future__ import annotations


class HookBoundaryError(RuntimeError):
    """The live module does not expose the specified activation boundary."""


class ConfigError(ValueError):
    """A required choice is missing or reuses a frozen E0 setting."""


class StudentE1Blocked(RuntimeError):
    """Student training is refused."""


class HoldoutTuningError(RuntimeError):
    """Holdout rows were used to choose paths, thresholds, or rho."""


class DifferentiableMaskUnavailable(RuntimeError):
    """Backpropagation through the activation mask is not supported."""
