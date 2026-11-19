"""Registration hooks for the local OpenUnlearning runtime."""

from __future__ import annotations


def register_workspace_trainers():
    from trainer import TRAINER_REGISTRY
    from open_unlearning_plugins.asu.trainer import ASU

    TRAINER_REGISTRY["ASU"] = ASU
    return TRAINER_REGISTRY


def register_workspace_evaluators():
    from evals import EVALUATOR_REGISTRY

    return EVALUATOR_REGISTRY
