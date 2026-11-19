"""Registration hooks for the local OpenUnlearning runtime."""

from __future__ import annotations


def register_workspace_trainers():
    from trainer import TRAINER_REGISTRY
    from open_unlearning_plugins.asu.trainer import ASU
    from open_unlearning_plugins.riu.trainer import RIU, RIU_GA, RIU_NPO, RIU_ASU

    TRAINER_REGISTRY["ASU"] = ASU
    TRAINER_REGISTRY["RIU"] = RIU
    TRAINER_REGISTRY["RIU_GA"] = RIU_GA
    TRAINER_REGISTRY["RIU_NPO"] = RIU_NPO
    TRAINER_REGISTRY["RIU_ASU"] = RIU_ASU
    return TRAINER_REGISTRY


def register_workspace_evaluators():
    from evals import EVALUATOR_REGISTRY

    return EVALUATOR_REGISTRY
