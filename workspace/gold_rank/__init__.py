"""Frozen teacher-side gold-rank edits for the ASU forget KL."""

from .demotion import MODES, apply_gold_rank_logits, validate_spec

__all__ = ["MODES", "apply_gold_rank_logits", "validate_spec"]
