"""Fixed independent activation-probe endpoint for M0b.

The probe is trained once on forget_train_reader activations from the starting
student and then frozen across all arms. It is independent of the editing
reader and never sees confirmation. It is an in-sample probe of the starting
student, not a zero-shot generalization or L2 claim.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from ..activations.extraction import extract_activation_batches
from ..activations.specs import build_spec
from ..baseline_adapter import BaselineAdapter
from ..readers.fact_value import FrozenEmbeddingReader
from ..readers.training import train_edit_reader
from ..types import ActivationSpec

COMPONENT_SUFFIX = "mlp.down_proj"


def _facts_scores_batch(adapter, facts, component, spec):
    encoded_prompts = [f.prompt for f in facts]
    tokenizer = adapter.tokenizer
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer(encoded_prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=128)
    encoded = {k: v.to(adapter.device) for k, v in encoded.items()}
    return extract_activation_batches(
        adapter, [spec], encoded["input_ids"],
        tuple(f.entity_id for f in facts),
        tuple(f.fact_id for f in facts),
        facts[0].split,
        tuple(f.value for f in facts),
        attention_mask=encoded.get("attention_mask"),
    )[component]


def train_fixed_probe(
    adapter: BaselineAdapter,
    real_facts: Sequence,
    nk_facts: Sequence,
    seed: int = 4001,
    steps: int = 96,
    lr: float = 0.02,
) -> tuple[FrozenEmbeddingReader, ActivationSpec, str]:
    """Train a probe on starting-student activations and freeze it."""
    components = [c for c in adapter.component_paths() if c.endswith(COMPONENT_SUFFIX)]
    if not components:
        raise RuntimeError("no mlp.down_proj component for probe")
    component = components[-1]
    spec = build_spec(adapter, component, pooling="last")
    batch = _facts_scores_batch(adapter, list(real_facts) + list(nk_facts), component, spec)
    values = [f.value for f in real_facts] + [f.value for f in nk_facts]
    probe = FrozenEmbeddingReader(values, int(batch.activations.shape[-1]), dim=32, seed=seed)
    probe.to(adapter.device)
    train_edit_reader(probe, [batch], steps, lr, seed)
    return probe, spec, component


def score_fixed_probe(
    adapter: BaselineAdapter,
    probe: FrozenEmbeddingReader,
    spec: ActivationSpec,
    component: str,
    real_facts: Sequence,
    nk_facts: Sequence,
) -> Mapping[str, float]:
    real_batch = _facts_scores_batch(adapter, list(real_facts), component, spec)
    nk_batch = _facts_scores_batch(adapter, list(nk_facts), component, spec)
    nk_values = tuple(f.value for f in nk_facts)
    real_score = probe.score(real_batch, real_batch.fact_ids, real_batch.split, nk_values)
    nk_score = probe.score(nk_batch, nk_batch.fact_ids, nk_batch.split, nk_values)
    real_acc = real_score.fact_value_accuracy
    nk_acc = nk_score.fact_value_accuracy
    return {
        "probe_real": real_acc,
        "probe_nk": nk_acc,
        "probe_advantage": real_acc - nk_acc,
        "probe_confidence": real_score.confidence,
    }
