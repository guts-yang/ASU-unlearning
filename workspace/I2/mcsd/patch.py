"""Clean donor cache and activation-patch forwards.

Donors come from prompt-only forwards. Scoring forwards may append answer
tokens; patches are limited to prompt positions stored in the cache.
"""

from __future__ import annotations

import torch

from I2.mcsd.hooks import (
    ActivationEditor,
    bind_editor,
    parse_path,
    restore_editor,
)


def freeze_model(model):
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)


def module_logits(model, batch):
    if batch.get("attention_temp") not in (None,):
        raise RuntimeError("localization forwards do not apply attention temperature")
    output = model(
        input_ids=batch["input_ids"],
        attention_mask=batch.get("attention_mask"),
        labels=batch.get("labels"),
        attention_temp=None,
        use_cache=False,
    )
    return output.logits


def capture_clean(model, batch, forward_fn=module_logits):
    freeze_model(model)
    editor = ActivationEditor(mode="record", sample_ids=list(batch["sample_ids"]))
    saved = bind_editor(model, editor)
    try:
        with torch.no_grad():
            forward_fn(model, batch)
    finally:
        restore_editor(saved)
    return editor.cache


def patched_logits(
    model,
    batch,
    paths,
    source,
    donor,
    strength=1.0,
    weights=None,
    forward_fn=module_logits,
    audit=None,
):
    parsed = [parse_path(path) for path in paths]
    if audit is not None:
        audit.append(
            {
                "paths": parsed,
                "strength": None if weights is not None else strength,
                "direction_samples": list(batch["sample_ids"]),
            }
        )
    editor = ActivationEditor(
        mode="patch",
        sample_ids=list(batch["sample_ids"]),
        paths=parsed,
        source=source,
        donor=donor,
        strength=strength,
        weights=weights,
    )
    saved = bind_editor(model, editor)
    try:
        logits = forward_fn(model, batch)
    finally:
        restore_editor(saved)
    return logits, editor
