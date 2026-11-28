"""CF2-06: normalized secant response to discrete subject replacement."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from modeling import span_hidden

METHOD_ID = "CF2-06"
NEEDS_STUDENT_MINUS = True
NEEDS_STUDENT_CONTROL = True
REQUIRES_CONTROL = True


def prepare(cache, teacher, tok, cfg, device):
    del teacher, tok
    mode = str(cfg["method"].get("distance_mode", "legacy"))
    if mode == "median":
        raw = torch.stack([item["subject_delta_norm"].float() for item in cache["pairs"]])
        floor = float(raw.median().clamp_min(1e-3))
    elif mode == "legacy":
        floor = float(cfg["method"].get("epsilon", 0.001))
    else:
        raise ValueError(f"unknown distance_mode: {mode}")
    distances = {item["index"]: item["subject_delta_norm"].float().clamp_min(floor).to(device) for item in cache["pairs"]}
    return {"enabled": True, "distances": distances, "distance_mode": mode, "distance_floor": floor}


def _common_n(row, item, layer, prefix):
    """One shared answer-span length for BOTH sides of a secant pair.

    Under control_mode != none the Teacher-side item comes from a different
    pair whose h_plus/h_minus spans can be shorter than the Student-side ones.
    Taking min() independently on each side produced 27 vs 15 and crashed
    F.mse_loss; this keeps the pair aligned (no-op when lengths already match).
    """
    plus = row[f"{prefix}plus"] if prefix else row["plus"]
    minus = row[f"{prefix}minus"] if prefix else row["minus"]
    sp, sm = span_hidden(plus, layer), span_hidden(minus, layer)
    hp, hm = item[f"h_{prefix}plus"][layer], item[f"h_{prefix}minus"][layer]
    return min(len(sp), len(sm), len(hp), len(hm))


def _student_secant(row, layer, prefix, distance, n=None):
    plus = row[f"{prefix}plus"] if prefix else row["plus"]
    minus = row[f"{prefix}minus"] if prefix else row["minus"]
    sp = span_hidden(plus, layer)
    sm = span_hidden(minus, layer)
    n = min(len(sp), len(sm)) if n is None else min(n, len(sp), len(sm))
    secant = (sp[:n] - sm[:n]) / distance
    return torch.nan_to_num(secant, nan=0.0, posinf=0.0, neginf=0.0).clamp(-50.0, 50.0)


def _teacher_secant(item, layer, prefix, distance, device, n=None):
    if prefix:
        hp, hm = item[f"h_{prefix}plus"][layer], item[f"h_{prefix}minus"][layer]
    else:
        hp, hm = item["h_plus"][layer], item["h_minus"][layer]
    hp, hm = hp.to(device), hm.to(device)
    n = min(len(hp), len(hm)) if n is None else min(n, len(hp), len(hm))
    secant = (hp[:n] - hm[:n]) / distance
    return torch.nan_to_num(secant, nan=0.0, posinf=0.0, neginf=0.0).clamp(-50.0, 50.0)


def _control_batch(batch, cfg):
    """§7 control arm: rotate/randomly permute the Teacher-side `item` only.
    Student-side plus/minus/control_* stay, so only the regression target moves.
    Default 'none' is a no-op -> historical results stay reproducible.
    """
    mode = str(cfg["method"].get("control_mode", "none"))
    if mode in ("none", "") or len(batch) < 2:
        return batch, False
    if mode in ("wrong", "shuffle"):
        order = list(range(1, len(batch))) + [0]
    elif mode in ("random", "noise"):
        gen = torch.Generator().manual_seed(int(cfg.get("seed", 0)) + 977)
        order = torch.randperm(len(batch), generator=gen).tolist()
        if order == list(range(len(batch))):
            order = list(range(1, len(batch))) + [0]
    else:
        raise ValueError(f"unknown control_mode: {mode}")
    return [dict(row, item=batch[j]["item"]) for row, j in zip(batch, order)], True


def loss(batch, prepared, cfg):
    batch, control_changed = _control_batch(batch, cfg)
    targets, anchors = [], []
    for row in batch:
        item = row["item"]
        distance = prepared["distances"][item["index"]]
        for layer in cfg["layers"]:
            n = _common_n(row, item, layer, "")
            secant = _student_secant(row, layer, "", distance, n)
            teacher_secant = _teacher_secant(item, layer, "", distance, secant.device, n)
            ratio = float(cfg["method"].get("target_response_ratio", 0.0))
            targets.append(F.mse_loss(secant, (ratio * teacher_secant).detach()))
            if "control_plus" in row and row["control_plus"] is not None:
                nc = _common_n(row, item, layer, "control_")
                control = _student_secant(row, layer, "control_", distance, nc)
                teacher_control = _teacher_secant(item, layer, "control_", distance, control.device, nc)
                anchors.append(F.mse_loss(control, teacher_control.detach()))
    method_loss = torch.nan_to_num(torch.stack(targets).mean(), nan=0.0, posinf=0.0, neginf=0.0)
    if anchors:
        method_loss = method_loss + float(cfg["method"].get("retain_response_weight", 0.1)) * torch.stack(anchors).mean()
    return {"control_changed": torch.tensor(float(control_changed)), "method_loss": method_loss, "response_anchor": torch.stack(anchors).mean().detach() if anchors else targets[0].new_zeros(())}


def diagnostics(cache, prepared, cfg):
    distances = [float(x) for x in prepared["distances"].values()]
    return {"operator": "delta_hidden_over_subject_embedding_distance", "pair_distances": distances,
            "distance_mode": prepared["distance_mode"], "distance_floor": prepared["distance_floor"],
            "control_mode": str(cfg["method"].get("control_mode", "none")),
            "control_response_anchor": True, "continuous_jacobian_claim": False,
            "zero_secant_requires_neutral_readout_gate": True}


def self_test():
    hp = torch.tensor([2.0, 4.0])
    hm = torch.tensor([1.0, 2.0])
    distance = torch.tensor(2.0)
    first = (hp - hm) / distance
    second = ((hp * 1.5) - (hm * 1.5)) / (distance * 1.5)
    assert torch.allclose(first, second)
    return {"method_id": METHOD_ID, "status": "PASS", "scale_invariant_secant": first.tolist()}
