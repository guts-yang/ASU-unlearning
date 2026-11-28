"""CF2-01: Phi_l = Delta A_{l+1} - Delta A_l."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from modeling import span_hidden

METHOD_ID = "CF2-01"
NEEDS_STUDENT_MINUS = False
REQUIRES_CONTROL = False


def prepare(cache, teacher, tok, cfg, device):
    del teacher, tok
    weights = {}
    spectra = {}
    for left, right in zip(cfg["layers"][:-1], cfg["layers"][1:]):
        true_values, shuffled_values = [], []
        items = cache["pairs"]
        for i, item in enumerate(items):
            other = items[(i + 1) % len(items)]
            tensors = (
                item["h_plus"][right], item["h_minus"][right],
                item["h_plus"][left], item["h_minus"][left],
                other["h_plus"][right], other["h_plus"][left],
            )
            n = min(t.shape[0] for t in tensors)
            if n == 0:
                continue
            hp_r, hm_r, hp_l, hm_l, ohp_r, ohp_l = (t[:n] for t in tensors)
            flux = (hp_r - hm_r) - (hp_l - hm_l)
            wrong = (ohp_r - hm_r) - (ohp_l - hm_l)
            true_values.append(flux.norm(dim=-1).mean())
            shuffled_values.append(wrong.norm(dim=-1).mean())
        true = torch.stack(true_values)
        wrong = torch.stack(shuffled_values)
        threshold = wrong.median()
        gate_mode = str(cfg["method"].get("gate_mode", "legacy"))
        if gate_mode == "relative":
            # Pairing specificity: the TRUE flux is systematically SMALLER than
            # the mismatched one. The legacy sigmoid has this direction inverted.
            weight = torch.sigmoid(torch.log(wrong.mean().clamp_min(1e-8) / true.mean().clamp_min(1e-8)))
        elif gate_mode == "legacy":
            weight = torch.sigmoid((true.mean() - threshold) / float(cfg["method"].get("temperature", 0.1)))
        else:
            raise ValueError(f"unknown gate_mode: {gate_mode}")
        weights[(left, right)] = weight.to(device)
        spectra[str((left, right))] = {"true_flux_norm": float(true.mean()), "shuffled_flux_norm": float(wrong.mean()), "weight": float(weight)}
    return {"enabled": True, "weights": weights, "spectra": spectra}


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
    terms = []
    for row in batch:
        item, plus = row["item"], row["plus"]
        for left, right in zip(cfg["layers"][:-1], cfg["layers"][1:]):
            hp_l = item["h_plus"][left].to(plus["nll"].device)
            hm_l = item["h_minus"][left].to(plus["nll"].device)
            hp_r = item["h_plus"][right].to(plus["nll"].device)
            hm_r = item["h_minus"][right].to(plus["nll"].device)
            sh = span_hidden(plus, right)
            n = min(len(sh), len(hp_r), len(hm_r), len(hp_l), len(hm_l))
            if n == 0:
                continue
            flux = (hp_r[:n] - hm_r[:n]) - (hp_l[:n] - hm_l[:n])
            target = hp_r[:n] - float(cfg["method"].get("alpha", 0.1)) * prepared["weights"][(left, right)] * flux
            terms.append(F.mse_loss(sh[:n], target.detach()))
    if not terms:
        raise RuntimeError("no valid consecutive-layer answer-span activations")
    return {"control_changed": torch.tensor(float(control_changed)), "method_loss": torch.stack(terms).mean(), "flux_terms": torch.tensor(float(len(terms)), device=batch[0]["plus"]["nll"].device)}


def diagnostics(cache, prepared, cfg):
    gate_mode = str(cfg["method"].get("gate_mode", "legacy"))
    control_mode = str(cfg["method"].get("control_mode", "none"))
    return {"operator": "layer_difference_of_counterfactual_delta", "spectra": prepared["spectra"],
            "gate_mode": gate_mode, "control_mode": control_mode,
            "interpretation": "weights are pairing-specificity diagnostics, not causal certificates"}


def self_test():
    plus = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4)
    static = torch.randn(2, 4)
    minus = plus - static.unsqueeze(0)
    delta = plus - minus
    flux = delta[1:] - delta[:-1]
    assert torch.allclose(flux, torch.zeros_like(flux), atol=1e-5, rtol=1e-5)
    return {"method_id": METHOD_ID, "status": "PASS", "static_offset_flux_norm": float(flux.norm()), "telescoping": bool(torch.allclose(delta[0] + flux.sum(0), delta[-1], atol=1e-5, rtol=1e-5))}
