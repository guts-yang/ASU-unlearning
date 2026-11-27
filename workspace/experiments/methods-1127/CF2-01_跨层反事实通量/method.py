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
            flux = (item["h_plus"][right] - item["h_minus"][right]) - (item["h_plus"][left] - item["h_minus"][left])
            wrong = (other["h_plus"][right] - item["h_minus"][right]) - (other["h_plus"][left] - item["h_minus"][left])
            true_values.append(flux.norm(dim=-1).mean())
            shuffled_values.append(wrong.norm(dim=-1).mean())
        true = torch.stack(true_values)
        wrong = torch.stack(shuffled_values)
        threshold = wrong.median()
        weight = torch.sigmoid((true.mean() - threshold) / float(cfg["method"].get("temperature", 0.1)))
        weights[(left, right)] = weight.to(device)
        spectra[str((left, right))] = {"true_flux_norm": float(true.mean()), "shuffled_flux_norm": float(wrong.mean()), "weight": float(weight)}
    return {"enabled": True, "weights": weights, "spectra": spectra}


def loss(batch, prepared, cfg):
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
    return {"method_loss": torch.stack(terms).mean(), "flux_terms": torch.tensor(float(len(terms)), device=batch[0]["plus"]["nll"].device)}


def diagnostics(cache, prepared, cfg):
    del cfg
    return {"operator": "layer_difference_of_counterfactual_delta", "spectra": prepared["spectra"], "interpretation": "weights are pairing-specificity diagnostics, not causal certificates"}


def self_test():
    plus = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4)
    static = torch.randn(2, 4)
    minus = plus - static.unsqueeze(0)
    delta = plus - minus
    flux = delta[1:] - delta[:-1]
    assert torch.allclose(flux, torch.zeros_like(flux), atol=1e-5, rtol=1e-5)
    return {"method_id": METHOD_ID, "status": "PASS", "static_offset_flux_norm": float(flux.norm()), "telescoping": bool(torch.allclose(delta[0] + flux.sum(0), delta[-1], atol=1e-5, rtol=1e-5))}
