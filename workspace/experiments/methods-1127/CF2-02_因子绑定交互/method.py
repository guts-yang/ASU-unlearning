"""CF2-02: factorial subject x relation binding interaction."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from modeling import span_hidden

METHOD_ID = "CF2-02"
NEEDS_STUDENT_MINUS = False
REQUIRES_CONTROL = True


def prepare(cache, teacher, tok, cfg, device):
    del teacher, tok
    valid = {}
    ood = {}
    for item in cache["pairs"]:
        shared = float(item.get("control_shared_nll", item["teacher_nll"]))
        natural = float(item.get("control_ood_nll", shared))
        gap = abs(shared - natural)
        valid[item["index"]] = gap <= float(cfg["method"].get("ood_max", 5.0))
        ood[str(item["index"])] = {"shared_answer_nll": shared, "natural_answer_nll": natural, "gap": gap, "valid": valid[item["index"]]}
    return {"enabled": any(valid.values()), "valid": valid, "ood": ood, "device": device}


def loss(batch, prepared, cfg):
    terms = []
    used = 0
    for row in batch:
        item, plus = row["item"], row["plus"]
        if not prepared["valid"].get(item["index"], False):
            continue
        for layer in cfg["layers"]:
            ff = item["h_plus"][layer].to(plus["nll"].device)
            df = item["h_minus"][layer].to(plus["nll"].device)
            fc = item["h_control_plus"][layer].to(plus["nll"].device)
            dc = item["h_control_minus"][layer].to(plus["nll"].device)
            sh = span_hidden(plus, layer)
            n = min(len(sh), len(ff), len(df), len(fc), len(dc))
            if n == 0:
                continue
            interaction = (ff[:n] - df[:n]) - (fc[:n] - dc[:n])
            target = ff[:n] - float(cfg["method"].get("alpha", 0.1)) * interaction
            terms.append(F.mse_loss(sh[:n], target.detach()))
            used += 1
    if not terms:
        raise RuntimeError("no OOD-valid four-cell factorial pairs")
    return {"method_loss": torch.stack(terms).mean(), "factorial_terms": torch.tensor(float(used), device=batch[0]["plus"]["nll"].device)}


def diagnostics(cache, prepared, cfg):
    del cfg
    return {"operator": "double_counterfactual_difference", "ood_controls": prepared["ood"], "valid_pairs": sum(prepared["valid"].values()), "total_pairs": len(cache["pairs"]), "shared_answer_forcing_is_a_stress_test": True}


def self_test():
    alpha = {0: torch.tensor([1.0, 2.0]), 1: torch.tensor([3.0, 4.0])}
    beta = {0: torch.tensor([2.0, -1.0]), 1: torch.tensor([-2.0, 3.0])}
    interaction = torch.tensor([0.7, -0.2])
    a = {(s, r): alpha[s] + beta[r] + (interaction if (s, r) == (0, 0) else 0.0) for s in (0, 1) for r in (0, 1)}
    delta2 = (a[0, 0] - a[1, 0]) - (a[0, 1] - a[1, 1])
    assert torch.allclose(delta2, interaction)
    return {"method_id": METHOD_ID, "status": "PASS", "main_effects_cancelled": True, "interaction": delta2.tolist()}
