"""CF2-04: unbalanced optimal transport with source-mass destruction."""
from __future__ import annotations

import torch
import torch.nn.functional as F

METHOD_ID = "CF2-04"
NEEDS_STUDENT_MINUS = False
REQUIRES_CONTROL = False


def transport_plan(plus, minus, epsilon=0.1, tau=1.0, iterations=50):
    cost = torch.cdist(plus.float(), minus.float()).pow(2)
    scale = cost.median().clamp_min(1e-6)
    cost = cost / scale
    kernel = torch.exp(-cost / max(epsilon, 1e-5)).clamp_min(1e-30)
    a = torch.full((plus.shape[0],), 1.0 / plus.shape[0], device=plus.device)
    b = torch.full((minus.shape[0],), 1.0 / minus.shape[0], device=minus.device)
    u = torch.ones_like(a)
    v = torch.ones_like(b)
    rho = tau / (tau + epsilon)
    for _ in range(iterations):
        u = (a / (kernel @ v).clamp_min(1e-30)).pow(rho)
        v = (b / (kernel.T @ u).clamp_min(1e-30)).pow(rho)
    plan = u[:, None] * kernel * v[None, :]
    matched = plan.sum(1)
    destroyed = (a - matched).clamp_min(0)
    bary = (plan @ minus) / matched.clamp_min(1e-8)[:, None]
    global_bary = minus.mean(0, keepdim=True)
    bary = torch.where((matched > 1e-8)[:, None], bary, global_bary)
    return {"plan": plan, "matched": matched, "destroyed": destroyed, "barycenter": bary, "source": a}


def prepare(cache, teacher, tok, cfg, device):
    del teacher, tok
    targets, reports = {}, {}
    for layer in cfg["layers"]:
        plus = torch.stack([x["h_plus"][layer].mean(0) for x in cache["pairs"]])
        minus = torch.stack([x["h_minus"][layer].mean(0) for x in cache["pairs"]])
        result = transport_plan(plus, minus, float(cfg["method"].get("epsilon", 0.1)), float(cfg["method"].get("tau", 1.0)), int(cfg["method"].get("iterations", 50)))
        wrong = transport_plan(plus, minus[torch.roll(torch.arange(len(minus)), 1)], float(cfg["method"].get("epsilon", 0.1)), float(cfg["method"].get("tau", 1.0)), int(cfg["method"].get("iterations", 50)))
        ratio = result["destroyed"] / result["source"].clamp_min(1e-8)
        target = plus - float(cfg["method"].get("alpha", 0.1)) * ratio[:, None] * (plus - result["barycenter"])
        targets[layer] = target.to(device)
        reports[str(layer)] = {"destroyed_mass": float(result["destroyed"].sum()), "matched_mass": float(result["matched"].sum()), "wrong_destroyed_mass": float(wrong["destroyed"].sum()), "mean_destroy_ratio": float(ratio.mean())}
    return {"enabled": True, "targets": targets, "reports": reports}


def loss(batch, prepared, cfg):
    terms = []
    for row in batch:
        item, plus = row["item"], row["plus"]
        for layer in cfg["layers"]:
            h = plus["hidden"][layer][plus["enc"]["answer_start"]:plus["enc"]["answer_end"]].mean(0)
            target = prepared["targets"][layer][item["index"]]
            terms.append(F.mse_loss(h, target.detach()))
    return {"method_loss": torch.stack(terms).mean()}


def diagnostics(cache, prepared, cfg):
    del cfg
    return {"operator": "unbalanced_sinkhorn_source_mass_destruction", "granularity": "layerwise answer-span mean", "layers": prepared["reports"], "donor_barycenter_is_not_training_target": True}


def self_test():
    plus = torch.tensor([[0.0, 0.0], [1.0, 0.0], [4.0, 4.0]])
    minus = torch.tensor([[0.0, 0.1], [1.0, 0.1], [-3.0, -3.0]])
    result = transport_plan(plus, minus, iterations=20)
    assert bool((result["plan"] >= 0).all())
    assert bool((result["destroyed"] >= 0).all())
    assert result["destroyed"].sum() <= result["source"].sum() + 1e-6
    return {"method_id": METHOD_ID, "status": "PASS", "destroyed_mass": float(result["destroyed"].sum()), "nonnegative_plan": True}
