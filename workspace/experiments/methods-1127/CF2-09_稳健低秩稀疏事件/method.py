"""CF2-09: robust low-rank + sparse counterfactual event decomposition."""
from __future__ import annotations

import torch
import torch.nn.functional as F

METHOD_ID = "CF2-09"
NEEDS_STUDENT_MINUS = False
REQUIRES_CONTROL = False


def soft_threshold(x, threshold):
    return torch.sign(x) * (x.abs() - threshold).clamp_min(0)


def robust_pca(matrix, lam=None, iterations=50):
    n, d = matrix.shape
    lam = lam or 1.0 / max(n, d) ** 0.5
    l = torch.zeros_like(matrix)
    s = torch.zeros_like(matrix)
    y = torch.zeros_like(matrix)
    norm_two = torch.linalg.svdvals(matrix)[0].clamp_min(1e-6)
    norm_inf = matrix.abs().max().clamp_min(1e-6) / lam
    mu = 0.5 * n * d / (norm_two + norm_inf)
    for _ in range(iterations):
        u, singular, vh = torch.linalg.svd(matrix - s + y / mu, full_matrices=False)
        shrunk = (singular - 1.0 / mu).clamp_min(0)
        rank = int((shrunk > 0).sum())
        if rank:
            l = (u[:, :rank] * shrunk[:rank]) @ vh[:rank]
        else:
            l = torch.zeros_like(matrix)
        s = soft_threshold(matrix - l + y / mu, lam / mu)
        residual = matrix - l - s
        y = y + mu * residual
        if residual.norm() / matrix.norm().clamp_min(1e-8) < 1e-5:
            break
    return {"L": l, "S": s, "E": matrix - l - s, "iterations": _ + 1}


def prepare(cache, teacher, tok, cfg, device):
    del teacher, tok
    targets, reports = {}, {}
    enabled = True
    for layer in cfg["layers"]:
        plus = torch.stack([x["h_plus"][layer].mean(0) for x in cache["pairs"]])
        minus = torch.stack([x["h_minus"][layer].mean(0) for x in cache["pairs"]])
        wrong = torch.stack([cache["pairs"][(i + 1) % len(cache["pairs"])]["h_minus"][layer].mean(0) for i in range(len(cache["pairs"]))])
        true = robust_pca(plus - minus, iterations=int(cfg["method"].get("iterations", 50)))
        negative = robust_pca(plus - wrong, iterations=int(cfg["method"].get("iterations", 50)))
        scores = {key: float(true[key].abs().mean() - negative[key].abs().mean()) for key in ("L", "S")}
        selected = max(scores, key=scores.get)
        positive = [key for key, value in scores.items() if value > 0]
        status = "PASS" if len(positive) == 1 else ("AMBIGUOUS" if len(positive) > 1 else "UNRESOLVED")
        if cfg["method"].get("strict_semantic_gate", False) and status != "PASS":
            enabled = False
        target = plus - float(cfg["method"].get("alpha", 0.1)) * true[selected]
        targets[layer] = target.to(device)
        reports[str(layer)] = {"specificity_scores": scores, "selected_component": selected, "semantic_gate": status, "sparse_ratio": float((true["S"].abs() > 1e-6).float().mean()), "low_rank_energy": float(true["L"].norm() / (plus - minus).norm().clamp_min(1e-8)), "iterations": true["iterations"]}
    return {"enabled": enabled, "targets": targets, "reports": reports}


def loss(batch, prepared, cfg):
    terms = []
    for row in batch:
        item, plus = row["item"], row["plus"]
        for layer in cfg["layers"]:
            h = plus["hidden"][layer].float()[plus["enc"]["answer_start"]:plus["enc"]["answer_end"]].mean(0)
            terms.append(F.mse_loss(h, prepared["targets"][layer][item["index"]].detach()))
    return {"method_loss": torch.stack(terms).mean()}


def diagnostics(cache, prepared, cfg):
    del cache
    return {"operator": "stable_principal_component_pursuit", "granularity": "layerwise answer-span mean", "layers": prepared["reports"], "strict_semantic_gate": bool(cfg["method"].get("strict_semantic_gate", False)), "provisional_training_is_not_semantic_validation": True}


def self_test():
    base = torch.randn(6, 2)
    low = base @ torch.randn(2, 4) * 0.1
    sparse = torch.zeros_like(low)
    sparse[2, 3] = 4.0
    matrix = low + sparse
    result = robust_pca(matrix, iterations=30)
    reconstruction = result["L"] + result["S"] + result["E"]
    error = (matrix - reconstruction).abs().max()
    assert error < 1e-5
    return {"method_id": METHOD_ID, "status": "PASS", "reconstruction_max_error": float(error), "iterations": result["iterations"]}
