"""CF2-08: paired scatter normalized by independent retain covariance."""
from __future__ import annotations

import torch
import torch.nn.functional as F

METHOD_ID = "CF2-08"
NEEDS_STUDENT_MINUS = False
REQUIRES_CONTROL = False


def whiten_spectrum(deltas, retains, rank=4, ridge=1e-3):
    cd = deltas.T @ deltas / max(1, len(deltas))
    centered = retains - retains.mean(0, keepdim=True)
    cr = centered.T @ centered / max(1, len(retains) - 1)
    cr = cr + ridge * torch.eye(cr.shape[0], device=cr.device)
    evals_r, ur = torch.linalg.eigh(cr)
    inv_sqrt = ur @ torch.diag((evals_r.clamp_min(ridge)).rsqrt()) @ ur.T
    whitened = inv_sqrt @ cd @ inv_sqrt
    evals, vec = torch.linalg.eigh(whitened)
    order = torch.argsort(evals, descending=True)[:rank]
    basis = inv_sqrt @ vec[:, order]
    return {"basis": basis, "eigenvalues": evals[order], "retain_covariance": cr}


def prepare(cache, teacher, tok, cfg, device):
    del teacher, tok
    q = int(cfg["method"].get("projection_dim", 16))
    targets, reports = {}, {}
    for layer in cfg["layers"]:
        width = cache["pairs"][0]["h_plus"][layer].shape[-1]
        q_layer = min(q, width)
        projection = torch.linalg.qr(torch.randn(width, q_layer, generator=torch.Generator().manual_seed(int(cfg["seed"]) + layer)), mode="reduced").Q
        plus_full = torch.stack([x["h_plus"][layer].mean(0) for x in cache["pairs"]])
        minus_full = torch.stack([x["h_minus"][layer].mean(0) for x in cache["pairs"]])
        plus = plus_full @ projection
        minus = minus_full @ projection
        retains = torch.stack([x["pooled"][layer] for x in cache["retain"]]) @ projection
        spectrum = whiten_spectrum(plus - minus, retains, min(int(cfg["method"].get("rank", 4)), q_layer), float(cfg["method"].get("ridge", 0.001)))
        basis, cr = spectrum["basis"], spectrum["retain_covariance"]
        delta = plus - minus
        projected = basis @ basis.T @ cr @ delta.T
        full_delta = projection @ projected
        target = plus_full - float(cfg["method"].get("alpha", 0.1)) * full_delta.T
        targets[layer] = {"teacher_plus": plus_full.to(device), "target": target.to(device), "projection": projection.to(device)}
        reports[str(layer)] = {"eigenvalues": [float(x) for x in spectrum["eigenvalues"]], "metric_orthogonality_error": float((basis.T @ cr @ basis - torch.eye(basis.shape[1])).abs().max())}
    return {"enabled": True, "targets": targets, "reports": reports}


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
        for layer in cfg["layers"]:
            h = plus["hidden"][layer].float()[plus["enc"]["answer_start"]:plus["enc"]["answer_end"]].mean(0)
            target = prepared["targets"][layer]["target"][item["index"]]
            terms.append(F.mse_loss(h, target.detach()))
    return {"control_changed": torch.tensor(float(control_changed)), "method_loss": torch.stack(terms).mean()}


def diagnostics(cache, prepared, cfg):
    del cache, cfg
    return {"operator": "generalized_paired_scatter_retain_spectrum", "layers": prepared["reports"], "leace_increment_only": True, "projection_is_explicit": True}


def self_test():
    deltas = torch.randn(8, 3)
    retains = torch.randn(20, 3) * 0.2
    result = whiten_spectrum(deltas, retains, rank=2)
    basis, cr = result["basis"], result["retain_covariance"]
    error = (basis.T @ cr @ basis - torch.eye(2)).abs().max()
    assert error < 1e-3
    return {"method_id": METHOD_ID, "status": "PASS", "metric_orthogonality_error": float(error)}
