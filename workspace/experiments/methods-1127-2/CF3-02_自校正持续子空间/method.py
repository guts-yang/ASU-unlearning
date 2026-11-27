"""CF3-02 SCPS: controlled downstream replay and persistent difference subspace."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from modeling import patch_forward_output_multi, span_hidden

METHOD_ID = "CF3-02"
NEEDS_STUDENT_MINUS = False
REQUIRES_CONTROL = False


def generalized_persistent_basis(true_deltas, null_deltas, projection, rank, ridge):
    true = true_deltas @ projection
    null = null_deltas @ projection
    ct = true.T @ true / max(1, len(true))
    cn = null.T @ null / max(1, len(null))
    cn = cn + ridge * torch.eye(cn.shape[0], device=cn.device)
    evals_n, un = torch.linalg.eigh(cn)
    inv = un @ torch.diag(evals_n.clamp_min(ridge).rsqrt()) @ un.T
    evals, vec = torch.linalg.eigh(inv @ ct @ inv)
    order = torch.argsort(evals, descending=True)[:min(rank, len(evals))]
    raw = projection @ inv @ vec[:, order]
    basis = torch.linalg.qr(raw, mode="reduced").Q
    return basis, evals[order]


def _valid_null(items, index, layer):
    source = items[index]
    for offset in range(1, len(items)):
        candidate = items[(index + offset) % len(items)]
        if candidate["full_plus"][layer].shape == source["full_plus"][layer].shape:
            return candidate
    return None


def _replay(teacher, item, replacement, layer, target_layer, device):
    if target_layer == layer:
        start = item["enc_plus"]["answer_start"]
        end = item["enc_plus"]["answer_end"]
        return replacement[start:end].float().cpu()
    out = patch_forward_output_multi(teacher, item["enc_plus"], device, {layer - 1: replacement}, layers={target_layer})
    h = out["hidden"][target_layer]
    start, end = item["enc_plus"]["answer_start"], item["enc_plus"]["answer_end"]
    return h[start:end].detach().float().cpu()


def prepare(cache, teacher, tok, cfg, device, student=None, out_dir=None):
    del tok, student, out_dir
    if not cache["pairs"][0]["full_plus"]:
        raise RuntimeError("CF3-02 requires data.cache_full_sequence=true")
    items = cache["pairs"]
    offsets = [int(x) for x in cfg["method"].get("replay_offsets", [0, 1, 2])]
    rank = int(cfg["method"].get("rank", 4))
    ridge = float(cfg["method"].get("ridge", 0.01))
    width = items[0]["h_plus"][cfg["layers"][0]].shape[-1]
    q = min(int(cfg["method"].get("projection_dim", 16)), width)
    generator = torch.Generator().manual_seed(int(cfg["seed"]))
    projection = torch.linalg.qr(torch.randn(width, q, generator=generator), mode="reduced").Q
    targets, reports, bases = {}, {}, {}
    max_layers = teacher.config.num_hidden_layers
    strict = bool(cfg["method"].get("strict_gate", True))
    for layer in cfg["layers"]:
        true_rows, null_rows, survival = [], [], []
        valid = []
        for i, item in enumerate(items):
            null = _valid_null(items, i, layer)
            if null is None:
                continue
            valid.append(i)
            for offset in offsets:
                target_layer = min(max_layers, layer + offset)
                if target_layer < layer:
                    continue
                zp = _replay(teacher, item, item["full_plus"][layer], layer, target_layer, device)
                zm = _replay(teacher, item, item["full_minus"][layer], layer, target_layer, device)
                zn = _replay(teacher, item, null["full_minus"][layer], layer, target_layer, device)
                n = min(len(zp), len(zm), len(zn))
                if n == 0:
                    continue
                true, null_diff = zp[:n] - zm[:n], zp[:n] - zn[:n]
                true_rows.append(true)
                null_rows.append(null_diff)
                base = (item["h_plus"][layer][:n] - item["h_minus"][layer][:n]).norm(dim=-1).clamp_min(1e-8)
                survival.append((true.norm(dim=-1) / base).mean())
        if not true_rows:
            bases[layer] = None
            reports[str(layer)] = {"valid_pairs": 0}
            continue
        true_all, null_all = torch.cat(true_rows), torch.cat(null_rows)
        basis, eigenvalues = generalized_persistent_basis(true_all, null_all, projection, rank, ridge)
        spectral_gate = eigenvalues > float(cfg["method"].get("min_eigenvalue", 1.0))
        if not spectral_gate.any() and not strict:
            spectral_gate = torch.ones_like(spectral_gate)
        basis = basis[:, spectral_gate] if spectral_gate.any() else None
        bases[layer] = basis
        reports[str(layer)] = {"valid_pairs": len(valid), "eigenvalues": eigenvalues.tolist(),
                               "selected_directions": int(spectral_gate.sum()),
                               "mean_survival_ratio": float(torch.stack(survival).mean()),
                               "true_vectors": len(true_all), "null_vectors": len(null_all)}
        if basis is not None:
            for i in valid:
                item = items[i]
                delta = item["h_plus"][layer] - item["h_minus"][layer]
                filtered = delta @ basis @ basis.T
                targets[(i, layer)] = item["h_plus"][layer] - float(cfg["method"].get("alpha", 0.1)) * filtered
    enabled = all(bases[layer] is not None and any((i, layer) in targets for i in range(len(items))) for layer in cfg["layers"])
    if not strict and not enabled:
        raise RuntimeError("CF3-02 tiny fallback still requires valid replay shapes")
    return {"enabled": enabled, "targets": targets, "reports": reports, "bases": bases,
            "replay_offsets": offsets, "projection_dim": q}


def loss(batch, prepared, cfg):
    terms = []
    for row in batch:
        item, plus = row["item"], row["plus"]
        for layer in cfg["layers"]:
            key = (item["index"], layer)
            if key not in prepared["targets"]:
                continue
            target = prepared["targets"][key].to(plus["nll"].device)
            sh = span_hidden(plus, layer)
            n = min(len(sh), len(target))
            terms.append(F.mse_loss(sh[:n], target[:n].detach()))
    if not terms:
        raise RuntimeError("no persistent replay targets")
    return {"method_loss": torch.stack(terms).mean()}


def diagnostics(cache, prepared, cfg):
    del cache, cfg
    return {"operator": "controlled_downstream_replay_generalized_survival_subspace",
            "layers": prepared["reports"], "replay_offsets": prepared["replay_offsets"],
            "enabled": prepared["enabled"], "ood_warning": "patched states require distribution checks"}


def self_test():
    generator = torch.Generator().manual_seed(3)
    persistent = torch.randn(24, 1, generator=generator) @ torch.ones(1, 4)
    transient = torch.randn(24, 4, generator=generator) * 0.01
    true = persistent + transient
    null = torch.randn(24, 4, generator=generator) * 0.01
    projection = torch.eye(4)
    basis, _ = generalized_persistent_basis(true, null, projection, 1, 0.01)
    alignment = basis.abs().mean()
    assert float(alignment) > 0.4
    return {"method_id": METHOD_ID, "status": "PASS", "persistent_alignment": float(alignment)}
