"""CF3-05 AGSE: answer-span Grassmann projector difference with null normalization."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from modeling import span_hidden

METHOD_ID = "CF3-05"
NEEDS_STUDENT_MINUS = False
REQUIRES_CONTROL = False


def subspace(matrix, rank):
    rank = min(rank, min(matrix.shape))
    if rank < 1:
        raise ValueError("answer span and hidden width must permit rank >= 1")
    _, _, vh = torch.linalg.svd(matrix.double(), full_matrices=False)
    return vh[:rank].T.float()


def projector_difference(plus, minus, mean, rank):
    up = subspace(plus - mean[:len(plus)], rank)
    um = subspace(minus - mean[:len(minus)], rank)
    return up @ up.T - um @ um.T


def _retain_position_mean(cache, layer, width):
    max_len = max(len(x["answer"][layer]) for x in cache["retain"])
    rows = []
    for position in range(max_len):
        values = [x["answer"][layer][position] for x in cache["retain"] if len(x["answer"][layer]) > position]
        rows.append(torch.stack(values).mean(0) if values else torch.zeros(width))
    return torch.stack(rows)


def prepare(cache, teacher, tok, cfg, device, student=None, out_dir=None):
    del teacher, tok, student, out_dir
    rank = int(cfg["method"].get("rank", 4))
    selected_rank = int(cfg["method"].get("selected_rank", 2))
    threshold = float(cfg["method"].get("null_threshold", 0.0))
    targets, bases, reports = {}, {}, {}
    items = cache["pairs"]
    for layer in cfg["layers"]:
        width = items[0]["h_plus"][layer].shape[-1]
        mean = _retain_position_mean(cache, layer, width)
        true = torch.zeros(width, width)
        null_mean = torch.zeros_like(true)
        first = torch.zeros_like(true)
        second = torch.zeros_like(true)
        permutation = torch.randperm(len(items), generator=torch.Generator().manual_seed(int(cfg["seed"]) + layer))
        for i, item in enumerate(items):
            difference = projector_difference(item["h_plus"][layer], item["h_minus"][layer], mean, rank)
            true += difference / len(items)
            if i % 2 == 0:
                first += difference / len(items[0::2])
            else:
                second += difference / len(items[1::2])
            null = items[int(permutation[i])]
            null_mean += projector_difference(item["h_plus"][layer], null["h_minus"][layer], mean, rank) / len(items)
        split_cosine = F.cosine_similarity(first.flatten(), second.flatten(), dim=0)
        evals, vec = torch.linalg.eigh((true - null_mean).double())
        order = torch.argsort(evals.abs(), descending=True)
        null_scale = torch.linalg.eigvalsh(null_mean.double()).abs().median() + 1e-6
        rows = []
        for index in order:
            score = float((evals[index].abs() - null_scale) / (null_scale + 1e-6))
            if score > threshold:
                rows.append((index, score))
            if len(rows) >= selected_rank:
                break
        stable = float(split_cosine) >= float(cfg["method"].get("min_split_cosine", 0.0))
        if rows and (stable or not bool(cfg["method"].get("strict_gate", True))):
            raw = vec[:, [x[0] for x in rows]].float()
            basis = torch.linalg.qr(raw, mode="reduced").Q
            bases[layer] = basis
            for item in items:
                delta = item["h_plus"][layer] - item["h_minus"][layer]
                filtered = delta @ basis @ basis.T
                targets[(item["index"], layer)] = item["h_plus"][layer] - float(cfg["method"].get("alpha", 0.1)) * filtered
        else:
            bases[layer] = None
        principal = torch.linalg.svdvals(subspace(items[0]["h_plus"][layer] - mean[:len(items[0]["h_plus"][layer])], rank).T @
                                         subspace(items[0]["h_minus"][layer] - mean[:len(items[0]["h_minus"][layer])], rank))
        reports[str(layer)] = {"selected_directions": int(bases[layer].shape[1]) if bases[layer] is not None else 0,
                               "split_projector_cosine": float(split_cosine), "split_gate": stable,
                               "null_eigenvalue_scale": float(null_scale),
                               "first_pair_principal_cosines": principal.tolist()}
    enabled = all(bases[layer] is not None for layer in cfg["layers"])
    if not enabled and not bool(cfg["method"].get("strict_gate", True)):
        for layer in cfg["layers"]:
            if bases[layer] is None:
                basis = subspace(items[0]["h_plus"][layer] - items[0]["h_minus"][layer], selected_rank)
                bases[layer] = basis
                for item in items:
                    delta = item["h_plus"][layer] - item["h_minus"][layer]
                    targets[(item["index"], layer)] = item["h_plus"][layer] - float(cfg["method"].get("alpha", 0.1)) * (delta @ basis @ basis.T)
        enabled = True
    return {"enabled": enabled, "targets": targets, "bases": bases, "reports": reports, "rank": rank}


def loss(batch, prepared, cfg):
    terms = []
    for row in batch:
        item, plus = row["item"], row["plus"]
        for layer in cfg["layers"]:
            target = prepared["targets"][(item["index"], layer)].to(plus["nll"].device)
            sh = span_hidden(plus, layer)
            n = min(len(sh), len(target))
            terms.append(F.mse_loss(sh[:n], target[:n].detach()))
    return {"method_loss": torch.stack(terms).mean()}


def diagnostics(cache, prepared, cfg):
    del cache, cfg
    return {"operator": "answer_position_grassmann_projector_difference", "layers": prepared["reports"],
            "enabled": prepared["enabled"], "centering": "position-matched independent retain mean",
            "scope": "basis-rotation invariant, not arbitrary hidden-space rotation invariant"}


def self_test():
    generator = torch.Generator().manual_seed(11)
    base = torch.randn(6, 4, generator=generator).double()
    up = subspace(base, 2)
    rotation, _ = torch.linalg.qr(torch.randn(2, 2, generator=generator).double())
    rotated = up @ rotation.float()
    error = (up @ up.T - rotated @ rotated.T).abs().max()
    assert float(error) < 1e-5
    return {"method_id": METHOD_ID, "status": "PASS", "basis_rotation_error": float(error)}
