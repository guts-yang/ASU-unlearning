"""CF2-07: third-order subject-relation-answer cumulant difference."""
from __future__ import annotations

import torch

METHOD_ID = "CF2-07"
NEEDS_STUDENT_MINUS = True
REQUIRES_CONTROL = False


def _projection(width, q, seed):
    return torch.linalg.qr(torch.randn(width, q, generator=torch.Generator().manual_seed(seed)), mode="reduced").Q


def prepare(cache, teacher, tok, cfg, device):
    del teacher, tok
    q = int(cfg["method"].get("projection_dim", 4))
    projections = {}
    for layer in cfg["layers"]:
        width = cache["pairs"][0]["subject_plus"][layer].numel()
        projections[layer] = {name: _projection(width, q, int(cfg["seed"]) + layer * 37 + i).to(device) for i, name in enumerate(("subject", "relation", "answer"))}
    return {"enabled": True, "projections": projections}


def _student(row, layer, side, p):
    out = row[side]
    item = row["item"]
    # Student hidden states are bf16 while the fixed projections and teacher cache are fp32.
    hidden = out["hidden"][layer].float()
    u = hidden[item["subject_indices"]].mean(0) @ p["subject"]
    v = hidden[out["enc"]["answer_start"]:out["enc"]["answer_end"]].mean(0) @ p["answer"]
    relation = [i for i in range(out["enc"]["prompt_length"]) if i not in set(item["subject_indices"])]
    r = hidden[relation].mean(0) @ p["relation"]
    return u, r, v


def _teacher(item, layer, side, p):
    suffix = "plus" if side == "plus" else "minus"
    u = item[f"subject_{suffix}"][layer].to(p["subject"].device) @ p["subject"]
    r = item[f"relation_{suffix}"][layer].to(p["relation"].device) @ p["relation"]
    v = item[f"h_{suffix}"][layer].to(p["answer"].device).mean(0) @ p["answer"]
    return u, r, v


def _center(x):
    return x - x.mean(0, keepdim=True)


def third_cumulant(u, r, v):
    return torch.einsum("ni,nj,nk->ijk", _center(u), _center(r), _center(v)) / u.shape[0]


def second_cross(u, v):
    return _center(u).T @ _center(v) / max(1, u.shape[0] - 1)


def loss(batch, prepared, cfg):
    thirds, guards = [], []
    for layer in cfg["layers"]:
        p = prepared["projections"][layer]
        for side in ("plus", "minus"):
            us, rs, vs = [torch.stack(x) for x in zip(*[_student(row, layer, side, p) for row in batch])]
            ut, rt, vt = [torch.stack(x) for x in zip(*[_teacher(row["item"], layer, side, p) for row in batch])]
            third = third_cumulant(us, rs, vs)
            guards.append((us.mean(0) - ut.mean(0)).pow(2).mean() + (rs.mean(0) - rt.mean(0)).pow(2).mean() + (vs.mean(0) - vt.mean(0)).pow(2).mean())
            guards.append((second_cross(us, vs) - second_cross(ut, vt)).pow(2).mean())
            if side == "plus":
                kp = third
            else:
                km = third
        thirds.append((kp - km).pow(2).mean())
    third_loss = torch.stack(thirds).mean()
    guard = torch.stack(guards).mean()
    return {"method_loss": third_loss + float(cfg["method"].get("low_order_weight", 0.1)) * guard, "third_cumulant_delta": third_loss.detach(), "low_order_guard": guard.detach()}


def diagnostics(cache, prepared, cfg):
    return {"operator": "third_centered_moment_plus_minus", "projection_dim": int(cfg["method"].get("projection_dim", 4)), "pairs": len(cache["pairs"]), "distinct_from_factorial_mean_interaction": True}


def self_test():
    u = torch.randn(8, 2)
    r = torch.randn(8, 2)
    v = torch.randn(8, 2)
    first = third_cumulant(u, r, v)
    flipped = third_cumulant(-u, r, v)
    assert torch.allclose(first, -flipped)
    return {"method_id": METHOD_ID, "status": "PASS", "odd_order_sign_flip": True, "tensor_shape": list(first.shape)}
