"""CF2-03: counterfactual conditional subject-to-answer flow."""
from __future__ import annotations

import torch

METHOD_ID = "CF2-03"
NEEDS_STUDENT_MINUS = True
REQUIRES_CONTROL = False


def _projection(width, q, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.linalg.qr(torch.randn(width, q, generator=generator), mode="reduced").Q


def prepare(cache, teacher, tok, cfg, device):
    del teacher, tok
    q = int(cfg["method"].get("projection_dim", 4))
    projections = {}
    for layer in cfg["layers"]:
        width = cache["pairs"][0]["subject_plus"][layer].numel()
        projections[layer] = {name: _projection(width, q, int(cfg["seed"]) + 101 * layer + i).to(device) for i, name in enumerate(("subject", "answer", "context"))}
    return {"enabled": True, "projections": projections, "minimum_batch": 2}


def _student_vectors(row, layer, side, projections):
    out = row[side]
    item = row["item"]
    # Student hidden states are bf16 while the fixed projections and teacher cache are fp32.
    hidden = out["hidden"][layer].float()
    subject = hidden[item["subject_indices"]].mean(0) @ projections["subject"]
    answer = hidden[out["enc"]["answer_start"]:out["enc"]["answer_end"]].mean(0) @ projections["answer"]
    prompt = list(range(out["enc"]["prompt_length"]))
    relation = [i for i in prompt if i not in set(item["subject_indices"])]
    context = hidden[relation].mean(0) @ projections["context"]
    return subject, answer, context


def _teacher_vectors(item, layer, side, projections):
    suffix = "plus" if side == "plus" else "minus"
    u = item[f"subject_{suffix}"][layer].to(projections["subject"].device) @ projections["subject"]
    v = item[f"h_{suffix}"][layer].to(projections["answer"].device).mean(0) @ projections["answer"]
    q = item[f"relation_{suffix}"][layer].to(projections["context"].device) @ projections["context"]
    return u, v, q


def _center(x):
    return x - x.mean(0, keepdim=True)


def _cov(x, y):
    return _center(x).T @ _center(y) / max(1, x.shape[0] - 1)


def conditional_crosscov(u, v, q, ridge=1e-3):
    cuv = _cov(u, v)
    cuq = _cov(u, q)
    cqv = _cov(q, v)
    cqq = _cov(q, q)
    eye = torch.eye(cqq.shape[0], dtype=cqq.dtype, device=cqq.device)
    return cuv - cuq @ torch.linalg.solve(cqq + ridge * eye, cqv)


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
    flows, margins = [], []
    ridge = float(cfg["method"].get("ridge", 0.001))
    for layer in cfg["layers"]:
        projections = prepared["projections"][layer]
        for side in ("plus", "minus"):
            student = [_student_vectors(row, layer, side, projections) for row in batch]
            teacher = [_teacher_vectors(row["item"], layer, side, projections) for row in batch]
            us, vs, qs = [torch.stack(values) for values in zip(*student)]
            ut, vt, _ = [torch.stack(values) for values in zip(*teacher)]
            margins.append((us.mean(0) - ut.mean(0)).pow(2).mean() + (vs.mean(0) - vt.mean(0)).pow(2).mean())
            margins.append((_cov(us, us) - _cov(ut, ut)).pow(2).mean() + (_cov(vs, vs) - _cov(vt, vt)).pow(2).mean())
            if side == "plus":
                bp = conditional_crosscov(us, vs, qs, ridge)
            else:
                bm = conditional_crosscov(us, vs, qs, ridge)
        flows.append((bp - bm).pow(2).mean())
    flow = torch.stack(flows).mean()
    margin = torch.stack(margins).mean()
    return {"control_changed": torch.tensor(float(control_changed)), "method_loss": flow + float(cfg["method"].get("marginal_weight", 0.1)) * margin, "conditional_flow": flow.detach(), "marginal_deadzone": margin.detach()}


def diagnostics(cache, prepared, cfg):
    return {"operator": "conditional_crosscov_plus_minus", "projection_dim": int(cfg["method"].get("projection_dim", 4)), "pair_count": len(cache["pairs"]), "causal_claim": False, "requires_batch_at_least": prepared["minimum_batch"]}


def self_test():
    u = torch.tensor([[1.0], [-1.0], [1.0], [-1.0]])
    v = torch.tensor([[1.0], [1.0], [-1.0], [-1.0]])
    q = torch.tensor([[1.0], [-1.0], [-1.0], [1.0]])
    value = conditional_crosscov(u, v, q)
    assert torch.allclose(value, torch.zeros_like(value), atol=1e-6)
    return {"method_id": METHOD_ID, "status": "PASS", "independent_conditional_crosscov_norm": float(value.norm())}
