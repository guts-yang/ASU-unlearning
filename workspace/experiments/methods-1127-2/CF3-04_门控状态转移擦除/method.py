"""CF3-04 GRTE: SwiGLU four-state transition graphs and soft source-state exit."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from modeling import decoder_layers

METHOD_ID = "CF3-04"
NEEDS_TEACHER_GLU = True
NEEDS_STUDENT_GLU = True
NEEDS_STUDENT_MINUS = False
REQUIRES_CONTROL = False


def state_code(g, u):
    return 2 * (g > 0).long() + (u > 0).long()


def transition_matrix(source, target):
    intermediate = source.shape[-1]
    reduce_dims = tuple(range(source.ndim - 1))
    matrix = torch.zeros(intermediate, 4, 4, dtype=torch.float64)
    for a in range(4):
        for b in range(4):
            count = ((source == a) & (target == b)).sum(dim=reduce_dims)
            matrix[:, a, b] = count.double()
    matrix /= matrix.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return matrix.float()


def soft_occupancy(g, u, source, temperature):
    kg = 1.0 if source & 2 else -1.0
    ku = 1.0 if source & 1 else -1.0
    return torch.sigmoid(kg * g / temperature) * torch.sigmoid(ku * u / temperature)


def prepare(cache, teacher, tok, cfg, device, student=None, out_dir=None):
    del tok, student, out_dir
    selected, reports = {}, {}
    max_neurons = int(cfg["method"].get("max_neurons", 64))
    min_specificity = float(cfg["method"].get("min_specificity", 0.0))
    for layer in cfg["layers"]:
        gp = torch.stack([x["glu_plus"][layer]["g"] for x in cache["pairs"]]).float()
        up = torch.stack([x["glu_plus"][layer]["u"] for x in cache["pairs"]]).float()
        gm = torch.stack([x["glu_minus"][layer]["g"] for x in cache["pairs"]]).float()
        um = torch.stack([x["glu_minus"][layer]["u"] for x in cache["pairs"]]).float()
        sp, sm = state_code(gp, up), state_code(gm, um)
        true = transition_matrix(sp, sm)
        permutation = torch.randperm(len(gp), generator=torch.Generator().manual_seed(int(cfg["seed"]) + layer))
        null = transition_matrix(sp, sm[permutation])
        down = decoder_layers(teacher)[layer - 1].mlp.down_proj
        weight = down.weight if hasattr(down, "weight") else down.base_layer.weight
        out_norm = weight.detach().float().norm(dim=0).cpu()
        delta_plus = F.silu(gp) * up
        delta_minus = F.silu(gm) * um
        raw = (delta_plus - delta_minus).abs() * out_norm
        raw_null = (delta_plus - delta_minus[permutation]).abs() * out_norm
        support = torch.stack([(sp == a).sum(dim=(0, 1)) > 0 for a in range(4)], dim=-1)
        magnitude = torch.zeros_like(out_norm).unsqueeze(-1).repeat(1, 4)
        null_magnitude = torch.zeros_like(magnitude)
        for a in range(4):
            mask = sp == a
            count = mask.sum(dim=(0, 1)).clamp_min(1)
            magnitude[:, a] = (raw * mask).sum(dim=(0, 1)) / count
            null_magnitude[:, a] = (raw_null * mask).sum(dim=(0, 1)) / count
        specificity = (true - null).abs().sum(dim=-1) * magnitude / (null_magnitude + 1e-6)
        specificity = specificity.masked_fill(~support, -torch.inf)
        flat = specificity.flatten(0, 1)
        k = min(max_neurons, flat.numel())
        values, indices = torch.topk(flat, k)
        rows = []
        for value, index in zip(values, indices):
            neuron, source = divmod(int(index), 4)
            if float(value) >= min_specificity:
                rows.append({"neuron": neuron, "source": source, "weight": float(value)})
        if not rows:
            selected[layer] = []
            reports[str(layer)] = {"selected": 0}
            continue
        normalizer = max(sum(x["weight"] for x in rows), 1e-8)
        for row in rows:
            row["weight"] = max(row["weight"] / normalizer, 1e-4)
        selected[layer] = rows
        odd, even = sp[0::2], sp[1::2]
        half_true = transition_matrix(sp[0::2], sm[0::2])
        half_second = transition_matrix(sp[1::2], sm[1::2])
        stability = F.cosine_similarity((half_true - null).flatten(), (half_second - null).flatten(), dim=0)
        stable = float(stability) >= float(cfg["method"].get("min_split_cosine", 0.0))
        if not stable and bool(cfg["method"].get("strict_gate", True)):
            selected[layer] = []
        reports[str(layer)] = {"selected": len(selected[layer]), "max_specificity": float(values.max()),
                               "split_specificity_cosine": float(stability), "split_gate": stable,
                               "transition_l1_mean": float((true - null).abs().sum(-1).mean()),
                               "source_states": {str(a): int((sp == a).sum()) for a in range(4)}}
    has_selection = any(selected[layer] for layer in cfg["layers"])
    enabled = has_selection
    if not has_selection and not bool(cfg["method"].get("strict_gate", True)):
        raise RuntimeError("no SwiGLU source states available even without a strict gate")
    return {"enabled": enabled, "selected": selected, "reports": reports,
            "temperature": float(cfg["method"].get("temperature", 0.1))}


def loss(batch, prepared, cfg):
    terms = []
    for row in batch:
        item, plus = row["item"], row["plus"]
        for layer in cfg["layers"]:
            for entry in prepared["selected"][layer]:
                neuron, source = entry["neuron"], entry["source"]
                g = plus["glu"][layer]["g"][:, neuron]
                u = plus["glu"][layer]["u"][:, neuron]
                occupancy = soft_occupancy(g, u, source, prepared["temperature"]).mean()
                terms.append(occupancy * entry["weight"])
    if not terms:
        raise RuntimeError("no GRTE source-state loss")
    return {"method_loss": torch.stack(terms).sum(), "active_source_states": torch.tensor(float(len(terms)))}


def diagnostics(cache, prepared, cfg):
    del cache, cfg
    return {"operator": "swiglu_four_state_true_null_transition_source_exit", "layers": prepared["reports"],
            "selected_per_layer": {str(k): len(v) for k, v in prepared["selected"].items()},
            "enabled": prepared["enabled"], "donor_target_state": False,
            "warning": "state specificity is not causal without the proposal's patch/readout audit"}


def self_test():
    g = torch.tensor([2.0, -2.0])
    u = torch.tensor([1.0, -1.0])
    assert state_code(g, u).tolist() == [3, 0]
    match = soft_occupancy(g[0], u[0], 3, 0.1)
    mismatch = soft_occupancy(g[1], u[1], 3, 0.1)
    assert match > 0.99 and mismatch < 0.01
    matrix = transition_matrix(torch.tensor([[[3], [3]]]), torch.tensor([[[0], [1]]]))
    assert float(matrix[0, 3].sum()) == 1.0
    assert float(matrix[0, 0].sum()) == 0.0
    return {"method_id": METHOD_ID, "status": "PASS", "state_codes": True, "soft_occupancy": True}
