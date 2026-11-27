"""CF3-08 CTDE: retain-rank counterfactual copula tail dependence and marginal deadzones."""
from __future__ import annotations

import torch
import torch.nn.functional as F

METHOD_ID = "CF3-08"
NEEDS_STUDENT_MINUS = False
REQUIRES_CONTROL = False


def empirical_cdf(values, reference, temperature=0.0):
    sorted_reference = reference.sort(dim=0).values
    if temperature > 0:
        return torch.sigmoid((values.unsqueeze(1) - sorted_reference.unsqueeze(0)) / temperature).mean(1)
    output = []
    for row in values:
        output.append(torch.stack([(sorted_reference[:, j] <= row[j]).float().mean() for j in range(values.shape[-1])]))
    return torch.stack(output)


def soft_tail(u, v, threshold, temperature):
    au = torch.sigmoid((u - threshold) / temperature)
    av = torch.sigmoid((v - threshold) / temperature)
    return (au * av).mean(0) / av.mean(0).clamp_min(1e-5)


def _tail_vector(u, v, quantiles, temperature=0.02):
    return torch.cat([soft_tail(u, v, q, temperature) for q in quantiles])


def prepare(cache, teacher, tok, cfg, device, student=None, out_dir=None):
    del teacher, tok, student, out_dir
    generator = torch.Generator().manual_seed(int(cfg["seed"]))
    quantiles = [float(x) for x in cfg["method"].get("quantiles", [0.7, 0.8, 0.9])]
    references, weights, null_targets, reports = {}, {}, {}, {}
    for layer in cfg["layers"]:
        width = cache["pairs"][0]["h_plus"][layer].shape[-1]
        q = min(int(cfg["method"].get("projection_dim", 2)), width)
        ps = torch.linalg.qr(torch.randn(width, q, generator=generator), mode="reduced").Q
        pa = torch.linalg.qr(torch.randn(width, q, generator=generator), mode="reduced").Q
        up = torch.stack([x["subject_plus"][layer] for x in cache["pairs"]]) @ ps
        um = torch.stack([x["subject_minus"][layer] for x in cache["pairs"]]) @ ps
        vp = torch.stack([x["h_plus"][layer].mean(0) for x in cache["pairs"]]) @ pa
        vm = torch.stack([x["h_minus"][layer].mean(0) for x in cache["pairs"]]) @ pa
        retain_u = torch.stack([x["subject"][layer] for x in cache["retain"]]) @ ps
        retain_v = torch.stack([x["pooled"][layer] for x in cache["retain"]]) @ pa
        u_plus, u_minus = empirical_cdf(up, retain_u), empirical_cdf(um, retain_u)
        v_plus, v_minus = empirical_cdf(vp, retain_v), empirical_cdf(vm, retain_v)
        true = _tail_vector(u_plus, v_plus, quantiles) - _tail_vector(u_minus, v_minus, quantiles)
        permutation = torch.randperm(len(up), generator=generator)
        null = _tail_vector(u_plus, v_plus[permutation], quantiles) - _tail_vector(u_minus, v_minus[permutation], quantiles)
        rho = true - null
        weight = rho.abs() / (rho.abs().sum().clamp_min(1e-6))
        references[layer] = {"u": retain_u.to(device), "v": retain_v.to(device), "ps": ps.to(device), "pa": pa.to(device)}
        weights[layer] = weight.to(device)
        null_targets[layer] = _tail_vector(u_plus, v_plus[permutation], quantiles).to(device)
        reports[str(layer)] = {"true_tail_delta": true.tolist(), "null_tail_delta": null.tolist(),
                               "rho": rho.tolist(), "mean_u_plus": u_plus.mean(0).tolist(),
                               "mean_v_plus": v_plus.mean(0).tolist(), "projection_dim": q}
    enabled = any(float(weights[layer].max()) > 0 for layer in cfg["layers"])
    return {"enabled": enabled, "references": references, "weights": weights, "null_targets": null_targets,
            "quantiles": quantiles, "reports": reports,
            "temperature": float(cfg["method"].get("temperature", 0.05)),
            "marginal_deadzone": float(cfg["method"].get("marginal_deadzone", 0.1))}


def _student_coordinates(row, layer, reference, temperature):
    item, plus = row["item"], row["plus"]
    subject = torch.stack([plus["hidden"][layer][i] for i in item["subject_indices"]]).mean(0).float() @ reference["ps"]
    start, end = plus["enc"]["answer_start"], plus["enc"]["answer_end"]
    answer = plus["hidden"][layer][start:end].mean(0).float() @ reference["pa"]
    u = empirical_cdf(subject.unsqueeze(0), reference["u"], temperature)[0]
    v = empirical_cdf(answer.unsqueeze(0), reference["v"], temperature)[0]
    return u, v


def loss(batch, prepared, cfg):
    tails, margins, used = [], [], []
    device = batch[0]["plus"]["nll"].device
    for layer in cfg["layers"]:
        coordinates = [_student_coordinates(row, layer, prepared["references"][layer], prepared["temperature"]) for row in batch]
        u = torch.stack([x[0] for x in coordinates])
        v = torch.stack([x[1] for x in coordinates])
        for index, quantile in enumerate(prepared["quantiles"]):
            current = soft_tail(u, v, quantile, prepared["temperature"])
            target = prepared["null_targets"][layer][index]
            tails.append(prepared["weights"][layer][index] * (current - target).pow(2).mean())
        for quantile in prepared["quantiles"]:
            marginal_u = (torch.quantile(u, quantile, dim=0) - quantile).abs()
            marginal_v = (torch.quantile(v, quantile, dim=0) - quantile).abs()
            margins.append(F.relu(marginal_u - prepared["marginal_deadzone"]).mean() +
                           F.relu(marginal_v - prepared["marginal_deadzone"]).mean())
        used.append(u.new_tensor(float(len(batch))))
    if not tails:
        raise RuntimeError("no copula tails")
    marginal_weight = float(cfg["method"].get("marginal_weight", 0.1))
    return {"method_loss": torch.stack(tails).mean(), "extra_loss": marginal_weight * torch.stack(margins).mean(),
            "tail_terms": torch.stack(used).sum()}


def diagnostics(cache, prepared, cfg):
    del cache, cfg
    return {"operator": "retain_empirical_cdf_counterfactual_soft_copula_tail", "layers": prepared["reports"],
            "enabled": prepared["enabled"], "null_training_target": "per-layer shuffled-pair tail level",
            "warning": "mini-batch soft tail estimates require sample-size bias checks"}


def self_test():
    reference = torch.linspace(0, 1, 101).unsqueeze(1)
    values = torch.tensor([[0.0], [0.5], [1.0]])
    transformed = empirical_cdf(values, reference)
    assert torch.allclose(transformed[:, 0], torch.tensor([0.0099, 0.50495, 1.0]), atol=1e-3)
    u = torch.linspace(0, 1, 100).unsqueeze(1)
    dependent = soft_tail(u, u, 0.8, 0.01)
    independent = soft_tail(u, u[torch.randperm(100)], 0.8, 0.01)
    assert float(dependent) > float(independent)
    return {"method_id": METHOD_ID, "status": "PASS", "ecdf": True, "tail_order": True}
