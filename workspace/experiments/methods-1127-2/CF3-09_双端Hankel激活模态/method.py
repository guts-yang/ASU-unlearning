"""CF3-09 DHAM: randomized low-rank dual controllability/readout Hankel modes."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from modeling import answer_nll, patch_forward_output_multi

METHOD_ID = "CF3-09"
NEEDS_STUDENT_MINUS = False
REQUIRES_CONTROL = False


def balanced_modes(cu, rank):
    gram = cu.T @ cu
    evals, vec = torch.linalg.eigh(gram.double())
    order = torch.argsort(evals.clamp_min(0), descending=True)[:min(rank, len(evals))]
    return vec[:, order].float(), evals[order].float()


@torch.no_grad()
def _source_state(model, embeddings, enc, layer):
    out = model(inputs_embeds=embeddings.unsqueeze(0), attention_mask=enc["attention_mask"].to(embeddings.device).unsqueeze(0),
                output_hidden_states=True, use_cache=False)
    return out.hidden_states[layer][0].float(), out.logits[0].float()


def _readout(logits, enc, projection):
    logits = logits.float()
    target = -answer_nll(logits, enc["labels"].to(logits.device)).unsqueeze(0)
    start, end = enc["answer_start"], enc["answer_end"]
    random_part = logits[start:end].mean(0) @ projection / (logits.shape[-1] ** 0.5)
    return torch.cat([target, random_part])


@torch.no_grad()
def _dual_basis(model, item, layer, cfg, device):
    enc = item["enc_plus"]
    ids, minus_ids = enc["input_ids"].to(device), item["enc_minus"]["input_ids"].to(device)
    embedding = model.get_input_embeddings()
    plus, minus = embedding(ids).detach(), embedding(minus_ids).detach()
    midpoint = plus.clone()
    subject = list(item["subject_indices"])
    midpoint[subject] = 0.5 * (plus[subject] + minus[subject])
    base_hidden, _ = _source_state(model, midpoint, enc, layer)
    start, end = enc["answer_start"], enc["answer_end"]
    probes = int(cfg["method"].get("random_probes", 4))
    epsilon = float(cfg["method"].get("jacobian_epsilon", 0.01))
    generator = torch.Generator().manual_seed(int(cfg["seed"]) + item["index"] * 100 + layer)
    columns = []
    for _ in range(probes):
        direction = torch.randn(len(subject), plus.shape[-1], generator=generator).to(device)
        direction = F.normalize(direction, dim=-1)
        low, high = midpoint.clone(), midpoint.clone()
        low[subject] -= epsilon * direction
        high[subject] += epsilon * direction
        low_h, _ = _source_state(model, low, enc, layer)
        high_h, _ = _source_state(model, high, enc, layer)
        columns.append((high_h[start:end].mean(0) - low_h[start:end].mean(0)) / (2 * epsilon))
    range_matrix = torch.stack(columns, dim=-1)
    u = torch.linalg.qr(range_matrix, mode="reduced").Q
    vocab = int(model.config.vocab_size)
    readout_dim = int(cfg["method"].get("readout_dim", 3))
    projection = torch.randn(vocab, readout_dim, generator=generator).to(device=device, dtype=base_hidden.dtype)
    cu_columns = []
    for column in u.T:
        low, high = base_hidden.clone(), base_hidden.clone()
        low[start:end] -= epsilon * column
        high[start:end] += epsilon * column
        low_out = patch_forward_output_multi(model, enc, device, {layer - 1: low})
        high_out = patch_forward_output_multi(model, enc, device, {layer - 1: high})
        low_readout = _readout(low_out["logits"], enc, projection)
        high_readout = _readout(high_out["logits"], enc, projection)
        cu_columns.append((high_readout - low_readout) / (2 * epsilon))
    cu = torch.stack(cu_columns, dim=-1)
    modes, spectrum = balanced_modes(cu, int(cfg["method"].get("rank", 2)))
    basis = torch.linalg.qr(u @ modes, mode="reduced").Q
    return basis.cpu(), spectrum.cpu(), cu.detach().cpu()


def prepare(cache, teacher, tok, cfg, device, student=None, out_dir=None):
    del tok, student, out_dir
    limit = min(int(cfg["method"].get("operator_pairs", 4)), len(cache["pairs"]))
    items = cache["pairs"]
    targets, reports = {}, {}
    margin = float(cfg["method"].get("null_margin", 0.0))
    for layer in cfg["layers"]:
        for i, item in enumerate(items[:limit]):
            basis, spectrum, _ = _dual_basis(teacher, item, layer, cfg, device)
            null = items[(i + 1) % limit]
            true_delta = item["h_plus"][layer].mean(0) - item["h_minus"][layer].mean(0)
            null_delta = item["h_plus"][layer].mean(0) - null["h_minus"][layer].mean(0)
            true_energy = (basis.T @ true_delta).pow(2)
            null_energy = (basis.T @ null_delta).pow(2)
            selected = true_energy > null_energy * (1.0 + margin)
            if not selected.any() and not bool(cfg["method"].get("strict_gate", True)):
                selected[true_energy.argmax()] = True
            if selected.any():
                chosen = basis[:, selected]
                filtered = chosen @ (chosen.T @ true_delta)
                targets[(item["index"], layer)] = item["h_plus"][layer].mean(0) - float(cfg["method"].get("alpha", 0.1)) * filtered
            reports[f"{item['index']}:{layer}"] = {"spectrum": spectrum.tolist(),
                                                    "true_modal_energy": true_energy.tolist(),
                                                    "null_modal_energy": null_energy.tolist(),
                                                    "selected_modes": int(selected.sum())}
    enabled = all(any((item["index"], layer) in targets for item in items[:limit]) for layer in cfg["layers"])
    return {"enabled": enabled, "targets": targets, "reports": reports,
            "operator_pairs": limit, "alpha": float(cfg["method"].get("alpha", 0.1))}


def loss(batch, prepared, cfg):
    terms = []
    for row in batch:
        item, plus = row["item"], row["plus"]
        start, end = plus["enc"]["answer_start"], plus["enc"]["answer_end"]
        for layer in cfg["layers"]:
            key = (item["index"], layer)
            if key not in prepared["targets"]:
                continue
            target = prepared["targets"][key].to(plus["nll"].device)
            terms.append(F.mse_loss(plus["hidden"][layer][start:end].mean(0), target.detach()))
    if not terms:
        raise RuntimeError("no dual-end Hankel targets")
    return {"method_loss": torch.stack(terms).mean()}


def diagnostics(cache, prepared, cfg):
    del cache, cfg
    return {"operator": "randomized_finite_jvp_dual_hankel_subspace", "operator_pairs": prepared["operator_pairs"],
            "enabled": prepared["enabled"], "reports": prepared["reports"],
            "scope": "local midpoint, answer-pooled, low-rank; no full d-by-d Jacobian"}


def self_test():
    cu = torch.diag(torch.tensor([4.0, 2.0, 1.0]))
    modes, spectrum = balanced_modes(cu, 2)
    assert torch.allclose(spectrum, torch.tensor([16.0, 4.0]))
    assert float((modes.T @ modes - torch.eye(2)).abs().max()) < 1e-5
    return {"method_id": METHOD_ID, "status": "PASS", "balanced_spectrum": spectrum.tolist()}
