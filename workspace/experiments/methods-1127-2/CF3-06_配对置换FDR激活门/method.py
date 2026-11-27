"""CF3-06 PEFG: paired sign permutations, BY-FDR and held-out patch readout gates."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from modeling import patch_forward_nll, span_hidden

METHOD_ID = "CF3-06"
NEEDS_STUDENT_MINUS = False
REQUIRES_CONTROL = False


def by_mask(pvalues, q_value):
    m = len(pvalues)
    if m == 0:
        return torch.zeros(0, dtype=torch.bool)
    order = torch.argsort(pvalues)
    harmonic = torch.arange(1, m + 1, dtype=torch.float64).reciprocal().sum()
    ranks = torch.arange(1, m + 1, dtype=pvalues.dtype)
    thresholds = q_value * ranks / (m * harmonic)
    passing = torch.nonzero(pvalues[order] <= thresholds, as_tuple=False).flatten()
    mask = torch.zeros(m, dtype=torch.bool)
    if len(passing):
        cutoff = pvalues[order][passing[-1]]
        mask = pvalues <= cutoff
    return mask


def _paired_pvalue(values, permutations, seed):
    observed = values.mean(0).pow(2).sum().double()
    generator = torch.Generator().manual_seed(seed)
    signs = torch.randint(0, 2, (permutations, len(values)), generator=generator).double() * 2 - 1
    null = (signs.unsqueeze(-1) * values.double().unsqueeze(0)).mean(1).pow(2).sum(-1)
    return (1 + (null >= observed).sum()).double() / (permutations + 1), observed


def prepare(cache, teacher, tok, cfg, device, student=None, out_dir=None):
    del tok, student, out_dir
    if not cache["pairs"][0]["full_plus"]:
        raise RuntimeError("CF3-06 requires data.cache_full_sequence=true for held-out patch gates")
    items = cache["pairs"]
    if len(items) < 4:
        raise RuntimeError("CF3-06 needs at least four pairs for selection/held-out splits")
    selection = items[0::2]
    heldout = items[1::2]
    max_positions = int(cfg["method"].get("max_answer_positions", 16))
    permutations = int(cfg["method"].get("permutations", 199))
    hypotheses, records = [], []
    for layer in cfg["layers"]:
        rms = torch.stack([x["answer"][layer].pow(2).mean().sqrt() for x in cache["retain"]]).mean().clamp_min(1e-6)
        for position in range(max_positions):
            values = [(x["h_plus"][layer][position] - x["h_minus"][layer][position]) / rms
                      for x in selection if len(x["h_plus"][layer]) > position]
            if len(values) < 2:
                continue
            stacked = torch.stack(values)
            pvalue, observed = _paired_pvalue(stacked, permutations, int(cfg["seed"]) + layer * 1000 + position)
            first, second = stacked[0::2].mean(0), stacked[1::2].mean(0)
            stability = float(F.cosine_similarity(first, second, dim=0, eps=1e-8))
            hypotheses.append(len(records))
            records.append({"layer": layer, "position": position, "pvalue": pvalue,
                            "observed": observed, "stability": stability})
    pvalues = torch.stack([x["pvalue"] for x in records])
    significant = by_mask(pvalues, float(cfg["method"].get("fdr_q", 0.1)))
    weights, reports = {}, []
    alpha = float(cfg["method"].get("alpha", 0.1))
    for index, record in enumerate(records):
        layer, position = record["layer"], record["position"]
        stable = record["stability"] >= float(cfg["method"].get("min_split_cosine", 0.0))
        candidate = bool(significant[index]) and stable
        effects = []
        if candidate:
            for item in heldout[:int(cfg["method"].get("gate_pairs", 2))]:
                if len(item["h_plus"][layer]) <= position:
                    continue
                full = item["full_plus"][layer].clone()
                absolute = item["enc_plus"]["answer_start"] + position
                full[absolute] -= alpha * (item["h_plus"][layer][position] - item["h_minus"][layer][position])
                patched = patch_forward_nll(teacher, item["enc_plus"], device, layer - 1, full)
                effects.append(float(patched) - float(item["teacher_nll"]))
        mean_effect = sum(effects) / max(1, len(effects))
        passed = candidate and mean_effect > float(cfg["method"].get("min_nll_increase", 0.0))
        if passed:
            weights[(layer, position)] = 1.0
        reports.append({"layer": layer, "position": position, "pvalue": float(record["pvalue"]),
                        "split_cosine": record["stability"], "significant": bool(significant[index]),
                        "heldout_nll_increase": mean_effect, "passed": passed})
    if not weights and not bool(cfg["method"].get("strict_gate", True)):
        best = max(records, key=lambda x: float(x["observed"]))
        weights[(best["layer"], best["position"])] = 1.0
    return {"enabled": bool(weights), "weights": weights, "reports": reports,
            "alpha": alpha, "hypotheses": len(records)}


def loss(batch, prepared, cfg):
    terms = []
    for row in batch:
        item, plus = row["item"], row["plus"]
        for (layer, position), weight in prepared["weights"].items():
            if len(item["h_plus"][layer]) <= position:
                continue
            hp = item["h_plus"][layer][position].to(plus["nll"].device)
            hm = item["h_minus"][layer][position].to(plus["nll"].device)
            sh = span_hidden(plus, layer)[position]
            target = hp - prepared["alpha"] * weight * (hp - hm)
            terms.append(F.mse_loss(sh, target.detach()))
    if not terms:
        raise RuntimeError("no PEFG-gated answer positions")
    return {"method_loss": torch.stack(terms).mean(), "active_positions": torch.tensor(float(len(terms)))}


def diagnostics(cache, prepared, cfg):
    del cache, cfg
    return {"operator": "paired_sign_permutation_BY_FDR_heldout_patch", "hypotheses": prepared["hypotheses"],
            "selected": len(prepared["weights"]), "enabled": prepared["enabled"], "gates": prepared["reports"],
            "data_split": "even pairs select; odd pairs are held out for patch readout"}


def self_test():
    pvalues = torch.tensor([0.001, 0.02, 0.2, 0.8], dtype=torch.float64)
    assert bool(by_mask(pvalues, 0.1)[0])
    assert not bool(by_mask(torch.ones(4), 0.1).any())
    values = torch.randn(8, 3)
    pvalue, _ = _paired_pvalue(values, 49, 3)
    assert 1 / 50 <= float(pvalue) <= 1
    return {"method_id": METHOD_ID, "status": "PASS", "by_fdr": True, "pvalue_range": True}
