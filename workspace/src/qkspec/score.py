"""Discovery scoring for QKSpec.

Stage A (screening / correlation score, **not** causal evidence)
---------------------------------------------------------------
For each candidate (layer, head, singular-rank) we accumulate, over D_F, the
spectral logit contribution on "target retrieval edges":

    edges      = answer positions (labels != -100) x prompt positions
    contrib_j  = sigma_j * sum_{a,p} attn[a,p] * (x_q[a] . u_j) * (v_j . x_k[p])

Then subtract the same quantity on D_R, which (per user clarification) is
itself the same-relation / same-template / different-entity near-neighbour
control set.  A is therefore

    A_j = mean_F[contrib_j] - mean_R[contrib_j]

used only to short-list Top-K directions.  It is a correlation/screening
score and must never be presented as causal proof.

Stage B (local causal re-check)
-------------------------------
For each Top-K direction we temporarily contract that single sigma by rho,
re-run forward on D_F / D_R, and measure NLL increase on answer tokens:

    B_j             = NLL_edited(D_F) - NLL_original(D_F)
    retain_impact_j = NLL_edited(D_R) - NLL_original(D_R)

Only B is labelled "local causal evidence".  Original weights are always
restored after each direction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import QKSpecConfig
from .hooks import ActivationCache, capture_qk_activations, require_eager_cache
from .qk_math import (
    DirectionSpec,
    compute_delta_wq,
    fold_into_q,
    get_head_slices,
    get_k_head_slices,
    svd_qk_lowrank,
)


@dataclass
class ScoreResult:
    """Diagnostics produced by stage A and stage B."""

    stage_a_scores: List[DirectionSpec] = field(default_factory=list)
    stage_b_scores: List[DirectionSpec] = field(default_factory=list)
    head_prescreen: Dict[str, float] = field(default_factory=dict)
    retain_pairing: str = "paired"
    notes: List[str] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        return {
            "stage_a_direction_count": len(self.stage_a_scores),
            "stage_b_direction_count": len(self.stage_b_scores),
            "retain_pairing": self.retain_pairing,
            "notes": list(self.notes),
        }


def _move_batch(batch: Any, device: torch.device) -> Dict[str, torch.Tensor]:
    """Normalise a collator output to a flat tensor dict on ``device``."""
    if isinstance(batch, dict) and "input_ids" in batch:
        return {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
            "labels": batch["labels"].to(device),
        }
    raise ValueError(
        "expected a flat batch with input_ids/attention_mask/labels; "
        "pass D_F / D_R loaders built from QADataset directly"
    )


def _answer_prompt_masks(
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return boolean masks ``[B, S]`` for answer tokens and prompt tokens."""
    answer = labels.ne(-100) & attention_mask.bool()
    prompt = labels.eq(-100) & attention_mask.bool()
    return answer, prompt


def _head_dim(attn: nn.Module) -> int:
    hd = int(getattr(attn, "head_dim", 0) or 0)
    if hd > 0:
        return hd
    return int(attn.q_proj.out_features // attn.num_heads)


def _get_layers(model: nn.Module):
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise RuntimeError("QKSpec requires a Llama-like model with .model.layers")
    return layers


def _layer_indices(model: nn.Module, cfg: QKSpecConfig) -> List[int]:
    layers = _get_layers(model)
    indices = list(range(len(layers)))
    if cfg.layer_start is not None:
        indices = [i for i in indices if i >= cfg.layer_start]
    if cfg.layer_end is not None:
        indices = [i for i in indices if i <= cfg.layer_end]
    return indices


def _spectral_head_contribution(
    x_q: torch.Tensor,          # [B, S, d_model]
    x_k: torch.Tensor,          # [B, S, d_model]
    attn_head: torch.Tensor,    # [B, S, S]  (one head)
    answer: torch.Tensor,       # [B, S] bool
    prompt: torch.Tensor,       # [B, S] bool
    w_q_head: torch.Tensor,     # [d_head, d_model]
    w_k_head: torch.Tensor,     # [d_head, d_model]
    ranks: Sequence[int],
    edge_top_fraction: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vectorised spectral contribution over retrieval edges for one head.

    Returns ``(contrib [|ranks|], sigma [d_head])``.
    """
    U, sigma, V = svd_qk_lowrank(w_q_head, w_k_head)  # U,V: [d_model, r]
    ranks = [r for r in ranks if r < U.shape[1]]
    if not ranks:
        return torch.zeros(0, dtype=torch.float32, device=x_q.device), sigma
    U_J = U[:, ranks].to(torch.float32)    # [d_model, |J|]
    V_J = V[:, ranks].to(torch.float32)

    contrib = torch.zeros(len(ranks), dtype=torch.float32, device=x_q.device)
    B = x_q.shape[0]
    for b in range(B):
        a_idx = answer[b].nonzero(as_tuple=False).squeeze(-1)
        p_idx = prompt[b].nonzero(as_tuple=False).squeeze(-1)
        if a_idx.numel() == 0 or p_idx.numel() == 0:
            continue
        edge = attn_head[b][a_idx][:, p_idx].float()     # [A, P]
        if edge_top_fraction < 1.0:
            keep = max(1, int(edge.numel() * edge_top_fraction))
            flat = edge.flatten()
            threshold = torch.topk(flat, keep).values.min()
            edge = (edge >= threshold).float() * edge
        q_proj = x_q[b, a_idx].to(torch.float32) @ U_J   # [A, |J|]
        k_proj = x_k[b, p_idx].to(torch.float32) @ V_J   # [P, |J|]
        weighted_k = edge @ k_proj                        # [A, |J|]
        contrib += (q_proj * weighted_k).sum(dim=0)       # [|J|]
    contrib = contrib * sigma[list(ranks)].to(torch.float32)
    return contrib, sigma.detach()


def prescreen_heads(
    model: nn.Module,
    forget_loader: DataLoader,
    cfg: QKSpecConfig,
    device: torch.device,
) -> Tuple[List[Tuple[int, int]], Dict[str, float]]:
    """Cheap head pre-screen using attention mass on answer->prompt edges."""
    layer_indices = _layer_indices(model, cfg)
    cache = ActivationCache()
    mass: Dict[Tuple[int, int], float] = {}
    seen = 0
    checked = False
    with capture_qk_activations(model, cache):
        for batch in forget_loader:
            if cfg.max_discovery_batches is not None and seen >= cfg.max_discovery_batches:
                break
            inputs = _move_batch(batch, device)
            with torch.inference_mode():
                model(**inputs, output_attentions=True)
            if not checked:
                require_eager_cache(cache)
                checked = True
            answer, prompt = _answer_prompt_masks(inputs["labels"], inputs["attention_mask"])
            for li in layer_indices:
                act = cache.get(li)
                if act.attn_weights is None:
                    continue
                a = act.attn_weights  # [B,H,S,S] for this layer
                for b in range(a.shape[0]):
                    for h in range(a.shape[1]):
                        sub = a[b, h][answer[b]][:, prompt[b]]
                        mass[(li, h)] = mass.get((li, h), 0.0) + float(sub.sum())
            seen += 1

    per_layer: Dict[int, List[Tuple[float, int]]] = {li: [] for li in layer_indices}
    for (li, h), v in mass.items():
        per_layer.setdefault(li, []).append((v, h))
    selected: List[Tuple[int, int]] = []
    stats: Dict[str, float] = {}
    for li in layer_indices:
        ranked = sorted(per_layer.get(li, []), reverse=True)[: cfg.shortlist_per_layer]
        for v, h in ranked:
            selected.append((li, h))
            stats[f"layer_{li}_head_{h}"] = v
    selected = selected[: cfg.max_candidate_heads]
    return selected, stats


def score_stage_a(
    model: nn.Module,
    forget_loader: DataLoader,
    retain_loader: DataLoader,
    cfg: QKSpecConfig,
    device: torch.device,
) -> ScoreResult:
    """Compute A = mean_F(contrib) - mean_R(contrib) for shortlisted heads."""
    result = ScoreResult()
    heads, stats = prescreen_heads(model, forget_loader, cfg, device)
    result.head_prescreen = stats
    rank_budget = cfg.ranks_per_head
    layers = _get_layers(model)

    f_sum: Dict[Tuple[int, int], torch.Tensor] = {}
    r_sum: Dict[Tuple[int, int], torch.Tensor] = {}
    sigmas: Dict[Tuple[int, int], torch.Tensor] = {}
    counts = {"f": 0, "r": 0}

    def accumulate(loader, sign, target, count_key):
        cache = ActivationCache()
        seen = 0
        checked = False
        with capture_qk_activations(model, cache):
            for batch in loader:
                if cfg.max_discovery_batches is not None and seen >= cfg.max_discovery_batches:
                    break
                inputs = _move_batch(batch, device)
                with torch.inference_mode():
                    model(**inputs, output_attentions=True)
                if not checked:
                    require_eager_cache(cache)
                    checked = True
                answer, prompt = _answer_prompt_masks(inputs["labels"], inputs["attention_mask"])
                for li, h in heads:
                    act = cache.get(li)
                    if act.q_input is None or act.k_input is None or act.attn_weights is None:
                        continue
                    if h >= act.attn_weights.shape[1]:
                        continue
                    attn_head = act.attn_weights[:, h]         # [B,S,S]
                    attn_mod = layers[li].self_attn
                    hd = _head_dim(attn_mod)
                    w_q_h = get_head_slices(attn_mod.q_proj.weight, h, attn_mod.num_heads, hd)
                    w_k_h = get_k_head_slices(
                        attn_mod.k_proj.weight, h, attn_mod.num_heads,
                        attn_mod.num_key_value_heads, hd,
                    )
                    contrib, sigma = _spectral_head_contribution(
                        act.q_input, act.k_input, attn_head, answer, prompt,
                        w_q_h, w_k_h, list(range(rank_budget)), cfg.edge_top_fraction,
                    )
                    key = (li, h)
                    if key not in target:
                        target[key] = torch.zeros(rank_budget, dtype=torch.float32, device=device)
                    target[key] += sign * contrib
                    sigmas[key] = sigma
                counts[count_key] += 1
                seen += 1

    accumulate(forget_loader, +1.0, f_sum, "f")
    accumulate(retain_loader, -1.0, r_sum, "r")

    for li, h in heads:
        f = f_sum.get((li, h), torch.zeros(rank_budget))
        r = r_sum.get((li, h), torch.zeros(rank_budget))
        denom = max(1, counts["f"], counts["r"])
        diff = (f - r) / denom
        sigma_vec = sigmas.get((li, h), torch.zeros(rank_budget))
        for j, value in enumerate(diff.tolist()):
            result.stage_a_scores.append(
                DirectionSpec(
                    layer=li,
                    head=h,
                    rank=j,
                    sigma=float(sigma_vec[j]) if j < sigma_vec.numel() else 0.0,
                    stage_a_score=float(value),
                    rho=cfg.rho,
                    meta={
                        "forget_contrib": float(f[j]) / denom,
                        "retain_contrib": float(r[j]) / denom,
                    },
                )
            )
    result.notes.append(
        "Stage A scores are screening/correlation scores, NOT causal evidence."
    )
    return result


def _apply_temporary_contraction(
    model: nn.Module,
    direction: DirectionSpec,
    rho: float,
) -> torch.Tensor:
    """Temporarily fold delta for one direction; return original q weight."""
    layers = _get_layers(model)
    attn = layers[direction.layer].self_attn
    q_proj = attn.q_proj
    hd = _head_dim(attn)
    w_q_h = get_head_slices(q_proj.weight, direction.head, attn.num_heads, hd)
    w_k_h = get_k_head_slices(
        attn.k_proj.weight, direction.head, attn.num_heads,
        attn.num_key_value_heads, hd,
    )
    U, sigma, V = svd_qk_lowrank(w_q_h, w_k_h)
    if direction.rank >= U.shape[1]:
        raise ValueError(
            f"direction rank {direction.rank} out of range for head "
            f"(rank budget {U.shape[1]})"
        )
    _, delta_w_q_doc = compute_delta_wq(U, sigma, V, [direction.rank], rho, w_k_h)
    original = q_proj.weight.detach().clone()
    edited = fold_into_q(q_proj.weight.data, direction.head, delta_w_q_doc, attn.num_heads, hd)
    with torch.no_grad():
        q_proj.weight.data.copy_(edited)
    return original


def _restore_q_weight(model: nn.Module, direction: DirectionSpec, original: torch.Tensor) -> None:
    q_proj = _get_layers(model)[direction.layer].self_attn.q_proj
    with torch.no_grad():
        q_proj.weight.data.copy_(original)


def _nll_over_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int],
) -> float:
    total_loss = 0.0
    total_tokens = 0
    seen = 0
    with torch.inference_mode():
        for batch in loader:
            if max_batches is not None and seen >= max_batches:
                break
            inputs = _move_batch(batch, device)
            outputs = model(**inputs)
            if outputs.loss is None:
                raise RuntimeError("model forward returned no loss; check labels")
            n_tokens = int(inputs["labels"].ne(-100).sum())
            total_loss += float(outputs.loss) * n_tokens
            total_tokens += n_tokens
            seen += 1
    return total_loss / max(total_tokens, 1)


def score_stage_b(
    model: nn.Module,
    forget_loader: DataLoader,
    retain_loader: DataLoader,
    directions: Sequence[DirectionSpec],
    cfg: QKSpecConfig,
    device: torch.device,
) -> List[DirectionSpec]:
    """Ablate each direction and measure NLL change on D_F and D_R."""
    baseline_f = _nll_over_loader(model, forget_loader, device, cfg.max_stage_b_batches)
    baseline_r = _nll_over_loader(model, retain_loader, device, cfg.max_stage_b_batches)

    results: List[DirectionSpec] = []
    for direction in directions:
        original = _apply_temporary_contraction(model, direction, cfg.rho)
        try:
            edited_f = _nll_over_loader(model, forget_loader, device, cfg.max_stage_b_batches)
            edited_r = _nll_over_loader(model, retain_loader, device, cfg.max_stage_b_batches)
        finally:
            _restore_q_weight(model, direction, original)
        results.append(
            DirectionSpec(
                layer=direction.layer,
                head=direction.head,
                rank=direction.rank,
                sigma=direction.sigma,
                stage_a_score=direction.stage_a_score,
                stage_b_score=float(edited_f - baseline_f),
                rho=cfg.rho,
                causal_verified=True,
                retain_impact=float(edited_r - baseline_r),
                meta=dict(direction.meta),
            )
        )
    return results
