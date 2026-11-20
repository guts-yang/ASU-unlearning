"""Pure math primitives for QKSpec.

Notation follows ``workspace/docs/K4_QKSpec_故事与机制.md`` (row-vector
convention):

    q . k = x_q @ M_{l,h} @ x_k.T
    M_{l,h} = W_Q_{l,h} @ W_K_{l,h}.T       (pre-RoPE, [d_model, d_model])

where the doc's ``W_Q / W_K`` are ``[d_model, d_head]`` (i.e. the transpose of
HF's ``q_proj.weight`` head slice).  The doc's SVD is

    M = U @ diag(sigma) @ V.T
    M' = U @ diag(sigma') @ V.T,  sigma'[j] = rho * sigma[j] if j in J
    delta_M = M' - M
    delta_W_Q_doc = delta_M @ pinv(W_K.T)

and ``delta_W_Q_doc`` (``[d_model, d_head]``) is folded back into the HF
``q_proj.weight`` slice as its transpose (``[d_head, d_model]``).

Because ``M = A @ B.T`` with ``A, B`` both ``[d_model, d_head]`` and typically
``d_head << d_model``, we never materialise a full dense ``d_model x d_model``
SVD: QR of ``A`` and ``B`` reduces the problem to a small ``d_head x d_head``
SVD, then reconstruct U = Q_A @ U_C and V = Q_B @ V_C.

All functions work on plain 2-D tensors and are model / Trainer / Hydra free so
they can be unit-tested in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Sequence, Tuple

import torch


@dataclass
class DirectionSpec:
    """A single selected (layer, head, singular-index) direction."""

    layer: int
    head: int
    rank: int
    sigma: float
    stage_a_score: float = 0.0
    stage_b_score: Optional[float] = None
    rho: float = 0.0
    causal_verified: bool = False
    retain_impact: Optional[float] = None
    meta: Dict[str, float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.meta is None:
            self.meta = {}

    def to_dict(self) -> Dict:
        return asdict(self)

    @property
    def key(self) -> Tuple[int, int, int]:
        return (self.layer, self.head, self.rank)


def get_head_slices(
    weight: torch.Tensor,
    head: int,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Return the 2-D ``[d_head, d_model]`` slice of one head in HF layout."""
    if weight.ndim != 2:
        raise ValueError(f"expected 2-D weight, got shape {tuple(weight.shape)}")
    start = head * head_dim
    end = start + head_dim
    if end > weight.shape[0]:
        raise ValueError(
            f"head {head} out of range: weight rows={weight.shape[0]}, "
            f"num_heads={num_heads}, head_dim={head_dim}"
        )
    return weight[start:end, :]


def get_kv_head_index(head: int, num_heads: int, num_kv_heads: int) -> int:
    """Map a query head index to its shared key/value head under GQA."""
    if num_kv_heads < 1 or num_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_heads ({num_heads}) must be a positive multiple of "
            f"num_kv_heads ({num_kv_heads})"
        )
    groups = num_heads // num_kv_heads
    return head // groups


def get_k_head_slices(
    weight: torch.Tensor,
    head: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Return the key head slice shared by query ``head`` (GQA-safe)."""
    kv_head = get_kv_head_index(head, num_heads, num_kv_heads)
    return get_head_slices(weight, kv_head, num_kv_heads, head_dim)


def build_qk_matrix(
    w_q_head: torch.Tensor,
    w_k_head: torch.Tensor,
) -> torch.Tensor:
    """Build the doc's ``[d_model, d_model]`` matrix ``M = W_Q @ W_K.T``.

    ``w_q_head / w_k_head`` are HF-layout ``[d_head, d_model]`` weight slices;
    the doc's matrices are their transposes, hence ``M = W_Q_head.T @ W_K_head``.
    """
    if w_q_head.ndim != 2 or w_k_head.ndim != 2:
        raise ValueError("w_q_head and w_k_head must be 2-D")
    if w_q_head.shape != w_k_head.shape:
        raise ValueError(
            f"W_Q/W_K head shape mismatch: {tuple(w_q_head.shape)} vs "
            f"{tuple(w_k_head.shape)}"
        )
    return w_q_head.T @ w_k_head


def svd_qk(
    M: torch.Tensor,
    rank_budget: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Economy SVD ``M = U @ diag(sigma) @ V.T`` computed in float32.

    ``M`` is ``[d_model, d_model]`` but has rank <= d_head.  We call
    ``torch.linalg.svd(..., full_matrices=False)``: since M is square this
    returns U, V of shape ``[d_model, d_model]``.  Callers that need cheap
    storage should use :func:`svd_qk_lowrank` instead.
    """
    M32 = M.to(torch.float32)
    U, sigma, Vh = torch.linalg.svd(M32, full_matrices=False)
    if rank_budget is not None:
        U, sigma, V = U[:, :rank_budget], sigma[:rank_budget], Vh.T[:, :rank_budget]
        return U, sigma, V
    return U, sigma, Vh.T


def svd_qk_lowrank(
    w_q_head: torch.Tensor,
    w_k_head: torch.Tensor,
    max_rank: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Low-rank SVD of ``M = W_Q.T @ W_K`` without dense d_model SVD.

    Uses QR: ``W_Q.T = Q_Q R_Q``, ``W_K.T = Q_K R_K`` (both Q are
    ``[d_model, d_head]``), then SVD of the small ``C = R_Q @ R_K.T``
    (``[d_head, d_head]``): ``C = U_C diag(sigma) V_C.T``.  Therefore

        U = Q_Q @ U_C  ,  V = Q_K @ V_C  ,  M = U diag(sigma) V.T

    Returns ``(U, sigma, V)`` with U, V of shape ``[d_model, r]`` where
    ``r = min(d_head, max_rank or d_head)``.
    """
    if w_q_head.shape != w_k_head.shape:
        raise ValueError("W_Q/W_K head shape mismatch")
    d_model = w_q_head.shape[1]
    d_head = w_q_head.shape[0]

    A = w_q_head.T.to(torch.float32)   # [d_model, d_head] = doc's W_Q
    B = w_k_head.T.to(torch.float32)   # [d_model, d_head] = doc's W_K

    Q_Q, R_Q = torch.linalg.qr(A, mode="reduced")   # [d_model, d_head] each
    Q_K, R_K = torch.linalg.qr(B, mode="reduced")
    C = R_Q @ R_K.T                                # [d_head, d_head]
    U_C, sigma, V_C_T = torch.linalg.svd(C, full_matrices=False)
    V_C = V_C_T.T

    U = Q_Q @ U_C                                  # [d_model, d_head]
    V = Q_K @ V_C                                  # [d_model, d_head]

    if max_rank is not None:
        r = min(max_rank, U.shape[1])
        U, sigma, V = U[:, :r], sigma[:r], V[:, :r]
    return U, sigma, V


def compute_delta_wq(
    U: torch.Tensor,
    sigma: torch.Tensor,
    V: torch.Tensor,
    selected_ranks: Sequence[int],
    rho: float,
    w_k_head: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute ``delta_M = M' - M`` and the doc-space ``delta_W_Q``.

    Parameters
    ----------
    U, sigma, V:
        Low-rank SVD of ``M`` with ``M = U diag(sigma) V.T``,
        U/V ``[d_model, r]``.
    selected_ranks:
        Singular indices in ``J`` (into ``sigma``).
    rho:
        Contraction coefficient ``0 <= rho < 1``.
    w_k_head:
        HF-layout key head weight ``[d_head, d_model]``; its transpose is the
        doc's ``W_K``.

    Returns
    -------
    delta_M:
        ``[d_model, d_model]`` residual ``M' - M`` (only ranks in J change).
    delta_W_Q_doc:
        ``[d_model, d_head]`` update to the doc's ``W_Q``; transpose before
        adding into the HF ``q_proj.weight`` head slice.
    """
    # rho == 1.0 is allowed for the no-op control (yields an exact zero
    # update); normal edits keep rho < 1 via QKSpecConfig validation.
    if not 0.0 <= rho <= 1.0:
        raise ValueError(f"rho must be in [0, 1], got {rho}")
    ranks = list(selected_ranks)
    if not ranks:
        zeros_m = torch.zeros(
            (U.shape[0], U.shape[0]), dtype=torch.float32, device=U.device
        )
        return zeros_m, torch.zeros(
            (U.shape[0], w_k_head.shape[0]), dtype=torch.float32, device=U.device
        )

    U_J = U[:, ranks]                       # [d_model, |J|]
    sigma_J = sigma[ranks].to(torch.float32)  # [|J|]
    V_J = V[:, ranks]                       # [d_model, |J|]
    scale = (rho - 1.0) * sigma_J           # <= 0
    delta_M = (U_J * scale.unsqueeze(0)) @ V_J.T            # [d_model, d_model]

    # doc's W_K = w_k_head.T  [d_model, d_head]; pinv(W_K.T) = pinv(w_k_head)
    # [d_model, d_head].
    pinv_w_k = torch.linalg.pinv(w_k_head.to(torch.float32))  # [d_model, d_head]
    delta_W_Q_doc = delta_M @ pinv_w_k                         # [d_model, d_head]
    return delta_M, delta_W_Q_doc


def fold_into_q(
    q_proj_weight: torch.Tensor,
    head: int,
    delta_W_Q_doc: torch.Tensor,
    num_heads: int,
    head_dim: int,
    inplace: bool = False,
) -> torch.Tensor:
    """Add ``delta_W_Q_doc.T`` into ``q_proj.weight`` head slice.

    ``q_proj.weight`` is ``[num_heads * head_dim, d_model]`` and
    ``delta_W_Q_doc`` is ``[d_model, head_dim]`` (doc convention).  Only the
    target head slice is modified; K, V, O and other heads stay byte-identical.
    """
    if q_proj_weight.ndim != 2 or delta_W_Q_doc.ndim != 2:
        raise ValueError("expected 2-D tensors")
    expected = (q_proj_weight.shape[1], head_dim)
    if tuple(delta_W_Q_doc.shape) != expected:
        raise ValueError(
            f"delta_W_Q_doc shape {tuple(delta_W_Q_doc.shape)} != {expected}"
        )
    out = q_proj_weight if inplace else q_proj_weight.clone()
    start = head * head_dim
    out[start : start + head_dim, :] += delta_W_Q_doc.T.to(out.dtype)
    return out


def spectral_logit_contribution(
    x_q: torch.Tensor,
    x_k: torch.Tensor,
    U: torch.Tensor,
    sigma: torch.Tensor,
    V: torch.Tensor,
    ranks: Sequence[int],
) -> torch.Tensor:
    """Per-direction logit ``sigma_j (x_q . u_j)(v_j . x_k)``.

    ``x_q, x_k`` are ``[d_model]`` hidden states feeding q_proj/k_proj;
    ``U, V`` are ``[d_model, r]``.  Returns ``[|ranks|]``.
    """
    if not ranks:
        return torch.zeros(0, dtype=torch.float32, device=U.device)
    ranks_t = torch.tensor(list(ranks), dtype=torch.long, device=U.device)
    U_J = U[:, ranks_t].to(torch.float32)   # [d_model, |J|]
    V_J = V[:, ranks_t].to(torch.float32)
    q_proj = U_J.T @ x_q.to(torch.float32)  # [|J|]
    k_proj = V_J.T @ x_k.to(torch.float32)  # [|J|]
    return sigma[ranks_t].to(torch.float32) * q_proj * k_proj


def qk_logit_from_hidden(
    x_q: torch.Tensor,
    x_k: torch.Tensor,
    w_q_head: torch.Tensor,
    w_k_head: torch.Tensor,
) -> torch.Tensor:
    """Scalar pre-RoPE logit ``x_q @ W_Q.T @ W_K @ x_k.T`` for diagnostics."""
    q = w_q_head.to(torch.float32).T @ x_q.to(torch.float32)  # [d_model]->[d_head]
    k = w_k_head.to(torch.float32).T @ x_k.to(torch.float32)
    return torch.dot(q, k)
