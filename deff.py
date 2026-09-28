"""
D_eff: effective depth of a transformer's residual stream.

Re-implementation of Definitions 1 and 2 and Proposition 3 of
arXiv:2609.31098v1, "The Residual Stream's Effective Depth"
(Gahtan, Galil, Bronstein; ACML 2026).

    rho_hat(k) = (1 / (L - k)) * sum_{l=1}^{L-k} CKA(H_l, H_{l+k})        (Def 1, Eq 4)
    D_eff      = L / (1 + 2 * sum_{k=1}^{L-1} (1 - k/L) * rho_hat(k))     (Def 2, Eq 5)
    F_L        = 2L / (L + 1)                                           (Prop 3)
    gap        = (F_L - D_eff) / F_L

CKA is the linear centred kernel alignment of Kornblith et al. (2019) on
column-centred inputs, Eq 2 of the paper:

    CKA(H, H') = ||H'^T H||_F^2 / (||H^T H||_F * ||H'^T H'||_F)

Protocol fidelity (paper Section 5, "Models and setup"):
  - mean-pooled hidden states over valid non-padding positions
  - the embedding layer (l = 0) is excluded
  - CKA in float64 with Frobenius normalisation
  - full autocorrelation, K = L - 1, no truncation
  - D_eff computed on CPU

The paper does not state a numerator convention for the pooling mask, so this
module takes explicit per-position weights and divides by their sum.  With
full-length packed sequences every weight is 1 and this reduces to a plain
mean.  Nothing is inferred or imputed: if the weights do not sum to the number
of valid positions the function raises.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import torch

__all__ = [
    "linear_cka",
    "layer_similarity_autocorr",
    "effective_depth",
    "reference_depth",
    "signed_gap",
    "d_eff_from_layers",
    "update_norm_ratio",
    "bartlett_weights",
]


def _as_f64(x: torch.Tensor) -> torch.Tensor:
    if not torch.is_floating_point(x):
        x = x.float()
    return x.to(torch.float64)


def center_columns(h: torch.Tensor) -> torch.Tensor:
    """Zero the per-column mean over the n samples (samples are rows)."""
    return h - h.mean(dim=0, keepdim=True)


def gram(h: torch.Tensor) -> torch.Tensor:
    """G = H^T H, the (d, d) layer Gram matrix, float64.

    Computed once per layer and reused across every lag pair: forming the
    self-Gram is O(n d^2) and forming a cross-Gram is another O(n d^2), so the
    whole estimator is O(L^2 n d^2) either way but the self-Grams are paid once.
    """
    return h.t() @ h


def cka_from_grams(g: torch.Tensor, g2: torch.Tensor, cross: torch.Tensor) -> float:
    """Linear CKA (Kornblith et al. 2019) from two self-Grams and a cross-Gram.

    Eq 2 of the paper:

        CKA(H, H') = ||H'^T H||_F^2 / (||H^T H||_F * ||H'^T H'||_F)

    where `g` = H^T H, `g2` = H'^T H' and `cross` = H^T H'.  The numerator is
    the squared Frobenius norm of the cross-Gram, NOT a product of the two
    self-Grams: ||H'^T H||_F^2 is not ||G G'||_F^2.

    ||H^T H||_F is sqrt(sum G^2), i.e. sqrt(sum s_k^4) in terms of H's singular
    values.  It equals ||H||_F^2 only at rank 1, so the Gram must be formed
    explicitly; the shortcut is wrong by about a factor of sqrt(d).
    """
    num = (cross * cross).sum()
    n1 = (g * g).sum().sqrt()
    n2 = (g2 * g2).sum().sqrt()
    if float(n1) <= 0 or float(n2) <= 0:
        raise ValueError("CKA is undefined for a constant (zero-centred) representation")
    return float(num / (n1 * n2))


def linear_cka(h: torch.Tensor, h2: torch.Tensor) -> float:
    """Linear CKA between two column-centred float64 (n, d) matrices."""
    if h.shape != h2.shape:
        raise ValueError(f"shape mismatch: {tuple(h.shape)} vs {tuple(h2.shape)}")
    g, g2, cross = gram(h), gram(h2), h.t() @ h2
    return cka_from_grams(g, g2, cross)


def bartlett_weights(L: int) -> Dict[int, float]:
    """w_k = 1 - k/L for k = 1 .. L-1 (paper Section 4, Eq 5)."""
    return {k: 1.0 - k / L for k in range(1, L)}


def layer_similarity_autocorr(
    layers: Sequence[torch.Tensor],
    keep_matrix: bool = False,
) -> Dict[int, float]:
    """Definition 1, Eq 4: the layer similarity autocorrelation at lag k.

    `layers` must already exclude the embedding layer, i.e. layers[0] is H_1.
    Each element is a (n, d) matrix of pooled representations, float64,
    column-centred.
    """
    L = len(layers)
    if L < 2:
        raise ValueError("need at least two layers to form a lag-1 autocorrelation")
    grams = [gram(h) for h in layers]
    out: Dict[int, float] = {}
    for k in range(1, L):
        acc = 0.0
        for ell in range(L - k):
            acc += cka_from_grams(grams[ell], grams[ell + k], layers[ell].t() @ layers[ell + k])
        out[k] = acc / (L - k)
    if keep_matrix:
        mat = torch.zeros(L, L, dtype=torch.float64)
        for i in range(L):
            mat[i, i] = 1.0
        for k, v in out.items():
            for i in range(L - k):
                mat[i, i + k] = v
                mat[i + k, i] = v
        out["matrix"] = mat  # type: ignore[index]
    return out


def effective_depth(rho: Dict[int, float], L: int) -> float:
    """Definition 2, Eq 5: the Bartlett-aggregated effective depth."""
    denom = 1.0
    for k in range(1, L):
        w = 1.0 - k / L
        if w < 0:
            raise ValueError(f"bartlett weight negative at k={k}, L={L}")
        denom += 2.0 * w * float(rho[k])
    if denom <= 0:
        raise ValueError(f"denominator is {denom}; D_eff undefined (paper bounds it to [1, L])")
    return L / denom


def reference_depth(L: int) -> float:
    """Proposition 3: F_L = 2L / (L + 1)."""
    return 2.0 * L / (L + 1.0)


def signed_gap(d_eff: float, L: int) -> float:
    """gap = (F_L - D_eff) / F_L, positive when states are more similar than reference."""
    f = reference_depth(L)
    return (f - d_eff) / f


def d_eff_from_layers(
    pooled: Sequence[torch.Tensor],
    L: Optional[int] = None,
    keep_matrix: bool = False,
) -> Dict[str, object]:
    """Full protocol: centre, CKA in float64, Bartlett, full K = L - 1."""
    layers = [center_columns(_as_f64(h)) for h in pooled]
    n_layers = len(layers) if L is None else L
    rho = layer_similarity_autocorr(layers, keep_matrix=keep_matrix)
    d = effective_depth(rho, n_layers)
    mat = rho.pop("matrix", None)  # type: ignore[call-overload]
    return {
        "D_eff": d,
        "D_eff_over_L": d / n_layers,
        "F_L": reference_depth(n_layers),
        "gap": signed_gap(d, n_layers),
        "rho": {int(k): float(v) for k, v in rho.items()},
        "rho_lag1": float(rho[1]),
        "L": n_layers,
        "n_passages": int(layers[0].shape[0]),
        "d": int(layers[0].shape[1]),
        "cka_matrix": mat,
    }


def update_norm_ratio(pooled: Sequence[torch.Tensor]) -> float:
    """||f_l|| / ||h_l|| for f_l = h_l - h_{l-1}, averaged over layers and samples.

    Reported by the paper in Table S13 and Table S14.  The embedding layer is
    excluded, so the first difference is taken between H_1 and H_0 only if H_0
    is supplied; with embedding excluded we start the difference at H_2 - H_1.
    """
    if len(pooled) < 2:
        raise ValueError("need at least two layers")
    ratios: List[float] = []
    for i in range(1, len(pooled)):
        f = pooled[i] - pooled[i - 1]
        num = float(torch.linalg.vector_norm(f, dim=1).mean())
        den = float(torch.linalg.vector_norm(pooled[i], dim=1).mean())
        if den > 0:
            ratios.append(num / den)
    return sum(ratios) / len(ratios)


def synthetic_orthogonal_check(L: int, n: int = 512, d: int = 64, seed: int = 0) -> float:
    """Proposition 3 sanity check on synthetic orthogonal per-layer updates.

    Builds h_l = sum_{j<=l} f_j with zero cross-covariance f_j, then evaluates
    the same estimator.  The paper's proof says this must land on 2L/(L+1).
    """
    g = torch.Generator().manual_seed(seed)
    f = [torch.randn(n, d, generator=g, dtype=torch.float64) for _ in range(L)]
    pooled = []
    acc = torch.zeros(n, d, dtype=torch.float64)
    for i in range(L):
        acc = acc + f[i]
        pooled.append(acc.clone())
    return float(d_eff_from_layers(pooled, L=L)["D_eff"])


def _unused(*_args, **_kwargs) -> None:  # pragma: no cover
    _ = math
