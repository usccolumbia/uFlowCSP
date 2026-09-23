"""Polar (log-Euclidean) lattice representation utilities.

Ported from CrystalFlow (``diffcsp/common/data_utils.py``) to convert between a
3x3 lattice matrix ``L`` and a 6-dimensional representation ``k``.

The 6-D representation is obtained from the symmetric polar factor of ``L``:

    J = L @ L^T                     (symmetric positive-definite, 3x3)
    S = 1/2 * U @ diag(log A) @ U^T (matrix log of the polar stretch)
    k = (S01, S02, S12,
         (S00 - S11) / 2,
         (S00 + S11 - 2 S22) / 6,
         (S00 + S11 + S22) / 3)

The inverse rebuilds the symmetric matrix ``S`` from ``k`` and exponentiates it.

This representation is rotation-invariant and unconstrained (any vector in R^6
maps to a valid lattice), which makes it well suited as a flow-matching target.

References
----------
X. Luo et al., "CrystalFlow: A Flow-Based Generative Model for Crystalline
Materials", arXiv:2412.11693.
"""

from __future__ import annotations

import torch


def decompose_symmetric_matrix(S: torch.Tensor) -> torch.Tensor:
    """Extract the 6-D vector ``k`` from a batch of symmetric matrices.

    Args:
        S: ``(B, 3, 3)`` symmetric matrices.

    Returns:
        ``(B, 6)`` representation ``k``.
    """
    k0 = S[:, 0, 1]
    k1 = S[:, 0, 2]
    k2 = S[:, 1, 2]
    k3 = (S[:, 0, 0] - S[:, 1, 1]) / 2
    k4 = (S[:, 0, 0] + S[:, 1, 1] - 2 * S[:, 2, 2]) / 6
    k5 = (S[:, 0, 0] + S[:, 1, 1] + S[:, 2, 2]) / 3
    return torch.stack([k0, k1, k2, k3, k4, k5], dim=-1)


@torch.no_grad()
def lattice_polar_decompose_torch(lattices: torch.Tensor) -> torch.Tensor:
    """Convert batched 3x3 lattice matrices to the 6-D polar representation.

    Args:
        lattices: ``(B, 3, 3)`` lattice matrices (rows are lattice vectors).

    Returns:
        ``(B, 6)`` polar representation ``k``.
    """
    assert lattices.dim() == 3, "input must be batched lattices of shape (B, 3, 3)"
    # torch.linalg.eigh does not support BFloat16 on CUDA. Disable autocast and
    # upcast to float32 so the matmul result is also float32 before the eigh call.
    orig_dtype = lattices.dtype
    device_type = "cuda" if lattices.is_cuda else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        lattices = lattices.float()
        # J = L @ L^T is symmetric positive-definite.
        A, U = torch.linalg.eigh(lattices @ lattices.transpose(-1, -2))
        # S = 1/2 * U @ log(diag(A)) @ U^T
        A = torch.diag_embed(A.clamp_min(1e-12).log()) / 2
        S = U @ A @ U.transpose(-1, -2)
    return decompose_symmetric_matrix(S).to(orig_dtype)


def lattice_polar_build_torch(k: torch.Tensor) -> torch.Tensor:
    """Rebuild batched 3x3 lattice matrices from the 6-D polar representation.

    This is differentiable (uses ``torch.matrix_exp``); it is normally only
    needed at sampling/reconstruction time, but is left grad-enabled so it can
    also be used inside an autograd graph if required.

    Args:
        k: ``(B, 6)`` polar representation.

    Returns:
        ``(B, 3, 3)`` lattice matrices.
    """
    assert k.dim() == 2, "input must be batched k of shape (B, 6)"
    s0 = torch.stack([k[:, 3] + k[:, 4] + k[:, 5], k[:, 0], k[:, 1]], dim=1)
    s1 = torch.stack([k[:, 0], -k[:, 3] + k[:, 4] + k[:, 5], k[:, 2]], dim=1)
    s2 = torch.stack([k[:, 1], k[:, 2], -2 * k[:, 4] + k[:, 5]], dim=1)
    S = torch.stack([s0, s1, s2], dim=1)  # (B, 3, 3) symmetric
    return torch.matrix_exp(S)
