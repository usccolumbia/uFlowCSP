"""Spacegroup-aware symmetry projection utilities for MeanFlow CSP.

Provides three projection operators that can be plugged into the MeanFlow
training / sampling pipeline behind feature flags:

  1. ``project_lattice_to_spacegroup`` (stage 1)
       Symmetrise the per-sample lattice metric tensor over the spacegroup's
       point-group operations, then rebuild the lattice via Cholesky and
       broadcast back to all atom tokens. Inspired by CrystalFlow's
       ``LatticeDecompNN.proj_k_to_spacegroup`` but adapted to the raw
       (3,3) flattened lattice representation used in ``meanflow_raw``.

  2. ``project_coords_rotavg`` (stage 2)
       Rotationally-average the predicted (cartesian) coordinates / velocity
       over the spacegroup's symmetry operations, working in fractional space
       under the current per-sample lattice. CrystalFlow's
       ``SymmetrizeRotavg.symmetrize_rank1_scaled`` analogue.

  3. ``project_coords_anchor`` (stage 3)
       Wyckoff-anchor symmetrisation. Requires per-structure anchor / op data
       preprocessed at dataset load time — currently raises
       ``NotImplementedError`` so the feature flag can be plumbed without a
       silent fallback.

All three operators:
  - Accept the **null spacegroup (index 0)** and return the input unchanged.
  - Are differentiable (no in-place ops) and JVP-compatible.
  - Cache spacegroup → operation tensors lazily; first call per SG hits
    pymatgen and caches the resulting torch tensors on CPU. ``.to(device)``
    is handled at call time.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import numpy as np
import torch

try:
    from pymatgen.symmetry.groups import SpaceGroup
    _PYMATGEN_AVAILABLE = True
except ImportError:  # pragma: no cover - pymatgen is a hard dep elsewhere
    SpaceGroup = None
    _PYMATGEN_AVAILABLE = False


__all__ = [
    "project_lattice_to_spacegroup",
    "project_coords_rotavg",
    "project_coords_anchor",
]


# ---------------------------------------------------------------------------
# Spacegroup operation cache
# ---------------------------------------------------------------------------

@lru_cache(maxsize=256)
def _spacegroup_rotations_np(sg_number: int) -> np.ndarray:
    """Return all rotation parts (fractional basis) of SG ``sg_number``.

    Shape: ``(n_ops, 3, 3)``.  ``sg_number == 0`` (null class used for CFG
    dropout) returns a single identity, making the symmetrisation a no-op.
    """
    if sg_number == 0:
        return np.eye(3, dtype=np.float64)[None, :, :]

    if not _PYMATGEN_AVAILABLE:
        raise ImportError(
            "pymatgen is required for spacegroup symmetry projection."
        )

    if not (1 <= sg_number <= 230):
        raise ValueError(f"Spacegroup must be in [0, 230]; got {sg_number}")

    sg = SpaceGroup.from_int_number(int(sg_number))
    rots = np.stack(
        [np.asarray(op.rotation_matrix, dtype=np.float64) for op in sg.symmetry_ops],
        axis=0,
    )
    return rots  # (n_ops, 3, 3)


@lru_cache(maxsize=256)
def _spacegroup_symops_np(sg_number: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (rotations, translations) in fractional basis for SG.

    Shapes: rotations ``(n_ops, 3, 3)``, translations ``(n_ops, 3)``.
    ``sg_number == 0`` returns identity + zero translation.
    """
    if sg_number == 0:
        return (
            np.eye(3, dtype=np.float64)[None, :, :],
            np.zeros((1, 3), dtype=np.float64),
        )

    if not _PYMATGEN_AVAILABLE:
        raise ImportError(
            "pymatgen is required for spacegroup symmetry projection."
        )

    if not (1 <= sg_number <= 230):
        raise ValueError(f"Spacegroup must be in [0, 230]; got {sg_number}")

    sg = SpaceGroup.from_int_number(int(sg_number))
    rots = np.stack(
        [np.asarray(op.rotation_matrix, dtype=np.float64) for op in sg.symmetry_ops],
        axis=0,
    )
    trans = np.stack(
        [np.asarray(op.translation_vector, dtype=np.float64) for op in sg.symmetry_ops],
        axis=0,
    )
    return rots, trans


def _rotations_torch(sg_number: int, device, dtype) -> torch.Tensor:
    """Cached spacegroup rotations as a torch tensor on the requested device."""
    rots_np = _spacegroup_rotations_np(int(sg_number))
    return torch.from_numpy(rots_np).to(device=device, dtype=dtype)


def _symops_torch(sg_number: int, device, dtype):
    """Cached spacegroup (rotations, translations) on the requested device."""
    rots_np, trans_np = _spacegroup_symops_np(int(sg_number))
    return (
        torch.from_numpy(rots_np).to(device=device, dtype=dtype),
        torch.from_numpy(trans_np).to(device=device, dtype=dtype),
    )


# ---------------------------------------------------------------------------
# Helpers: lattice reconstruction
# ---------------------------------------------------------------------------

def _lattice_per_sample(x_lattice_BN9: torch.Tensor,
                        mask: torch.Tensor) -> torch.Tensor:
    """Average the per-token lattice (B, N, 9) to a single (B, 3, 3) per sample.

    During the flow path the lattice channels of ``x`` are noised independently
    per token, so different tokens carry slightly different lattice guesses.
    We take the masked mean as the sample's lattice estimate.
    """
    B, N, _ = x_lattice_BN9.shape
    m = mask.to(x_lattice_BN9.dtype).unsqueeze(-1)            # (B, N, 1)
    denom = m.sum(dim=1).clamp(min=1.0)                       # (B, 1)
    L_mean = (x_lattice_BN9 * m).sum(dim=1) / denom            # (B, 9)
    return L_mean.view(B, 3, 3)


def _broadcast_lattice(L_B33: torch.Tensor, N: int) -> torch.Tensor:
    """Broadcast a per-sample (B, 3, 3) lattice back to per-token (B, N, 9)."""
    return L_B33.reshape(L_B33.shape[0], 1, 9).expand(-1, N, -1).contiguous()


# ---------------------------------------------------------------------------
# Stage 1 — lattice metric-tensor projection
# ---------------------------------------------------------------------------

def project_lattice_to_spacegroup(
    x: torch.Tensor,                # (B, N, d)  d >= coord_dim + lattice_dim
    spacegroup: torch.Tensor,       # (B,) long, 0 = null class (no-op)
    mask: torch.Tensor,             # (B, N) bool
    coord_dim: int = 3,
    lattice_dim: int = 9,
) -> torch.Tensor:
    """Project the lattice channels of ``x`` onto the SG-allowed subspace.

    The metric tensor G = L^T L must satisfy ``R^T G R = G`` for every point
    group operation R of the spacegroup. We enforce this by averaging the
    sample's metric tensor over all rotations:

        G_proj = (1 / |G|)  Σ  R_g^T  G  R_g
                            g

    The lattice is then rebuilt from ``G_proj`` via Cholesky (upper
    triangular convention used by pymatgen / pyxtal) and broadcast back to
    all tokens. Result is differentiable and JVP-safe.

    Samples with ``spacegroup == 0`` (CFG null class) are passed through
    unchanged.
    """
    B, N, d = x.shape
    assert d >= coord_dim + lattice_dim, (
        f"x last dim ({d}) is smaller than coord_dim + lattice_dim "
        f"({coord_dim + lattice_dim})"
    )
    assert lattice_dim == 9, "Only (3,3) lattice (lattice_dim=9) is supported."

    orig_dtype = x.dtype
    # bf16-mixed autocast silently recasts matmul inputs to bf16 even when
    # the tensors are float32.  linalg.eigh / cholesky / inv are only
    # implemented for float32 on CUDA, so we must disable autocast entirely
    # for this block.
    with torch.amp.autocast(device_type="cuda", enabled=False):
        x_f32 = x.float()

        coords = x_f32[..., :coord_dim]
        lat_BN9 = x_f32[..., coord_dim:coord_dim + lattice_dim]
        extras = x_f32[..., coord_dim + lattice_dim:]

        L_B33 = _lattice_per_sample(lat_BN9, mask)  # (B, 3, 3)
        G_B33 = L_B33.transpose(-1, -2) @ L_B33      # (B, 3, 3) metric tensor

        # Process each sample (different SGs ⇒ different op sets).  Group by SG
        # to avoid 230 launches per batch.
        # Accept a pre-computed list (needed inside torch.func.jvp where dual
        # tensors have no storage) or a regular tensor.
        if isinstance(spacegroup, torch.Tensor):
            sg_list = spacegroup.detach().cpu().tolist()
        else:
            sg_list = [int(s) for s in spacegroup]
        G_proj = G_B33.clone()  # start from input; only valid-SG rows are projected
        by_sg: dict = {}
        for i, sg in enumerate(sg_list):
            sg_i = int(sg)
            if sg_i == 0:
                continue  # null SG (CFG dropout) — leave the row untouched
            by_sg.setdefault(sg_i, []).append(i)

        for sg, idxs in by_sg.items():
            idx_t = torch.as_tensor(idxs, dtype=torch.long, device=G_B33.device)
            R = _rotations_torch(sg, device=G_B33.device, dtype=torch.float32)  # always f32
            Gi = G_B33[idx_t]                                                   # (k, 3, 3)
            Gi_avg = torch.einsum("gba,kbc,gcd->kad", R, Gi, R) / R.shape[0]
            Gi_avg = 0.5 * (Gi_avg + Gi_avg.transpose(-1, -2))
            G_proj[idx_t] = Gi_avg

        # Rebuild lattice via eigendecomposition.
        G_proj = 0.5 * (G_proj + G_proj.transpose(-1, -2))
        eigvals, eigvecs = torch.linalg.eigh(G_proj)   # float32 — works on CUDA
        trace = G_proj.diagonal(dim1=-2, dim2=-1).sum(dim=-1).clamp(min=1e-6)
        eigvals = eigvals.clamp_min((1e-6 * trace).unsqueeze(-1))
        G_pd = eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-1, -2)
        L_lower = torch.linalg.cholesky(G_pd)          # guaranteed PD
        L_up = L_lower.transpose(-1, -2)

        lat_proj_BN9 = _broadcast_lattice(L_up, N)
        result = torch.cat([coords, lat_proj_BN9, extras], dim=-1)

    # Cast back to original dtype (e.g. bf16) after leaving autocast-free zone.
    return result.to(orig_dtype)


# ---------------------------------------------------------------------------
# Stage 2 — coordinate rotational averaging
# ---------------------------------------------------------------------------

def project_coords_rotavg(
    x: torch.Tensor,                # (B, N, d)
    spacegroup: torch.Tensor,       # (B,) long
    mask: torch.Tensor,             # (B, N) bool
    coord_dim: int = 3,
    lattice_dim: int = 9,
    coords_are_fractional: bool = False,
    lattice_source: torch.Tensor = None,   # (B, N, d) or (B, 3, 3); see note
) -> torch.Tensor:
    """Rotationally-average the coordinate channels of ``x`` over the SG ops.

    For each sample, average the predicted coordinates (or velocity) over
    the spacegroup's rotation operations expressed in **fractional** basis.
    If the coordinate channels are Cartesian (default for ``meanflow_raw``),
    they are first converted to fractional via the sample's lattice
    (averaged across tokens), averaged in fractional space, then converted
    back to Cartesian.

    Translations are **not** added — this operator symmetrises a *vector
    field* (a velocity), not a coordinate field. Position-level translations
    must be applied separately if symmetrising actual positions.

    ``lattice_source``: when projecting a **velocity** field, ``x`` itself
    carries a velocity-of-lattice in its lattice channels, not a lattice;
    ``pinv`` of that is numerical garbage and was the root cause of the
    RA-mode training collapse.  Pass the current structure-shaped tensor
    (``z_t``) here so the cart↔frac conversion uses its lattice instead.
    Accepts either a full ``(B, N, d)`` token tensor (lattice extracted via
    the same broadcast scheme) or a pre-built ``(B, 3, 3)`` matrix.  When
    ``None``, falls back to using ``x``'s own lattice channels (correct for
    structure-shaped inputs).

    Samples with ``spacegroup == 0`` (null) are passed through unchanged.
    """
    orig_dtype = x.dtype
    # bf16-mixed autocast will silently downcast matmul inputs; linalg.inv
    # is not implemented for bf16 on CUDA.  Disable autocast for this block.
    with torch.amp.autocast(device_type="cuda", enabled=False):
        x_f32 = x.float()

        B, N, d = x_f32.shape
        coords = x_f32[..., :coord_dim]
        lat_BN9 = x_f32[..., coord_dim:coord_dim + lattice_dim]
        extras = x_f32[..., coord_dim + lattice_dim:]

        if lattice_source is None:
            L_B33 = _lattice_per_sample(lat_BN9, mask)               # (B, 3, 3)
        elif lattice_source.dim() == 3 and lattice_source.shape[-2:] == (3, 3):
            L_B33 = lattice_source.float()                            # already (B, 3, 3)
        else:
            src_lat_BN9 = lattice_source[..., coord_dim:coord_dim + lattice_dim].float()
            L_B33 = _lattice_per_sample(src_lat_BN9, mask)            # (B, 3, 3)

        if coords_are_fractional:
            frac = coords
        else:
            # At intermediate flow times the lattice tokens may be noisy /
            # near-zero, making L_B33 singular.  pinv handles rank-deficient
            # matrices gracefully (no crash); for well-conditioned matrices it
            # gives the same result as inv.
            L_inv = torch.linalg.pinv(L_B33)                             # (B, 3, 3)
            frac = torch.einsum("bni,bij->bnj", coords, L_inv)

        # Accept a pre-computed list or a regular tensor (see stage-1 comment).
        if isinstance(spacegroup, torch.Tensor):
            sg_list = spacegroup.detach().cpu().tolist()
        else:
            sg_list = [int(s) for s in spacegroup]
        frac_proj = torch.empty_like(frac)
        by_sg: dict = {}
        for i, sg in enumerate(sg_list):
            by_sg.setdefault(int(sg), []).append(i)

        for sg, idxs in by_sg.items():
            idx_t = torch.as_tensor(idxs, dtype=torch.long, device=frac.device)
            if sg == 0:
                frac_proj[idx_t] = frac[idx_t]
                continue
            R = _rotations_torch(sg, device=frac.device, dtype=torch.float32)  # always f32
            v_g = torch.einsum("knj,gij->gkni", frac[idx_t], R)               # (n_ops, k, N, 3)
            frac_proj[idx_t] = v_g.mean(dim=0)

        if coords_are_fractional:
            coords_proj = frac_proj
        else:
            coords_proj = torch.einsum("bni,bij->bnj", frac_proj, L_B33)

        result = torch.cat([coords_proj, lat_BN9, extras], dim=-1)

    # Cast back to original dtype (e.g. bf16) after leaving autocast-free zone.
    return result.to(orig_dtype)


# ---------------------------------------------------------------------------
# Stage 3 — Wyckoff anchor symmetrisation
# ---------------------------------------------------------------------------

def project_coords_anchor(
    x: torch.Tensor,
    spacegroup: torch.Tensor,
    mask: torch.Tensor,
    coord_dim: int = 3,
    lattice_dim: int = 9,
    anchor_index: torch.Tensor = None,
    ops: torch.Tensor = None,
    ops_inv: torch.Tensor = None,
) -> torch.Tensor:
    """Wyckoff-anchor symmetrisation (stage 3).

    Selects one atom per Wyckoff orbit as the "anchor", then expands its
    coordinates over the orbit via the precomputed per-structure symmetry
    operations (``ops`` / ``ops_inv``).  Guarantees the resulting positions
    obey the spacegroup exactly.

    **Not yet implemented.** Requires per-structure Wyckoff data
    (``anchor_index``, ``ops``, ``ops_inv``) to be precomputed at dataset
    load time, which is a substantial change to ``MP20RawData`` /
    ``AlexRawData``. The flag is plumbed end-to-end so this can be filled
    in without a config or CLI change.
    """
    raise NotImplementedError(
        "project_coords_anchor (stage 3) requires per-structure Wyckoff "
        "preprocessing (anchor_index, ops, ops_inv) which is not yet wired "
        "into the raw datasets. Run with proj_coords_anchor=false until "
        "Wyckoff preprocessing lands."
    )
