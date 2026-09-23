"""PyG transform applying optional symmetry-preserving atom-order augmentation.

This transform runs at data-loading time (per sample, per epoch). It assumes the
underlying ``Data`` object was already stored in **canonical** atom order at
dataset-processing time, with the per-atom ordering labels ``orbit_id``,
``wyckoff_rank`` and ``en_value`` attached (see ``atom_ordering.py``).

If those labels are missing (e.g. an old cache), the transform is a safe
**identity** passthrough so existing pipelines keep working.

Two independent, default-off augmentations:
    * ``perm_augment``        — hierarchical inter-/intra-orbit permutation.
    * ``modulo_translation``  — random global fractional translation (mod 1).

When *both* are disabled the transform short-circuits to identity (the canonical
order baked into the cache is already what the model should see).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from src.data.components.atom_ordering import (
    canonical_order,
    compute_group_labels,
    global_modulo_translation,
    hierarchical_permutation,
)

# Per-atom tensor fields that must be reordered together when atoms are permuted.
_PER_ATOM_FIELDS = (
    "atom_types",
    "pos",
    "frac_coords",
    "lattice",
    "orbit_id",
    "wyckoff_rank",
    "en_value",
)


def apply_canonical_ordering(data, mode: str = "symmetry", symprec: float = 0.1):
    """Compute ordering labels, attach them, and canonically reorder a ``Data`` object.

    Run **once** at dataset-processing time. Stores per-atom ``orbit_id``,
    ``wyckoff_rank`` and ``en_value`` and reorders all per-atom fields into the
    deterministic canonical order. Returns the same ``Data`` instance.
    """
    n = int(data.atom_types.shape[0])
    if n == 0:
        data.orbit_id = torch.zeros(0, dtype=torch.long)
        data.wyckoff_rank = torch.zeros(0, dtype=torch.long)
        data.en_value = torch.zeros(0, dtype=torch.float)
        return data

    atom_types_np = data.atom_types.cpu().numpy()
    frac_np = data.frac_coords.cpu().numpy()
    cell_np = data.cell.reshape(3, 3).cpu().numpy()

    en_value, orbit_id, wyckoff_rank = compute_group_labels(
        atom_types=atom_types_np,
        frac_coords=frac_np,
        cell=cell_np,
        mode=mode,
        symprec=symprec,
    )
    perm = canonical_order(en_value, orbit_id, wyckoff_rank, frac_np)
    perm_t = torch.as_tensor(perm, dtype=torch.long)

    # Attach labels (in canonical order).
    data.orbit_id = torch.as_tensor(orbit_id[perm], dtype=torch.long)
    data.wyckoff_rank = torch.as_tensor(wyckoff_rank[perm], dtype=torch.long)
    data.en_value = torch.as_tensor(en_value[perm], dtype=torch.float)

    # Reorder all per-atom tensors consistently.
    for field in ("atom_types", "pos", "frac_coords", "lattice"):
        val = getattr(data, field, None)
        if val is not None and val.shape[0] == n:
            setattr(data, field, val[perm_t])
    if getattr(data, "token_idx", None) is not None:
        data.token_idx = torch.arange(n, dtype=data.token_idx.dtype)

    return data


class AtomOrderingTransform:
    """Apply optional symmetry-preserving permutation / translation augmentation.

    Args:
        perm_augment: enable hierarchical inter-/intra-orbit permutation.
        modulo_translation: enable random global fractional translation.
        seed: optional base seed for reproducibility (tests). If ``None`` the
            augmentation is non-deterministic across epochs (recommended for
            training).
    """

    def __init__(
        self,
        perm_augment: bool = False,
        modulo_translation: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.perm_augment = bool(perm_augment)
        self.modulo_translation = bool(modulo_translation)
        self.seed = seed

    def _rng(self, data) -> np.random.Generator:
        if self.seed is None:
            return np.random.default_rng()
        # Deterministic per-sample stream when a seed is provided.
        idx = 0
        struct_idx = getattr(data, "structure_idx", None)
        if struct_idx is not None:
            try:
                idx = int(struct_idx.reshape(-1)[0].item())
            except Exception:
                idx = 0
        return np.random.default_rng(self.seed + idx)

    def __call__(self, data):
        # Identity when no augmentation requested.
        if not self.perm_augment and not self.modulo_translation:
            return data

        # Safety: require ordering labels; otherwise pass through unchanged.
        if any(getattr(data, f, None) is None for f in ("orbit_id", "wyckoff_rank")):
            return data

        rng = self._rng(data)
        n = int(data.atom_types.shape[0])
        if n <= 1:
            return data

        if self.perm_augment:
            perm = hierarchical_permutation(
                atom_types=data.atom_types.cpu().numpy(),
                orbit_id=data.orbit_id.cpu().numpy(),
                wyckoff_rank=data.wyckoff_rank.cpu().numpy(),
                rng=rng,
            )
            perm_t = torch.as_tensor(perm, dtype=torch.long)
            for field in _PER_ATOM_FIELDS:
                val = getattr(data, field, None)
                if val is not None and val.shape[0] == n:
                    setattr(data, field, val[perm_t])
            # Keep token_idx as positional indices in the new sequence.
            if getattr(data, "token_idx", None) is not None:
                data.token_idx = torch.arange(n, dtype=data.token_idx.dtype)

        if self.modulo_translation:
            cell = data.cell  # (1, 3, 3)
            new_frac = global_modulo_translation(data.frac_coords.cpu().numpy(), rng)
            new_frac_t = torch.as_tensor(new_frac, dtype=data.frac_coords.dtype)
            data.frac_coords = new_frac_t
            cell_mat = cell.reshape(3, 3).to(new_frac_t.dtype)
            data.pos = (new_frac_t @ cell_mat).to(data.pos.dtype)

        return data
