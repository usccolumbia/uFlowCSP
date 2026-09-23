"""Composition- and symmetry-aware atom ordering utilities (MCFlow-style).

This module provides *optional* helpers to reorder the atoms of a crystal into a
canonical, symmetry-aware sequence before they are fed to the order-sensitive
transformer denoiser, plus hierarchical permutation augmentation that preserves
compositional / crystallographic equivalence.

Reference: "Multimodal Crystal Flow" (MCFlow), arXiv:2602.20210.

Design notes
------------
* Everything here is computed **once at dataset-processing time** (the heavy
  pymatgen / spglib symmetry analysis), and the per-atom labels
  (``en_value``, ``orbit_id``, ``wyckoff_rank``) are cached on the ``Data``
  object.  The per-epoch augmentation transform then only touches cheap integer
  tensors and never calls pymatgen.
* Two modes:
    - ``"simple"``    : sort by (electronegativity, fractional coords). No
      symmetry analysis. ``orbit_id = arange(N)``, ``wyckoff_rank = 0``.
    - ``"symmetry"``  : group atoms into space-group orbits (spglib
      ``equivalent_atoms``) and sort by (electronegativity, Wyckoff letter,
      orbit, fractional coords). Falls back to ``"simple"`` if symmetry
      analysis fails for a given structure.
* None of this changes the model. It only changes the *order* of atoms.
"""

from __future__ import annotations

import string
from typing import Tuple

import numpy as np

# Sentinel electronegativity for elements without a Pauling value (e.g. noble
# gases) so they sort deterministically *after* well-defined elements.
_EN_SENTINEL = 100.0

# Wyckoff letters are conventionally a..z then (rarely) capital letters; the
# general position uses a letter too. ``string.ascii_letters`` == 'a'..'z''A'..'Z'.
_WYCKOFF_RANKS = {ch: i for i, ch in enumerate(string.ascii_letters)}
_WYCKOFF_FALLBACK_RANK = 99


def _build_en_table() -> dict:
    """Map atomic number -> Pauling electronegativity (sentinel if undefined)."""
    table: dict = {}
    try:
        from pymatgen.core.periodic_table import Element
    except Exception:  # pragma: no cover - pymatgen always present in this repo
        return table
    for z in range(1, 119):
        try:
            x = Element.from_Z(z).X  # Pauling electronegativity (may be nan)
        except Exception:
            x = None
        if x is None or (isinstance(x, float) and np.isnan(x)):
            # Keep undefined-EN elements deterministic and after defined ones.
            table[z] = _EN_SENTINEL + z * 1e-3
        else:
            table[z] = float(x)
    return table


# Built once at import.
_EN_TABLE = _build_en_table()


def electronegativity_values(atom_types: np.ndarray) -> np.ndarray:
    """Per-atom Pauling electronegativity (sentinel for undefined elements)."""
    atom_types = np.asarray(atom_types).astype(int).reshape(-1)
    return np.array(
        [_EN_TABLE.get(int(z), _EN_SENTINEL + int(z) * 1e-3) for z in atom_types],
        dtype=np.float64,
    )


def _wyckoff_rank(letter: str) -> int:
    if not isinstance(letter, str) or len(letter) == 0:
        return _WYCKOFF_FALLBACK_RANK
    return _WYCKOFF_RANKS.get(letter[0], _WYCKOFF_FALLBACK_RANK)


def compute_group_labels(
    atom_types: np.ndarray,
    frac_coords: np.ndarray,
    cell: np.ndarray,
    mode: str = "symmetry",
    symprec: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-atom ordering labels.

    Args:
        atom_types: (N,) atomic numbers.
        frac_coords: (N, 3) fractional coordinates.
        cell: (3, 3) lattice matrix (rows are lattice vectors).
        mode: ``"simple"`` or ``"symmetry"``.
        symprec: symmetry tolerance for spglib (used in symmetry mode).

    Returns:
        en_value:     (N,) float electronegativity sort key.
        orbit_id:     (N,) int, atoms sharing a value are symmetry-equivalent
                      (intra-orbit interchangeable). In simple mode this is
                      ``arange(N)`` (every atom its own singleton orbit).
        wyckoff_rank: (N,) int Wyckoff-letter rank (0 = 'a'). 0 in simple mode.
    """
    atom_types = np.asarray(atom_types).astype(int).reshape(-1)
    n = atom_types.shape[0]
    en_value = electronegativity_values(atom_types)

    if mode == "simple" or n == 0:
        return en_value, np.arange(n, dtype=np.int64), np.zeros(n, dtype=np.int64)

    # Symmetry mode — best effort; on any failure fall back to simple labels.
    try:
        # Touch pymatgen.core.structure before SpacegroupAnalyzer import to avoid
        # a known circular-import edge case (see repo memory).
        import pymatgen.core.structure  # noqa: F401
        from pymatgen.core import Structure
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        structure = Structure(
            lattice=np.asarray(cell, dtype=np.float64),
            species=[int(z) for z in atom_types],
            coords=np.asarray(frac_coords, dtype=np.float64),
            coords_are_cartesian=False,
        )
        sga = SpacegroupAnalyzer(structure, symprec=symprec)
        dataset = sga.get_symmetry_dataset()
        if dataset is None:
            raise ValueError("spglib returned no symmetry dataset")

        # spglib >=2 returns an object/dict; support both access styles.
        equivalent_atoms = _dataset_field(dataset, "equivalent_atoms")
        wyckoffs = _dataset_field(dataset, "wyckoffs")
        if equivalent_atoms is None or wyckoffs is None:
            raise ValueError("symmetry dataset missing equivalent_atoms/wyckoffs")

        equivalent_atoms = np.asarray(equivalent_atoms).reshape(-1)
        if equivalent_atoms.shape[0] != n or len(wyckoffs) != n:
            raise ValueError("symmetry dataset size mismatch")

        # Renumber orbit ids to a compact 0..K-1 range.
        _, orbit_id = np.unique(equivalent_atoms, return_inverse=True)
        orbit_id = orbit_id.astype(np.int64)
        wyckoff_rank = np.array([_wyckoff_rank(w) for w in wyckoffs], dtype=np.int64)
        return en_value, orbit_id, wyckoff_rank
    except Exception:
        return en_value, np.arange(n, dtype=np.int64), np.zeros(n, dtype=np.int64)


def _dataset_field(dataset, name: str):
    """Read a field from a spglib symmetry dataset (dict or namedtuple/object)."""
    if isinstance(dataset, dict):
        return dataset.get(name, None)
    return getattr(dataset, name, None)


def canonical_order(
    en_value: np.ndarray,
    orbit_id: np.ndarray,
    wyckoff_rank: np.ndarray,
    frac_coords: np.ndarray,
) -> np.ndarray:
    """Return permutation indices that put atoms in canonical order.

    Orbits are ordered lexicographically by (electronegativity, Wyckoff rank,
    orbit representative fractional coords); atoms within an orbit are kept
    contiguous and ordered by their fractional coordinates. This guarantees a
    deterministic ordering and keeps symmetry-equivalent atoms adjacent.
    """
    n = en_value.shape[0]
    if n == 0:
        return np.arange(0, dtype=np.int64)

    frac = np.asarray(frac_coords, dtype=np.float64) % 1.0

    # Per-orbit representative key: orbit min fractional coordinate gives a
    # stable, geometry-based tiebreaker after (EN, Wyckoff).
    orbit_ids_unique = np.unique(orbit_id)
    orbit_rep_key = {}
    for oid in orbit_ids_unique:
        members = np.where(orbit_id == oid)[0]
        en = float(en_value[members[0]])
        wr = int(wyckoff_rank[members[0]])
        # Lexicographically smallest fractional coord within the orbit.
        rep_frac = frac[members]
        rep = rep_frac[np.lexsort((rep_frac[:, 2], rep_frac[:, 1], rep_frac[:, 0]))][0]
        orbit_rep_key[int(oid)] = (en, wr, float(rep[0]), float(rep[1]), float(rep[2]))

    # Sort orbits by their representative key.
    ordered_orbits = sorted(
        (int(o) for o in orbit_ids_unique), key=lambda o: orbit_rep_key[o]
    )

    perm = []
    for oid in ordered_orbits:
        members = np.where(orbit_id == oid)[0]
        member_frac = frac[members]
        within = np.lexsort(
            (member_frac[:, 2], member_frac[:, 1], member_frac[:, 0])
        )
        perm.extend(members[within].tolist())

    return np.asarray(perm, dtype=np.int64)


def hierarchical_permutation(
    atom_types: np.ndarray,
    orbit_id: np.ndarray,
    wyckoff_rank: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return permutation indices for symmetry-preserving augmentation.

    Two levels, both preserving compositional / crystallographic equivalence:

    * **Inter-orbit**: orbits that share the same (atom type, Wyckoff rank) are
      whole-block shuffled among themselves.
    * **Intra-orbit**: atoms inside each orbit are shuffled.

    Assumes atoms are already in canonical order (orbits contiguous). The
    returned permutation, applied on top of canonical order, stays within the
    reduced permutation space described by the paper.
    """
    n = atom_types.shape[0]
    if n <= 1:
        return np.arange(n, dtype=np.int64)

    atom_types = np.asarray(atom_types).astype(int).reshape(-1)
    orbit_id = np.asarray(orbit_id).astype(int).reshape(-1)
    wyckoff_rank = np.asarray(wyckoff_rank).astype(int).reshape(-1)

    # Build ordered list of orbits as contiguous index blocks (canonical order).
    blocks = []  # list of (super_key, np.ndarray of atom indices)
    seen = []
    for oid in orbit_id:
        if oid in seen:
            continue
        seen.append(oid)
        members = np.where(orbit_id == oid)[0]
        super_key = (int(atom_types[members[0]]), int(wyckoff_rank[members[0]]))
        blocks.append((super_key, members))

    # Intra-orbit shuffle of each block's members.
    shuffled_blocks = []
    for super_key, members in blocks:
        members = members.copy()
        if members.shape[0] > 1:
            rng.shuffle(members)
        shuffled_blocks.append((super_key, members))

    # Inter-orbit shuffle: permute blocks that share the same super_key, while
    # keeping the overall sequence of super_key groups fixed (canonical).
    from collections import defaultdict

    positions_by_key = defaultdict(list)
    for i, (super_key, _) in enumerate(shuffled_blocks):
        positions_by_key[super_key].append(i)

    new_block_order = list(range(len(shuffled_blocks)))
    for super_key, positions in positions_by_key.items():
        if len(positions) > 1:
            permuted = list(positions)
            rng.shuffle(permuted)
            for slot, src in zip(positions, permuted):
                new_block_order[slot] = src

    perm = []
    for bi in new_block_order:
        perm.extend(shuffled_blocks[bi][1].tolist())

    return np.asarray(perm, dtype=np.int64)


def global_modulo_translation(
    frac_coords: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply a random global translation to fractional coords, wrapped to [0, 1)."""
    shift = rng.random(3)
    return (np.asarray(frac_coords, dtype=np.float64) + shift[None, :]) % 1.0
