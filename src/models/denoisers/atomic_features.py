"""
Fixed, non-trainable per-element chemistry feature table.

Provides a small set of basic periodic-table properties (Pauling
electronegativity, atomic radius, period, group) indexed by atomic number Z,
for use as an explicit per-token chemistry feature -- attached directly onto
each atom's own token, alongside its learned Z-embedding.

This is intentionally separate from src/data/components/atom_ordering.py's
electronegativity table: that one is a *sort key* (with a large sentinel for
undefined values, which is fine for ordering but would be a bad, outlier-heavy
network input). This module instead mean-imputes and z-score normalizes each
column so the result is a well-scaled additive feature.
"""

import math

import torch

ATOMIC_FEATURE_DIM = 4
ATOMIC_FEATURE_NAMES = ("electronegativity", "atomic_radius", "period", "group")


def _raw_atomic_properties(max_z: int = 118) -> dict:
    """Return {Z: (X, atomic_radius, row, group)}, with None for undefined values."""
    props = {}
    try:
        from pymatgen.core.periodic_table import Element
    except Exception:  # pragma: no cover - pymatgen always present in this repo
        return props
    for z in range(1, max_z + 1):
        try:
            el = Element.from_Z(z)
            x = el.X
            if x is None or (isinstance(x, float) and math.isnan(x)):
                x = None
            else:
                x = float(x)
            radius = el.atomic_radius
            radius = float(radius) if radius is not None else None
            row = float(el.row) if el.row is not None else None
            group = float(el.group) if el.group is not None else None
        except Exception:
            x, radius, row, group = None, None, None, None
        props[z] = (x, radius, row, group)
    return props


def build_atomic_feature_table(atom_type_vocab_size: int, max_z: int = 118) -> torch.Tensor:
    """
    Build a (atom_type_vocab_size, ATOMIC_FEATURE_DIM) float32 lookup table,
    indexed by atomic number Z (same indexing scheme as atom_type_embedder).

    Each column is mean-imputed (for elements with an undefined property) and
    z-score normalized so it is a well-scaled additive input feature. Index 0
    (padding / null atom-type) is always the zero vector.

    Falls back to an all-zero table if pymatgen is unavailable, so the
    atomwise-feature pathway degrades to a harmless no-op rather than crashing.
    """
    table = torch.zeros(atom_type_vocab_size, ATOMIC_FEATURE_DIM, dtype=torch.float32)
    props = _raw_atomic_properties(max_z=max_z)
    if not props:
        return table

    columns = [[] for _ in range(ATOMIC_FEATURE_DIM)]
    for vals in props.values():
        for i, v in enumerate(vals):
            if v is not None:
                columns[i].append(v)

    means = [(sum(c) / len(c)) if c else 0.0 for c in columns]
    stds = [
        ((sum((v - m) ** 2 for v in c) / len(c)) ** 0.5) if c else 1.0
        for c, m in zip(columns, means)
    ]
    stds = [s if s > 1e-6 else 1.0 for s in stds]

    for z, vals in props.items():
        if z >= atom_type_vocab_size:
            break
        row = [
            ((v if v is not None else means[i]) - means[i]) / stds[i]
            for i, v in enumerate(vals)
        ]
        table[z] = torch.tensor(row, dtype=torch.float32)

    return table
