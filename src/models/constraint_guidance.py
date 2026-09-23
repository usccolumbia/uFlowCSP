"""Training-free constraint refinement for generated crystal structures.

Three ideas from the 2026 literature, all applied AFTER sampling so they cost no
training and can be A/B'd against an existing checkpoint:

  guidance  Differentiable constraint guidance (arXiv:2604.13354). A physical
            constraint is written as a differentiable energy and minimised by
            gradient descent on the generated structure. Here the constraint is
            the one a one-step generator actually violates: atoms sitting on top
            of each other. `structure_validity` in src/eval/crystal.py DISCARDS
            any sample with a pairwise distance below 0.5 Ang, so an overlapping
            sample is not merely bad, it is thrown away -- at k=1 that is an
            automatic miss. Nudging those atoms apart converts a discarded
            sample into a candidate.

  valence   Valence-aware radii (CrysVCD, Nat Comput Sci s43588-026-01037-2).
            CrysVCD constrains COMPOSITION by valence, which is moot for CSP
            (the composition is given). What survives the translation is the
            chemistry: if a charge-balanced oxidation assignment exists, ionic
            radii give per-PAIR distance targets, so a cation-anion contact is
            allowed to be short while a like-charge contact is not. This feeds
            the guidance energy above rather than standing alone.

  symmetry  Symmetry projection (SymmBFN, npj Comput Mater s41524-026-02140-8).
            SymmBFN builds symmetry into generation; the training-free shadow of
            it is to snap an ALMOST-symmetric generated cell onto the exact
            symmetric one. MP-20 ground truths are symmetric, so a near-miss can
            become a match.

Everything here is deliberately conservative: on a structure that already
satisfies the constraint the guidance energy is exactly zero and no step is
taken, and symmetry projection is rejected unless it preserves the composition
AND the atom count. `stats` records how often each path actually fired, so
"the refinement did nothing" is distinguishable from "the refinement did not
help".

Nothing in this module is imported by the training path.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

__all__ = ["ConstraintRefiner", "build_refiner", "REFINE_MODES"]

REFINE_MODES = ("none", "guidance", "symmetry", "both")

# 27 periodic images. MP-20 cells are small and can be strongly skewed, where the
# fractional minimum-image convention alone is not exact, so the true minimum is
# taken over the neighbouring images explicitly.
_OFFSETS = np.array(
    [[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
    dtype=np.float64,
)


# ---------------------------------------------------------------------------
# Radii
# ---------------------------------------------------------------------------
def _covalent_radius(el) -> float:
    for attr in ("covalent_radius", "atomic_radius_calculated", "atomic_radius"):
        r = getattr(el, attr, None)
        if r is not None and not (isinstance(r, float) and math.isnan(r)):
            try:
                return float(r)
            except (TypeError, ValueError):
                continue
    return 1.2


def _valence_radii(structure, enable: bool = True) -> Optional[Dict[str, float]]:
    """Per-element ionic radii from a charge-balanced oxidation assignment.

    Returns None when no charge-balanced assignment exists (an alloy, say) or
    when pymatgen has no ionic radius for one of the resulting species -- the
    caller then falls back to covalent radii. Never raises.
    """
    if not enable:
        return None
    try:
        import warnings

        from pymatgen.core.periodic_table import Species

        guesses = structure.composition.oxi_state_guesses()
        if not guesses:
            return None
        oxi = guesses[0]
        out = {}
        with warnings.catch_warnings():
            # An elemental phase guesses oxidation state 0, for which pymatgen
            # has no ionic radius and warns once per call. Expected: we fall
            # back to covalent radii below.
            warnings.simplefilter("ignore")
            for sym, ox in oxi.items():
                sp = Species(sym, int(round(float(ox))))
                r = sp.ionic_radius
                if r is None:
                    return None
                out[sym] = float(r)
        return out or None
    except Exception:
        return None


def _pair_targets(structure, scale: float, floor: float, use_valence: bool):
    """(N, N) matrix of minimum acceptable interatomic distances, in Angstrom."""
    from pymatgen.core.periodic_table import Element

    syms = [str(s.specie.symbol) for s in structure]
    ionic = _valence_radii(structure, enable=use_valence)
    radii = []
    for s in syms:
        if ionic is not None and s in ionic:
            radii.append(ionic[s])
        else:
            radii.append(_covalent_radius(Element(s)))
    r = np.asarray(radii, dtype=np.float64)
    tgt = scale * (r[:, None] + r[None, :])
    np.maximum(tgt, floor, out=tgt)
    np.fill_diagonal(tgt, 0.0)
    return tgt, (ionic is not None)


# ---------------------------------------------------------------------------
# Differentiable min-distance guidance
# ---------------------------------------------------------------------------
def _min_image_dists(frac, cell, offsets):
    """(N, N) periodic distances, differentiable in `frac` and `cell`.

    `torch.round` selects which periodic image is nearest; it contributes no
    gradient but `diff - round(diff)` still passes gradient 1 through `diff`,
    so this is a piecewise translation, not a barrier.
    """
    import torch

    diff = frac[:, None, :] - frac[None, :, :]          # (N, N, 3)
    diff = diff - torch.round(diff)                      # minimum image, fractional
    cand = diff[:, :, None, :] + offsets[None, None, :, :]   # (N, N, 27, 3)
    cart = cand @ cell                                   # (N, N, 27, 3)
    # sqrt(sum + eps), NOT torch.linalg.norm: the norm's derivative at exactly
    # zero is 0/0, and two generated atoms landing on the same site is a real
    # occurrence. A NaN there silently freezes the whole descent.
    sq = (cart ** 2).sum(dim=-1)
    d = torch.sqrt(sq + 1e-12)                           # (N, N, 27)
    return d.min(dim=-1).values                          # (N, N)


def guidance_refine(
    structure,
    steps: int = 150,
    lr: float = 0.01,
    radius_scale: float = 0.60,
    distance_floor: float = 0.90,
    use_valence: bool = True,
    allow_scale: bool = True,
    max_log_scale: float = 0.22,     # exp(0.22) ~ 1.25x linear, ~1.9x volume
    scale_penalty: float = 5.0,
    tol: float = 1e-4,
):
    """Gradient descent on fractional coords (+ an isotropic cell scale) that
    relieves interatomic overlap.

    Returns (structure, fired) where `fired` is False when the input already
    satisfied every pair target -- in that case the input is returned unchanged,
    bit for bit.

    The cell is allowed only an isotropic breathing scale, penalised toward 1.0
    and hard-clamped. Fractional coordinates carry the crystal's actual
    geometry, and pymatgen's StructureMatcher compares volume-normalised cells,
    so a small isotropic scale is nearly free on the metric while making the
    constraint satisfiable without distorting the predicted geometry.
    """
    import torch
    from pymatgen.core import Structure

    n = len(structure)
    if n < 2:
        return structure, False

    tgt_np, _ = _pair_targets(structure, radius_scale, distance_floor, use_valence)
    tgt = torch.tensor(tgt_np, dtype=torch.float64)
    iu = torch.triu_indices(n, n, offset=1)

    frac0 = torch.tensor(np.asarray(structure.frac_coords), dtype=torch.float64)
    cell0 = torch.tensor(np.asarray(structure.lattice.matrix), dtype=torch.float64)
    offsets = torch.tensor(_OFFSETS, dtype=torch.float64)

    with torch.no_grad():
        d0 = _min_image_dists(frac0, cell0, offsets)
        viol0 = torch.clamp(tgt - d0, min=0.0)[iu[0], iu[1]]
        if float(viol0.max()) <= tol:
            return structure, False          # already fine: do not touch it
        # Exactly coincident atoms sit at a symmetric saddle: the repulsive
        # gradient on each is ~0 because the separation direction is undefined,
        # so descent alone would leave them stacked forever. A tiny DETERMINISTIC
        # jitter (~0.005 Ang) breaks the tie without perturbing anything else;
        # a fixed generator keeps the run reproducible.
        if float(d0[iu[0], iu[1]].min()) < 1e-3:
            g = torch.Generator().manual_seed(0)
            frac0 = frac0 + 1e-3 * (
                torch.rand(frac0.shape, generator=g, dtype=torch.float64) - 0.5
            )

    frac = frac0.clone().requires_grad_(True)
    log_s = torch.zeros(1, dtype=torch.float64, requires_grad=allow_scale)
    params = [frac] + ([log_s] if allow_scale else [])
    opt = torch.optim.Adam(params, lr=lr)

    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        s = torch.clamp(log_s, -max_log_scale, max_log_scale)
        cell = cell0 * torch.exp(s)
        d = _min_image_dists(frac, cell, offsets)
        viol = torch.clamp(tgt - d, min=0.0)[iu[0], iu[1]]
        energy = (viol ** 2).sum()
        if allow_scale:
            energy = energy + scale_penalty * (s ** 2).sum()
        if float(viol.max()) <= tol:
            break
        energy.backward()
        opt.step()

    with torch.no_grad():
        s = torch.clamp(log_s, -max_log_scale, max_log_scale)
        cell_f = (cell0 * torch.exp(s)).numpy()
        frac_f = torch.remainder(frac, 1.0).numpy()

    try:
        out = Structure(
            lattice=cell_f,
            species=[s.specie for s in structure],
            coords=frac_f,
            coords_are_cartesian=False,
        )
    except Exception:
        return structure, False
    return out, True


# ---------------------------------------------------------------------------
# Symmetry projection
# ---------------------------------------------------------------------------
def symmetry_refine(structure, symprecs: Optional[List[float]] = None):
    """Snap a nearly-symmetric cell onto the exactly-symmetric one.

    Tries the LOOSEST tolerance first, so the highest symmetry the structure can
    plausibly claim wins; symprec doubles as a bound on how far any atom moves.

    Z MUST NOT CHANGE. The evaluation is known-Z: the model was handed the exact
    cell contents (Ti2O4, not TiO2), so a projection that returns a smaller
    primitive cell has answered a different question. spglib's refinement can
    move EITHER way -- the refined cell of a conventional input is conventional,
    while its primitive is smaller -- so several standard settings are tried and
    the first one with exactly the input's atom count and reduced formula wins.
    An earlier version called get_primitive_structure() unconditionally and so
    rejected precisely the cases where symmetrisation had worked.

    Returns (structure, fired), where `fired` means the geometry actually moved,
    not merely that a candidate was accepted.
    """
    if symprecs is None:
        symprecs = [0.3, 0.2, 0.1, 0.05]
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    except Exception:
        return structure, False

    n0 = len(structure)
    try:
        f0 = structure.composition.reduced_formula
    except Exception:
        return structure, False

    for sp in symprecs:
        try:
            sga = SpacegroupAnalyzer(structure, symprec=float(sp))
            # P1 has nothing to snap onto. Projecting it would only restandardise
            # the cell setting -- a no-op for a rotation-invariant matcher, but it
            # would inflate the `fired` count and waste spglib calls. Try the next
            # (tighter) tolerance instead.
            if int(sga.get_space_group_number()) <= 1:
                continue
        except Exception:
            continue
        for getter in ("get_refined_structure",
                       "get_conventional_standard_structure",
                       "get_primitive_standard_structure"):
            try:
                cand = getattr(sga, getter)()
            except Exception:
                continue
            try:
                if len(cand) != n0:
                    continue
                if cand.composition.reduced_formula != f0:
                    continue
                if cand.volume <= 0.1:
                    continue
            except Exception:
                continue
            return cand, _changed(structure, cand)
    return structure, False


def _changed(a, b, tol: float = 1e-6) -> bool:
    """Did the geometry actually move? Guards the `fired` counters from
    reporting a no-op projection as a hit."""
    try:
        if len(a) != len(b):
            return True
        if abs(a.volume - b.volume) > tol:
            return True
        da = np.abs(np.asarray(a.lattice.matrix) - np.asarray(b.lattice.matrix)).max()
        if da > tol:
            return True
        df = np.asarray(a.frac_coords) - np.asarray(b.frac_coords)
        df = df - np.round(df)
        return bool(np.abs(df).max() > tol)
    except Exception:
        return True


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
class ConstraintRefiner:
    """structure -> structure. Safe by construction: any failure returns the input."""

    def __init__(self, mode: str = "none", **kw):
        if mode not in REFINE_MODES:
            raise ValueError(f"refine mode must be one of {REFINE_MODES}, got {mode!r}")
        self.mode = mode
        self.kw = kw
        self.stats = {
            "seen": 0,
            "guidance_fired": 0,
            "symmetry_fired": 0,
            "errors": 0,
        }

    @property
    def enabled(self) -> bool:
        return self.mode != "none"

    def refine(self, structure):
        if not self.enabled or structure is None:
            return structure
        self.stats["seen"] += 1
        out = structure
        try:
            if self.mode in ("guidance", "both"):
                out, fired = guidance_refine(out, **self.kw)
                self.stats["guidance_fired"] += int(fired)
            if self.mode in ("symmetry", "both"):
                out, fired = symmetry_refine(out)
                self.stats["symmetry_fired"] += int(fired)
        except Exception:
            self.stats["errors"] += 1
            return structure
        return out

    def summary(self) -> str:
        s = self.stats
        return (
            f"refine[{self.mode}]: seen={s['seen']} "
            f"guidance_fired={s['guidance_fired']} "
            f"symmetry_fired={s['symmetry_fired']} errors={s['errors']}"
        )


def build_refiner(mode: str = "none", **kw) -> Optional[ConstraintRefiner]:
    """None when disabled, so callers can branch on identity."""
    if not mode or mode == "none":
        return None
    return ConstraintRefiner(mode, **kw)
