#!/usr/bin/env python3
"""
Rank relaxed generated CIFs per formula by CHGNet energy, select top-5
(lowest energy = most stable), copy them to a flat folder with descriptive
names, and evaluate top-5 structure-match and spacegroup-match rates against
ground truth.

Inputs
------
- gen_dir:    <gen_dir>/<formula>/sample_<N>.cif   (relaxed output of
              src/match_meanflow_raw_all_formulas.py with --relax)
- gt_dir:     directory of ground-truth CIFs named <formula>.cif
- csv:        the input CSV with material_id, primitive_formula, spacegroup
- output_dir: where to write top-5 cifs + per-formula CSV + summary JSON

Outputs
-------
<output_dir>/top5_cifs/<formula>_<material_id>_rank<R>_sample<N>.cif
<output_dir>/per_formula_results.csv
<output_dir>/summary.json

Protocol (matches paper)
------------------------
- StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=True)
  (pymatgen defaults; matches the CSPBench / TCSP2 180-test protocol)
  --diffcsp: switch to the DiffCSP (Jiao et al., 2023) CSP-task protocol
  instead -- looser StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10),
  an any-of-k match rate over ALL candidates (not top-k by energy), and a
  normalized RMSE over matched structures.
  --per-material: score one row per test material_id (each against its own GT
  polymorph <gt_dir>/<material_id>.cif) instead of per unique composition. This
  matches DiffCSP's per-structure denominator (e.g. 9046 for MP-20). For a
  fully DiffCSP-comparable number use both: --diffcsp --per-material.
- SpacegroupAnalyzer(symprec=0.1) for all symmetry analyses
- Ranking: energy per atom, ascending (lowest = best).
  --ranker chgnet  (default): CHGNet predicted energy per atom.
  --ranker matris:            MatRIS predicted energy per atom.

Usage
-----
# CHGNet ranking (default):
python select_top5_and_evaluate.py \
    --gen_dir results_generate/generated_struc_30Apr \
    --gt_dir ground_truth_180 \
    --csv 180_primitive.csv \
    --output_dir results_generate/top5_eval_30Apr \
    --device cuda

# MatRIS ranking:
python select_top5_and_evaluate.py \
    --gen_dir results_generate/generated_struc_30Apr \
    --gt_dir ground_truth_180 \
    --csv 180_primitive.csv \
    --output_dir results_generate/top5_eval_30Apr_matris \
    --ranker matris --matris_model matris_10m_oam --device cuda
"""

import argparse
import json
import re
import shutil
import warnings
from pathlib import Path

import pandas as pd
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pyxtal import pyxtal
from tqdm import tqdm

warnings.filterwarnings("ignore")

try:
    from chgnet.model import CHGNet
    CHGNET_AVAILABLE = True
except ImportError:
    CHGNET_AVAILABLE = False
    CHGNet = None

# MatRIS runs in a separate conda env ('matris') to avoid numpy/torch conflicts.
# It is called as a subprocess via matris_predict_worker.py — no direct import.


# ---------------------------------------------------------------------------
# Symmetry helpers (paper protocol: symprec=0.1 everywhere)
# ---------------------------------------------------------------------------

def get_spacegroup(structure):
    """SpacegroupAnalyzer(symprec=0.1) -> pyxtal -> SG number. Returns int or None."""
    try:
        spga = SpacegroupAnalyzer(structure, symprec=0.1)
        refined = spga.get_refined_structure()
        c = pyxtal()
        try:
            c.from_seed(refined, tol=0.01)
        except Exception:
            c.from_seed(refined, tol=0.0001)
        return int(c.group.number)
    except Exception:
        return None


def get_symmetrized(structure):
    try:
        return SpacegroupAnalyzer(structure, symprec=0.1).get_symmetrized_structure()
    except Exception:
        return structure


# Increasing tolerance ladder for robust (multi-tolerance) symmetrization.
# Tighter tols first; a looser tol can recover a higher-symmetry group when
# generated coordinates carry sub-threshold noise (the "P1 collapse" fix).
SYMPREC_LADDER = (0.01, 0.05, 0.1, 0.2, 0.3)

# Dedicated STRICT matcher for the symmetrization consistency guard. This is
# intentionally independent of the (possibly loose) evaluation matcher: a loose
# guard would let an over-idealised refinement pass and corrupt results.
_SYM_GUARD_MATCHER = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=True)


def get_spacegroup_robust(structure, symprecs=SYMPREC_LADDER, matcher=None):
    """Multi-tolerance spacegroup *labelling* (item 2).

    Sweep ``symprecs`` and return the SG number of the assignment with the MOST
    symmetry operations whose symmetry-refined structure still represents the
    same structure as the input (strict consistency guard). This only recovers
    symmetry that is genuinely (approximately) present — it never invents it.

    IMPORTANT: this is used ONLY to assign the spacegroup label. The refined /
    idealised structure is deliberately NOT returned for use in structure
    matching — feeding an idealised conventional cell to the StructureMatcher
    moves atoms and destroys true positive matches. Structure matching always
    runs on the lightly-symmetrised (symprec=0.1) structure, exactly as the
    baseline does. Returns ``sg_number`` (int) or None.
    """
    guard = matcher if matcher is not None else _SYM_GUARD_MATCHER
    best_sg, best_nops = None, -1
    for sp in symprecs:
        try:
            spga = SpacegroupAnalyzer(structure, symprec=sp)
            nops = len(spga.get_symmetry_operations())
            if nops <= best_nops:
                continue
            refined = spga.get_refined_structure()
            try:
                if not guard.fit(structure, refined):
                    continue
            except Exception:
                continue
            c = pyxtal()
            try:
                c.from_seed(refined, tol=0.01)
            except Exception:
                c.from_seed(refined, tol=0.0001)
            best_sg, best_nops = int(c.group.number), nops
        except Exception:
            continue
    if best_sg is None:
        return get_spacegroup(structure)
    return best_sg


def consensus_select(ranked, matcher, top_k):
    """Structure-consensus selection (item 1), hybrid / safe variant.

    Relaxer energies within a formula are often near-degenerate (<1 meV, below
    an MLIP's accuracy floor), so the lowest-energy pick is unreliable. We
    cluster candidates by structural identity, take the largest cluster (the
    generator's "mode" — the structure it produced most often) and PROMOTE its
    lowest-energy member to rank 1. The remaining slots are filled by energy
    order, exactly as the baseline.

    This guarantees ``top_k`` ⊇ {modal pick} ∪ {top-(k-1) by energy}, so it can
    never lose more than one energy slot relative to the baseline while adding
    the high-value consensus pick at rank 1. (An earlier pure "one rep per
    cluster, ranked by size" variant could push the correct low-energy
    structure out of the top-k and is the reason a naive consensus backfired.)

    ``ranked`` is a list of ``(energy, sample_idx, structure, src_path)``.
    Returns ``(ordered_indices_into_ranked, {index: cluster_size})``.
    """
    clusters = []  # list[list[int]] — indices into ranked
    for i in range(len(ranked)):
        si = ranked[i][2]
        placed = False
        for cl in clusters:
            try:
                if matcher.fit(ranked[cl[0]][2], si):
                    cl.append(i)
                    placed = True
                    break
            except Exception:
                continue
        if not placed:
            clusters.append([i])

    size_for = {j: len(cl) for cl in clusters for j in cl}

    # modal cluster: largest, tie-break by lowest energy in the cluster
    clusters.sort(key=lambda cl: (-len(cl), min(ranked[j][0] for j in cl)))
    modal_rep = min(clusters[0], key=lambda j: ranked[j][0])

    by_energy = sorted(range(len(ranked)), key=lambda j: ranked[j][0])
    order = [modal_rep] + [j for j in by_energy if j != modal_rep]
    order = order[:top_k]
    return order, size_for


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------

def normalize_formula(f):
    return Composition(f).reduced_formula.replace(" ", "")


def safe_tag(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "", s)


def parse_sample_idx(cif_path):
    m = re.search(r"sample_(\d+)", cif_path.stem)
    return int(m.group(1)) if m else -1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gen_dir", required=True,
                   help="Per-formula subfolders of relaxed CIFs.")
    p.add_argument("--gt_dir", required=True,
                   help="Ground-truth CIFs named <formula>.cif")
    p.add_argument("--csv", required=True,
                   help="CSV with material_id, primitive_formula, spacegroup columns.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"])
    p.add_argument("--material_id_col", default="material_id")
    p.add_argument("--formula_col", default="primitive_formula")
    p.add_argument("--spacegroup_col", default="spacegroup")
    # Selection strategy (item 1) and symmetrization (item 2) — both opt-in,
    # defaults reproduce the original energy-ranking + single-tol behaviour.
    p.add_argument("--selection", default="energy", choices=["energy", "consensus"],
                   help="Candidate selection: 'energy' (default, lowest relaxer "
                        "energy first) or 'consensus' (largest structure-cluster "
                        "first, energy as tie-break).")
    p.add_argument("--symmetrize", default="single", choices=["single", "multi"],
                   help="Spacegroup detection / symmetrization for candidates: "
                        "'single' (default, symprec=0.1) or 'multi' (tolerance "
                        "sweep with a structure-consistency guard).")
    # Ranker selection
    p.add_argument("--ranker", default="chgnet", choices=["chgnet", "matris", "none"],
                   help="Energy model for ranking: 'chgnet' (default), 'matris', or "
                        "'none' (no energy model; dummy energies in file order — use "
                        "for the pure DiffCSP protocol where only any-of-k matching "
                        "matters and top-k-by-energy metrics are meaningless).")
    p.add_argument("--skip_sg", action="store_true",
                   help="Skip per-candidate (and GT) spacegroup labelling entirely — "
                        "the slowest part of the eval (pyxtal per CIF). Spacegroup/"
                        "consensus metrics become empty; structure match + DiffCSP "
                        "any-of-k + RMSE are unaffected. Papers' protocol needs none "
                        "of the SG metrics.")
    p.add_argument("--matris_model", default="matris_10m_oam",
                   choices=["matris_10m_oam", "matris_10m_mp", "matris_10m_omat"],
                   help="MatRIS pretrained model key (used only when --ranker matris).")
    p.add_argument("--matris_python", default="auto",
                   help="Path to Python in the 'matris' conda env. "
                        "'auto' detects automatically. "
                        "Set to the output of: conda run -n matris which python")
    # DiffCSP CSP-task protocol (opt-in). Off by default => CSPBench-strict.
    p.add_argument("--diffcsp", action="store_true",
                   help="Evaluate with the DiffCSP (Jiao et al., 2023) CSP protocol "
                        "instead of the CSPBench-strict one: loosen StructureMatcher "
                        "tolerances to ltol=0.3, stol=0.5, angle_tol=10; report an "
                        "any-of-k match rate over ALL candidates (not top-k by "
                        "energy); and compute the normalized RMSE over matched "
                        "structures. Top-k energy selection / CIF export is unchanged.")
    # Per-material evaluation (DiffCSP-faithful). Off by default => per-formula.
    p.add_argument("--per-material", dest="per_material", action="store_true",
                   help="Score one row per test material_id (denominator = number "
                        "of CSV rows, e.g. 9046 for MP-20), each against ITS OWN "
                        "ground-truth polymorph loaded from <gt_dir>/<material_id>.cif. "
                        "This matches DiffCSP's per-structure protocol. Default "
                        "(off) keys by composition, collapsing polymorphs to one GT "
                        "per formula. Candidates are shared across materials of the "
                        "same composition, so energies are computed once per formula.")
    p.add_argument("--own-candidates-only", dest="own_candidates_only",
                   action="store_true",
                   help="With --per-material, score each material against ONLY the "
                        "candidates it generated (files named <material_id>_sample_*.cif, "
                        "written by the sampler's --material_prefix). Default (off) "
                        "pools every CIF in the formula folder, so polymorphs sharing a "
                        "composition are each scored against the union — an MP-20 "
                        "known-Z run has ~10%% of materials seeing far more than k "
                        "candidates (observed max 135 for k=20), which inflates the "
                        "any-of-k rate above the k you report. Requires prefixed "
                        "filenames; falls back to the pooled behaviour with a warning "
                        "if none are found.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# MatRIS subprocess helpers
# ---------------------------------------------------------------------------

def _find_matris_python():
    """Locate the Python binary in the 'matris' conda environment."""
    import subprocess
    from pathlib import Path

    # Try conda run (works whether env is in base or user prefix)
    try:
        r = subprocess.run(
            ["conda", "run", "-n", "matris", "which", "python"],
            capture_output=True, text=True, timeout=15,
        )
        if r.returncode == 0:
            path = r.stdout.strip()
            if path and Path(path).exists():
                return path
    except Exception:
        pass

    # Fallback: scan common conda base dirs
    home = Path.home()
    for base in ["miniconda3", "anaconda3", "miniforge3", "mambaforge", ".conda"]:
        candidate = home / base / "envs" / "matris" / "bin" / "python"
        if candidate.exists():
            return str(candidate)
    return None


def _run_matris_subprocess(cif_paths, python_exe, model, device):
    """Call matris_predict_worker.py and return {path_str: energy_per_atom}."""
    import subprocess
    import json
    from pathlib import Path

    worker = Path(__file__).parent / "matris_predict_worker.py"
    if not worker.exists():
        raise FileNotFoundError(
            f"matris_predict_worker.py not found at {worker}. "
            "It should be in the same directory as select_top5_and_evaluate.py."
        )

    cmd = [python_exe, str(worker), "--model", model, "--device", device]
    input_json = json.dumps([str(p) for p in cif_paths])

    result = subprocess.run(cmd, input=input_json, capture_output=True, text=True)

    # Always show worker stderr (progress + errors)
    if result.stderr:
        for line in result.stderr.strip().splitlines():
            print(f"  {line}")

    if result.returncode != 0:
        raise RuntimeError(
            f"MatRIS worker exited with code {result.returncode}.\n"
            f"stderr:\n{result.stderr}"
        )

    records = json.loads(result.stdout)
    cache = {}
    for rec in records:
        if rec["error"] is None:
            cache[rec["path"]] = rec["energy_per_atom"]
        else:
            print(f"  MatRIS failed for {rec['path']}: {rec['error']}")
    return cache  # path_str -> eV/atom


def resolve_device(arg):
    if arg != "auto":
        return arg
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# Scoring helpers (shared by the per-formula and per-material evaluation loops)
# ---------------------------------------------------------------------------

def predict_ranked(cif_paths, ranker, chgnet, matris_energy_cache):
    """Energy-rank candidates. Returns [(energy, sample_idx, structure, path), ...].

    ranker='none' assigns dummy 0.0 energies (file order) — no model is loaded or
    called; with top_k >= n_candidates the any-of-k / match metrics are identical,
    only the (meaningless) within-top-k ordering differs.
    """
    ranked = []
    for cif_path in cif_paths:
        try:
            s = Structure.from_file(str(cif_path))
            if ranker == "chgnet":
                e = float(chgnet.predict_structure(s, task="e")["e"])  # eV/atom
            elif ranker == "none":
                e = 0.0
            else:  # matris — use pre-computed batch cache
                e = matris_energy_cache.get(str(cif_path))
                if e is None:
                    raise ValueError("no pre-computed MatRIS energy (prediction failed in batch)")
            ranked.append((e, parse_sample_idx(cif_path), s, cif_path))
        except Exception as ex:
            print(f"  Energy prediction failed for {cif_path.name}: {ex}")
    return ranked


def mid_file_tag(material_id: str) -> str:
    """Filename prefix the sampler writes for a material under --material_prefix.

    Must stay byte-identical to match_meanflow_raw_all_formulas.py, which builds
    it as re.sub(r"[^A-Za-z0-9_.-]+", "", material_id).
    """
    return re.sub(r"[^A-Za-z0-9_.-]+", "", str(material_id).strip())


def fail_result(gt_sg, n_cand, diffcsp):
    """Result columns for a formula/material with no usable candidates."""
    return {
        "n_candidates": n_cand,
        "n_selected": 0,
        "structure_match_top5": False,
        "spacegroup_match_top5": False,
        "consensus_top5": False,
        "structure_match_anyk": False if diffcsp else "",
        "rmse": "",
        "gt_spacegroup": gt_sg,
        "selected_files": "",
        "selected_energies": "",
        "selected_spacegroups": "",
        "selected_cluster_sizes": "",
        "first_match_rank": "",
        "first_sg_match_rank": "",
        "first_consensus_rank": "",
    }


def score_candidates(ranked, gt_entry, gt_sg, matcher, args, top5_dir, out_tag, formula_norm):
    """Top-k select + copy CIFs + structure/spacegroup match, plus the DiffCSP
    any-of-k match + RMSE. Returns a dict of result columns (no formula /
    material_id / status). ``gt_entry`` may be None (all matches then False)."""
    if args.selection == "consensus":
        sel_order, csize_for = consensus_select(ranked, matcher, args.top_k)
        top = [ranked[j] for j in sel_order]
        top_sizes = [csize_for.get(j) for j in sel_order]
    else:
        ranked = sorted(ranked, key=lambda t: t[0])  # ascending: lowest energy first
        top = ranked[: args.top_k]
        top_sizes = [None] * len(top)

    sel_files, sel_energies, sel_sgs, sel_sizes = [], [], [], []
    sm_rank = sg_rank = cs_rank = None

    for rank, ((energy, sample_idx, struct, src_path), csize) in enumerate(
        zip(top, top_sizes), start=1
    ):
        dst_name = f"{formula_norm}_{out_tag}_rank{rank}_sample{sample_idx}.cif"
        shutil.copy2(src_path, top5_dir / dst_name)
        sel_files.append(dst_name)
        sel_energies.append(round(energy, 6))
        sel_sizes.append(csize if csize is not None else "")

        # Structure used for matching is ALWAYS the lightly-symmetrised
        # (symprec=0.1) structure, so the symmetrize mode never changes matches.
        struct_sym = get_symmetrized(struct)
        if getattr(args, "skip_sg", False):
            pred_sg = None  # SG labelling skipped (pure-matching protocol)
        elif args.symmetrize == "multi":
            pred_sg = get_spacegroup_robust(struct)  # affects the SG LABEL only
        else:
            pred_sg = get_spacegroup(struct_sym)
        sel_sgs.append(pred_sg if pred_sg is not None else "")

        if gt_entry is not None:
            try:
                is_sm = bool(matcher.fit(gt_entry["structure"], struct_sym))
            except Exception:
                is_sm = False
            is_sg = (pred_sg is not None and gt_sg is not None and pred_sg == gt_sg)
            if is_sm and sm_rank is None:
                sm_rank = rank
            if is_sg and sg_rank is None:
                sg_rank = rank
            if is_sm and is_sg and cs_rank is None:
                cs_rank = rank

    # DiffCSP: any-of-k match over ALL candidates + best (min) normalized RMSE.
    diffcsp_match, diffcsp_rmse = False, None
    if args.diffcsp and gt_entry is not None:
        for (_e, _idx, cand, _p) in ranked:
            try:
                rd = matcher.get_rms_dist(gt_entry["structure"], get_symmetrized(cand))
            except Exception:
                rd = None
            if rd is not None:
                diffcsp_match = True
                r = float(rd[0])
                diffcsp_rmse = r if diffcsp_rmse is None else min(diffcsp_rmse, r)

    return {
        "n_candidates": len(ranked),
        "n_selected": len(top),
        "structure_match_top5": sm_rank is not None,
        "spacegroup_match_top5": sg_rank is not None,
        "consensus_top5": cs_rank is not None,
        "structure_match_anyk": diffcsp_match if args.diffcsp else "",
        "rmse": round(diffcsp_rmse, 6) if diffcsp_rmse is not None else "",
        "gt_spacegroup": gt_sg,
        "selected_files": "|".join(sel_files),
        "selected_energies": "|".join(str(e) for e in sel_energies),
        "selected_spacegroups": "|".join(str(s) for s in sel_sgs),
        "selected_cluster_sizes": "|".join(str(s) for s in sel_sizes),
        "first_match_rank": sm_rank if sm_rank else "",
        "first_sg_match_rank": sg_rank if sg_rank else "",
        # Rank of the first candidate that matches structure AND spacegroup at
        # the same rank. Exported so a single top_k=10 run yields SMR/SGMR/CR at
        # every depth <= top_k (rate@n = fraction with rank <= n); consensus is
        # not recoverable from the two rank columns above.
        "first_consensus_rank": cs_rank if cs_rank else "",
    }


def main():
    args = parse_args()
    gen_dir    = Path(args.gen_dir)
    gt_dir     = Path(args.gt_dir)
    csv_path   = Path(args.csv)
    output_dir = Path(args.output_dir)

    if not gen_dir.is_dir():
        raise FileNotFoundError(f"gen_dir not found: {gen_dir}")
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"gt_dir not found: {gt_dir}")
    if not csv_path.is_file():
        raise FileNotFoundError(f"csv not found: {csv_path}")

    top5_dir = output_dir / "top5_cifs"
    top5_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f"Selection: {args.selection}   Symmetrize: {args.symmetrize}   "
          f"Ranker: {args.ranker}   top_k: {args.top_k}   "
          f"Protocol: {'diffcsp' if args.diffcsp else 'cspbench'}   "
          f"Eval-unit: {'material' if args.per_material else 'formula'}")

    # -- Load CHGNet (only when needed) --------------------------------------
    chgnet = None
    if args.ranker == "chgnet":
        if not CHGNET_AVAILABLE:
            raise ImportError("CHGNet not installed. Run: pip install chgnet")
        print(f"Loading CHGNet on device={device} ...")
        chgnet = CHGNet.load(use_device=device)
        chgnet.eval()
        print("CHGNet ready.")

    # MatRIS: resolve the worker Python path now; actual prediction happens
    # after we know all CIF paths (one batched subprocess call below).
    matris_python = None
    matris_energy_cache = {}  # populated later; path_str -> eV/atom
    if args.ranker == "matris":
        if args.matris_python == "auto":
            matris_python = _find_matris_python()
            if matris_python is None:
                raise RuntimeError(
                    "Could not auto-detect the 'matris' conda environment. "
                    "Run: bash setup_matris_env.sh\n"
                    "Then pass its Python path via --matris_python."
                )
            print(f"Auto-detected MatRIS Python: {matris_python}")
        else:
            matris_python = args.matris_python
            print(f"Using MatRIS Python: {matris_python}")

    # -- CSV: per-formula material_id and ground-truth spacegroup label -------
    df = pd.read_csv(csv_path)
    df[args.formula_col] = df[args.formula_col].astype(str).str.strip()
    formula_to_csv = {}
    for _, row in df.iterrows():
        f = normalize_formula(row[args.formula_col])
        sg_val = row[args.spacegroup_col]
        formula_to_csv[f] = {
            "material_id": str(row.get(args.material_id_col, "")).strip(),
            "spacegroup": int(sg_val) if not pd.isna(sg_val) else None,
        }

    # -- Build ground-truth lookups -------------------------------------------
    # gt_lookup     : formula      -> entry (per-formula mode; last CIF wins, so
    #                 polymorphs collapse to a single GT per composition).
    # gt_lookup_mid : material_id  -> entry (per-material mode). The material_id
    #                 is the CIF filename stem, exactly as make_gt_cifs.py writes
    #                 them (<material_id>.cif), so every test polymorph is kept.
    print(f"Loading ground-truth CIFs from {gt_dir} ...")
    gt_lookup = {}
    gt_lookup_mid = {}
    for cif_path in sorted(gt_dir.rglob("*.cif")):
        try:
            s = Structure.from_file(str(cif_path))
            s_sym = get_symmetrized(s)
            entry = {
                "structure": s_sym,
                "spacegroup": None if args.skip_sg else get_spacegroup(s_sym),
                "path": cif_path,
            }
            gt_lookup[normalize_formula(s.composition.reduced_formula)] = entry
            gt_lookup_mid[cif_path.stem] = entry
        except Exception as e:
            print(f"  Skipping invalid GT {cif_path.name}: {e}")
    print(f"Loaded {len(gt_lookup_mid)} GT structures "
          f"({len(gt_lookup)} unique formulas).")

    # Evaluation matcher. Default is the CSPBench / TCSP2 protocol (ltol=0.2,
    # stol=0.3, angle_tol=5) used on the 180-formula test set. --diffcsp switches
    # to the looser DiffCSP CSP-task tolerances. This single matcher object is
    # used for BOTH the top-k structure match and the DiffCSP any-of-k match, so
    # the two never disagree on tolerances.
    if args.diffcsp:
        matcher = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10, primitive_cell=True)
        matcher_desc = "ltol=0.3, stol=0.5, angle_tol=10, primitive_cell=True (DiffCSP)"
    else:
        matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=True)
        matcher_desc = "ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=True (CSPBench/TCSP2)"

    formula_folders = [p for p in sorted(gen_dir.iterdir()) if p.is_dir()]
    print(f"Found {len(formula_folders)} formula folders under {gen_dir}.")

    # All formulas expected from the CSV (denominator for paper-style match rate).
    csv_formulas = list(formula_to_csv.keys())
    folder_by_formula = {normalize_formula(p.name): p for p in formula_folders}
    print(f"CSV expects {len(csv_formulas)} formulas; {len(folder_by_formula)} have generated folders.")

    # -- MatRIS: pre-compute all energies in one batched subprocess call ------
    # (model is loaded once in the worker; avoids repeated process-spawn overhead)
    if args.ranker == "matris":
        all_cif_paths = [
            cif
            for folder in formula_folders
            for cif in sorted(folder.glob("*.cif"))
        ]
        print(f"Running MatRIS on {len(all_cif_paths)} CIFs "
              f"(model={args.matris_model!r}, device={device}) ...")
        matris_energy_cache = _run_matris_subprocess(
            all_cif_paths, matris_python, args.matris_model, device
        )
        print(f"MatRIS: energies cached for "
              f"{len(matris_energy_cache)}/{len(all_cif_paths)} CIFs.")

    rows = []
    if args.per_material:
        # DiffCSP-faithful: one row per test material_id (denominator = CSV rows),
        # each scored against ITS OWN GT polymorph. Candidates are shared across
        # materials of the same composition (generation is composition-conditioned),
        # so energies are computed once per formula and cached: (ranked, n_cif).
        records = list(zip(
            df[args.material_id_col].astype(str).str.strip().tolist(),
            [normalize_formula(x) for x in df[args.formula_col].tolist()],
        ))
        ranked_cache = {}
        _prefix_fallbacks = 0
        for mid, formula_norm in tqdm(records, desc="Materials"):
            folder = folder_by_formula.get(formula_norm)
            gt_entry = gt_lookup_mid.get(mid)
            gt_sg = gt_entry["spacegroup"] if gt_entry else None

            # Candidate pool. Default pools every CIF in the formula folder, which
            # is shared by all polymorphs of that composition. own_candidates_only
            # restricts it to the files THIS material generated, so it is scored at
            # exactly k rather than at k * (number of polymorphs).
            cache_key = mid if args.own_candidates_only else formula_norm
            if cache_key not in ranked_cache:
                if folder is None:
                    cif_paths = []
                elif args.own_candidates_only:
                    cif_paths = sorted(folder.glob(f"{mid_file_tag(mid)}_sample_*.cif"))
                    if not cif_paths:
                        cif_paths = sorted(folder.glob("*.cif"))
                        _prefix_fallbacks += 1
                else:
                    cif_paths = sorted(folder.glob("*.cif"))
                ranked_cache[cache_key] = (
                    predict_ranked(cif_paths, args.ranker, chgnet, matris_energy_cache),
                    len(cif_paths),
                )
            ranked, n_cif = ranked_cache[cache_key]

            base = {"material_id": mid, "formula": formula_norm}
            if not ranked:
                status = ("no_folder" if folder is None else
                          "no_candidates" if n_cif == 0 else "all_energy_failed")
                rows.append({**base, **fail_result(gt_sg, n_cif, args.diffcsp),
                             "status": status})
                continue
            res = score_candidates(ranked, gt_entry, gt_sg, matcher, args, top5_dir,
                                   safe_tag(mid) or "unknown", formula_norm)
            rows.append({**base, **res,
                         "status": "ok" if gt_entry is not None else "no_gt"})
    else:
        for formula_norm in tqdm(csv_formulas, desc="Formulas"):
            folder = folder_by_formula.get(formula_norm)
            cif_paths = sorted(folder.glob("*.cif")) if folder is not None else []
            gt_entry = gt_lookup.get(formula_norm)
            gt_sg = gt_entry["spacegroup"] if gt_entry else None
            material_id = formula_to_csv.get(formula_norm, {}).get("material_id", "")
            material_tag = safe_tag(material_id) if material_id else "unknown"

            ranked = predict_ranked(cif_paths, args.ranker, chgnet, matris_energy_cache)

            base = {"material_id": material_id, "formula": formula_norm}
            if not ranked:
                status = ("no_folder" if folder is None else
                          "no_candidates" if not cif_paths else "all_energy_failed")
                rows.append({**base, **fail_result(gt_sg, len(cif_paths), args.diffcsp),
                             "status": status})
                continue
            res = score_candidates(ranked, gt_entry, gt_sg, matcher, args, top5_dir,
                                   material_tag, formula_norm)
            rows.append({**base, **res,
                         "status": "ok" if gt_entry is not None else "no_gt"})

    # -- Save per-row CSV ------------------------------------------------------
    if args.per_material and args.own_candidates_only and _prefix_fallbacks:
        print(f"\nWARNING: --own-candidates-only requested but {_prefix_fallbacks} "
              f"material(s) had no <material_id>_sample_*.cif files; those fell back "
              f"to the POOLED formula folder. Re-sample with MATERIAL_PREFIX=1 for a "
              f"clean per-material pool.")
    eval_unit = "material" if args.per_material else "formula"
    per_df = pd.DataFrame(rows)
    per_csv = output_dir / f"per_{eval_unit}_results.csv"
    per_df.to_csv(per_csv, index=False)
    print(f"\nWrote per-{eval_unit} results to {per_csv}")

    # -- Aggregate summary -----------------------------------------------------
    # Denominator = every evaluation unit from the CSV (materials in per-material
    # mode, else unique formulas). Rows with no folder / no candidates / no GT
    # count as failures.
    n_total = len(per_df)
    if n_total == 0:
        print(f"No {eval_unit}s in CSV — skipping summary.")
        return

    n_sm = int(per_df["structure_match_top5"].sum())
    n_sg = int(per_df["spacegroup_match_top5"].sum())
    n_cs = int(per_df["consensus_top5"].sum())

    # DiffCSP headline metrics: any-of-k match rate + mean normalized RMSE
    # (averaged over matched formulas only). None outside --diffcsp mode.
    n_anyk = mean_rmse = None
    if args.diffcsp:
        n_anyk = int(per_df["structure_match_anyk"].sum())
        rmse_vals = pd.to_numeric(per_df["rmse"], errors="coerce").dropna()
        mean_rmse = float(rmse_vals.mean()) if len(rmse_vals) else None

    # Diagnostics: how many formulas were actually evaluated vs. failed upstream
    n_evaluated = int((per_df["status"] == "ok").sum())
    n_no_folder = int((per_df["status"] == "no_folder").sum())
    n_no_candidates = int((per_df["status"] == "no_candidates").sum())
    n_no_gt = int((per_df["status"] == "no_gt").sum())
    n_energy_failed = int((per_df["status"] == "all_energy_failed").sum())

    summary = {
        "eval_unit": eval_unit,
        "n_formulas_total": n_total,
        "n_formulas_evaluated": n_evaluated,
        "n_no_folder": n_no_folder,
        "n_no_candidates": n_no_candidates,
        "n_no_gt": n_no_gt,
        "n_energy_failed": n_energy_failed,
        "structure_match_rate_top5": n_sm / n_total,
        "spacegroup_match_rate_top5": n_sg / n_total,
        "consensus_rate_top5": n_cs / n_total,
        "structure_matches": n_sm,
        "spacegroup_matches": n_sg,
        "consensus_matches": n_cs,
        "top_k": args.top_k,
        "protocol": {
            "mode": "diffcsp" if args.diffcsp else "cspbench",
            "eval_unit": eval_unit + (" (per test material_id, DiffCSP-faithful)"
                                      if args.per_material else " (per unique composition)"),
            "structure_matcher": matcher_desc,
            "spacegroup_analyzer": "symprec=0.1",
            "ranking": ("none (dummy energies; any-of-k matching only)"
                        if args.ranker == "none" else
                        f"{args.ranker.upper()} energy per atom, ascending (lowest = best)"),
            "sg_labelling": "skipped" if args.skip_sg else "on",
            "selection": args.selection,
            "symmetrize": args.symmetrize + (f" (ladder={list(SYMPREC_LADDER)})"
                                             if args.symmetrize == "multi" else ""),
            "denominator": f"total {eval_unit}s in input CSV (missing folders count as failures)",
            "candidate_pool": (
                "own candidates only (<material_id>_sample_*.cif; each material scored at exactly k)"
                if (args.per_material and args.own_candidates_only) else
                "pooled per formula folder (polymorphs of one composition share candidates, "
                "so some materials are scored against more than k)"
            ),
        },
    }
    if args.diffcsp:
        # any-of-k match rate over all candidates (the DiffCSP CSP metric), plus
        # mean RMSE over matched rows. top-k rates above are kept for reference.
        summary["diffcsp_match_rate"] = n_anyk / n_total
        summary["diffcsp_matches"] = n_anyk
        summary["diffcsp_mean_rmse"] = mean_rmse
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    unit_cap = eval_unit.capitalize() + "s"  # "Materials" / "Formulas"
    print("\n" + "=" * 60)
    print(f"  Top-{args.top_k} Evaluation Summary  (per {eval_unit})")
    print("=" * 60)
    print(f"  {unit_cap} in CSV:          {n_total}")
    print(f"  {unit_cap} evaluated:       {n_evaluated}")
    if n_no_folder:
        print(f"  Missing folders:           {n_no_folder}  (counted as failures)")
    if n_no_candidates:
        print(f"  Empty folders:             {n_no_candidates}  (counted as failures)")
    if n_no_gt:
        print(f"  Missing GT:                {n_no_gt}  (counted as failures)")
    if n_energy_failed:
        print(f"  Energy-prediction failed:  {n_energy_failed}  (counted as failures)")
    print(f"  Structure match rate:      {n_sm}/{n_total} = {100 * n_sm / n_total:.2f}%  (top-{args.top_k} by energy)")
    print(f"  Spacegroup match rate:     {n_sg}/{n_total} = {100 * n_sg / n_total:.2f}%")
    print(f"  Consensus rate (SM & SG):  {n_cs}/{n_total} = {100 * n_cs / n_total:.2f}%")
    if args.diffcsp:
        print(f"  DiffCSP match rate:        {n_anyk}/{n_total} = {100 * n_anyk / n_total:.2f}%  (any-of-k, all candidates)")
        if mean_rmse is not None:
            print(f"  DiffCSP mean RMSE:         {mean_rmse:.4f}  (over {n_anyk} matched {eval_unit}s)")
    print("=" * 60)
    print(f"\nSummary JSON: {summary_path}")
    print(f"Top-{args.top_k} CIFs:   {top5_dir}/")


if __name__ == "__main__":
    main()
