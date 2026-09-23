import argparse
import inspect
import traceback
from datetime import datetime
from pathlib import Path

import lightning as L
import pandas as pd
import rootutils
import torch
from pymatgen.core import Composition

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.eval.crystal import structure_validity
from src.eval.crystal_generation import array_dict_to_crystal
from src.models.constraint_guidance import REFINE_MODES, build_refiner
from src.models.imeanflow_raw_transport import IMeanFlowRawTransport
from src.models.meanflow_transport import MeanFlowTransport
from src.models.meanflow_raw_module import MeanFlowRawLitModule
from src.models.meanflow_csp_module import MeanFlowCSPLitModule

try:
    from chgnet.model import CHGNet, StructOptimizer
    CHGNET_AVAILABLE = True
except ImportError:
    CHGNET_AVAILABLE = False

try:
    import orb_models.forcefield.pretrained  # noqa: F401
    ORB_AVAILABLE = True
    ORB_IMPORT_ERROR = None
except ImportError as _orb_exc:
    # Keep the real cause: a missing sub-dependency of orb-models (cached_path,
    # for instance) fails here exactly like orb-models being absent, and
    # reporting "not installed" then sends people down the wrong path.
    ORB_AVAILABLE = False
    ORB_IMPORT_ERROR = _orb_exc


class OrbRelaxer:
    """ASE-based relaxer with a CHGNet-``StructOptimizer``-like interface.

    Exposes ``.relax(structure, fmax, steps, verbose) -> {"final_structure": Structure}``
    so it is a drop-in replacement for the CHGNet relaxer used below. Relaxes both
    atomic positions and the cell (via a cell filter), matching CHGNet's default
    ``StructOptimizer`` behaviour. Uses an ORB v3 foundation potential.
    """

    def __init__(self, device="cpu", model_name="orb_v3_conservative_inf_omat",
                 precision="float32-high"):
        from orb_models.forcefield import pretrained
        try:
            # orb-models >= 0.7
            from orb_models.forcefield.inference.calculator import ORBCalculator
        except ImportError:
            # older orb-models (<= 0.6)
            from orb_models.forcefield.calculator import ORBCalculator
        from pymatgen.io.ase import AseAtomsAdaptor

        if not hasattr(pretrained, model_name):
            avail = [n for n in dir(pretrained) if n.startswith("orb_")]
            raise ValueError(
                f"Unknown ORB model '{model_name}'. Available: {avail}"
            )
        loader = getattr(pretrained, model_name)
        try:
            loaded = loader(device=device, precision=precision)
        except TypeError:
            # Older orb-models signatures don't accept `precision`.
            loaded = loader(device=device)

        # orb-models >= 0.7 returns (model, atoms_adapter); earlier returns the model.
        atoms_adapter = None
        if isinstance(loaded, tuple):
            orbff, atoms_adapter = loaded[0], loaded[1]
        else:
            orbff = loaded

        try:
            if atoms_adapter is not None:
                self._calc = ORBCalculator(orbff, atoms_adapter=atoms_adapter, device=device)
            else:
                self._calc = ORBCalculator(orbff, device=device)
        except TypeError:
            # Calculator signature without an atoms_adapter kwarg.
            self._calc = ORBCalculator(orbff, device=device)
        self._adaptor = AseAtomsAdaptor()

    def relax(self, structure, fmax=0.1, steps=200, verbose=False):
        from ase.optimize import FIRE
        try:
            from ase.filters import FrechetCellFilter as _CellFilter
        except ImportError:  # older ASE
            from ase.constraints import ExpCellFilter as _CellFilter

        atoms = self._adaptor.get_atoms(structure)
        atoms.calc = self._calc
        dyn = FIRE(_CellFilter(atoms), logfile="-" if verbose else None)
        dyn.run(fmax=fmax, steps=steps)
        final_structure = self._adaptor.get_structure(atoms)
        return {"final_structure": final_structure}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate raw MeanFlow samples for all formulas in a CSV "
                    "and optionally relax them with CHGNet. No GT matching is "
                    "done here -- use select_top5_and_evaluate.py downstream."
    )
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--num_samples_per_formula", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--material_id_col", default="material_id")
    parser.add_argument("--formula_col", default="primitive_formula")
    parser.add_argument("--spacegroup_col", default="spacegroup")
    parser.add_argument("--save_generated_dir", default="",
                        help="Parent directory for saving generated structures. "
                             "If empty, defaults to results_generate/generated_struc_<DDMMM>. "
                             "CIFs are saved as <dir>/<formula>/sample_<idx>.cif")
    # Spacegroup conditioning is OFF BY DEFAULT. Feeding the CSV's ground-truth
    # space group at inference leaks symmetry information and is not a
    # formula-only evaluation -- see the denoiser docstring. Opt in explicitly
    # with --use_spacegroup if you actually want a symmetry-conditioned run.
    parser.add_argument("--use_spacegroup", action="store_true",
                        help="Feed the CSV's ground-truth space group to the model. "
                             "OFF by default. This LEAKS symmetry information and is "
                             "NOT a legitimate formula-only evaluation -- use it only "
                             "for deliberate symmetry-conditioned experiments.")
    parser.add_argument("--disable_spacegroup", action="store_true",
                        help="Deprecated no-op: spacegroup conditioning is already off "
                             "by default. Accepted so existing scripts keep working.")
    parser.add_argument("--use_comp_validity", action="store_true",
                        help="Also require SMACT composition validity (in addition to structural validity) before saving a sample. Off by default.")
    parser.add_argument("--num_sampling_steps", type=int, default=None, help="Override sampling.num_sampling_steps from the checkpoint hparams.")
    parser.add_argument("--noise_scale", type=float, default=1.0,
                        help="Sampling-diversity temperature for the prior noise "
                             "(default 1.0 = training prior). >1 widens only the "
                             "Gaussian part of the prior (lattice for CSP models); "
                             "fractional coords stay U[0,1).")
    parser.add_argument("--material_prefix", action="store_true",
                        help="Prefix saved CIF filenames with the row's material_id "
                             "(<mid>_sample_<i>.cif). Required for known-Z CSVs where "
                             "multiple rows share a reduced formula (different cell "
                             "sizes) and would otherwise overwrite each other. "
                             "Off by default = existing filenames, unchanged.")
    parser.add_argument("--time_trials", type=int, default=0,
                        help="If >0, run a TIMING benchmark instead of saving: time "
                             "only the sampling calls over this many materials (after "
                             "--time_warmup warm-up materials), print ms/structure and "
                             "min/10k structures, and exit. No CIFs written.")
    parser.add_argument("--time_warmup", type=int, default=3,
                        help="Warm-up materials excluded from timing (CUDA lazy init).")
    parser.add_argument("--enumerate_crystal_systems", action="store_true",
                        help="Formula-only crystal-system enumeration: split the "
                             "num_samples_per_formula budget evenly across the 7 "
                             "crystal systems (one representative SG per system; "
                             "the 'system' denoiser variant only keeps the coarse "
                             "7-class token). The GT spacegroup from the CSV is "
                             "NOT used. Overrides --use_spacegroup. Off by "
                             "default = existing behaviour, unchanged.")
    parser.add_argument("--relax", action="store_true",
                        help="Relax each valid generated structure before saving.")
    # ORB v3 is the relaxer the reported runs used (the campaign launchers all
    # default RELAXER=orb). CHGNet remains available, and is still the default
    # energy RANKER in select_top5_and_evaluate.py -- ranking and relaxation are
    # separate choices, do not conflate them.
    parser.add_argument("--relaxer", type=str, default="orb", choices=["chgnet", "orb"],
                        help="Which foundation potential to relax with (default: orb).")
    parser.add_argument("--orb_model", type=str, default="orb_v3_conservative_inf_omat",
                        help="ORB pretrained model name (only used when --relaxer orb).")
    parser.add_argument("--relax_device", type=str, default="auto", choices=["auto", "cuda", "cpu", "mps"],
                        help="Device for the relaxer. 'auto' picks cuda/mps/cpu.")
    parser.add_argument("--relax_fmax", type=float, default=0.1,
                        help="CHGNet relaxation force convergence threshold (eV/A).")
    parser.add_argument("--relax_steps", type=int, default=200,
                        help="Max CHGNet relaxation steps per structure.")
    parser.add_argument("--max_relax_per_formula", type=int, default=0,
                        help="If >0, cap the number of valid samples relaxed/saved per formula.")
    # -- training-free constraint refinement (src/models/constraint_guidance.py)
    parser.add_argument("--refine", type=str, default="none", choices=list(REFINE_MODES),
                        help="Post-sampling structure refinement. 'guidance' relieves "
                             "interatomic overlap by gradient descent on a differentiable "
                             "constraint energy with valence-aware radii; 'symmetry' snaps "
                             "a nearly-symmetric cell onto the exact one; 'both' runs "
                             "guidance then symmetry. Default 'none' = untouched behaviour.")
    parser.add_argument("--refine_steps", type=int, default=150,
                        help="Max gradient steps per structure for --refine guidance.")
    parser.add_argument("--refine_lr", type=float, default=0.01,
                        help="Adam step size for --refine guidance.")
    parser.add_argument("--refine_radius_scale", type=float, default=0.60,
                        help="Pair distance target = scale * (r_i + r_j).")
    parser.add_argument("--refine_distance_floor", type=float, default=0.90,
                        help="Absolute floor (Ang) on the pair distance target. Must stay "
                             "above the 0.5 Ang structure_validity cutoff.")
    parser.add_argument("--refine_no_valence", action="store_true",
                        help="Use covalent radii only, ignoring oxidation states.")
    parser.add_argument("--refine_no_scale", action="store_true",
                        help="Hold the cell fixed; move fractional coordinates only.")

    return parser.parse_args()


def normalize_formula(formula: str) -> str:
    return Composition(formula).reduced_formula.replace(" ", "")


# One representative spacegroup per crystal system (first SG of each range).
# The 'system' denoiser derives the coarse 7-class crystal-system token from the
# SG number and discards the fine SG, so any SG inside the range is equivalent.
# triclinic 1-2, monoclinic 3-15, orthorhombic 16-74, tetragonal 75-142,
# trigonal 143-167, hexagonal 168-194, cubic 195-230.
CRYSTAL_SYSTEM_REP_SGS = (1, 3, 16, 75, 143, 168, 195)


def generate_enumerated_over_systems(model, formula, num_samples, batch_size, noise_scale):
    """Generate ``num_samples`` split evenly across the 7 crystal systems.

    Formula-only legitimate: the GT spacegroup is never used — all 7 systems are
    tried and downstream any-of-k matching / energy ranking picks. sample_idx is
    offset per system so CIF filenames never collide.
    """
    import math
    per_sys = max(1, math.ceil(num_samples / len(CRYSTAL_SYSTEM_REP_SGS)))
    out = []
    for sys_i, rep_sg in enumerate(CRYSTAL_SYSTEM_REP_SGS):
        arrs = model.generate_conditioned_samples(
            target_formula=formula,
            target_spacegroup=rep_sg,
            num_samples=per_sys,
            batch_size=batch_size,
            noise_scale=noise_scale,
        )
        for a in arrs:
            a["sample_idx"] = int(a["sample_idx"]) + sys_i * per_sys
        out.extend(arrs)
    return out


def replace_legacy_transport_if_needed(model: MeanFlowRawLitModule) -> None:
    transport = model.meanflow
    module_name = type(transport).__module__
    if not module_name.startswith("src.models.transport."):
        return

    common_kwargs = {
        "flow_ratio": getattr(transport, "flow_ratio", 0.50),
        "time_dist": getattr(transport, "time_dist", ["lognorm", -0.4, 1.0]),
        "cfg_ratio": getattr(transport, "cfg_ratio", 0.10),
        "cfg_scale": getattr(transport, "cfg_scale", 2.0),
        "jvp_api": getattr(transport, "jvp_api", "funtorch"),
        "atom_loss_weight": getattr(transport, "atom_loss_weight", 0.0),
    }
    class_name = type(transport).__name__.lower()
    if "imean" in class_name:
        model.meanflow = IMeanFlowRawTransport(
            **common_kwargs,
            detach_dudt_in_loss=getattr(transport, "detach_dudt_in_loss", False),
            coord_loss_weight=getattr(transport, "coord_loss_weight", 1.0),
            lattice_loss_weight=getattr(transport, "lattice_loss_weight", 1.0),
            coord_dim=getattr(transport, "coord_dim", 0),
            lattice_dim=getattr(transport, "lattice_dim", 0),
        )
    else:
        model.meanflow = MeanFlowTransport(
            **common_kwargs,
            proj_lattice=getattr(transport, "proj_lattice", False),
            proj_coords_rotavg=getattr(transport, "proj_coords_rotavg", False),
            proj_coords_anchor=getattr(transport, "proj_coords_anchor", False),
            sym_loss_weight=getattr(transport, "sym_loss_weight", 0.0),
            sym_coord_dim=getattr(transport, "sym_coord_dim", 3),
            sym_lattice_dim=getattr(transport, "sym_lattice_dim", 9),
            sym_coords_are_fractional=getattr(transport, "sym_coords_are_fractional", False),
            # Must be carried over: dropping these silently samples a torus
            # checkpoint from a Gaussian prior with no periodic wrapping.
            torus_coords=getattr(transport, "torus_coords", False),
            torus_coord_dim=getattr(transport, "torus_coord_dim", 3),
            coord_loss_weight=getattr(transport, "coord_loss_weight", 1.0),
            lattice_loss_weight=getattr(transport, "lattice_loss_weight", 1.0),
            coord_dim=getattr(transport, "coord_dim", 0),
            lattice_dim=getattr(transport, "lattice_dim", 0),
            # Must be carried over for the same reason as the torus flags:
            # dropping them samples a per-structure-lattice checkpoint from the
            # per-token prior, so the lattice the decode pools back is ~0.
            # Legacy transports have none of these -> False/defaults -> the
            # constructor's own defaults, i.e. no behaviour change for them.
            lattice_per_structure=getattr(transport, "lattice_per_structure", False),
            lattice_prior_sigma=getattr(transport, "lattice_prior_sigma", 1.0),
            lattice_prior_mean=getattr(transport, "lattice_prior_mean", None),
        )
    print(
        "Replaced legacy checkpoint transport "
        f"{module_name}.{type(transport).__name__} with "
        f"{type(model.meanflow).__module__}.{type(model.meanflow).__name__}."
    )


def main():
    args = parse_args()

    # Always state the conditioning mode. Feeding the GT space group silently is
    # the failure mode that makes a run look better than it is, so it is never
    # implicit: formula-only is the default and the leaky mode announces itself.
    if args.enumerate_crystal_systems:
        print("Conditioning:  formula + ENUMERATED crystal systems (GT spacegroup NOT used)")
    elif args.use_spacegroup:
        print("=" * 78)
        print("WARNING: spacegroup conditioning is ON (--use_spacegroup).")
        print("The ground-truth space group from the CSV is being fed to the model.")
        print("This LEAKS symmetry information -- results are NOT a formula-only")
        print("evaluation and are NOT comparable to DiffCSP / CrystalFlow numbers.")
        print("=" * 78)
    else:
        print("Conditioning:  formula only (GT spacegroup NOT used)")

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if args.num_samples_per_formula <= 0:
        raise ValueError("num_samples_per_formula must be positive")

    if args.save_generated_dir:
        gen_base_dir = Path(args.save_generated_dir)
    else:
        date_str = datetime.now().strftime("%d%b")
        gen_base_dir = Path("results_generate") / f"generated_struc_{date_str}"
    gen_base_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    L.seed_everything(29, workers=True)

    print(f"Loading checkpoint: {args.ckpt_path}")
    model = MeanFlowRawLitModule.load_from_checkpoint(args.ckpt_path, map_location=device, strict=False)
    # CSP checkpoints carry a MeanFlowCSPTransport + 6-D polar-lattice geometry,
    # which needs MeanFlowCSPLitModule's encode/decode (the base module would
    # misread the 6 polar-lattice channels as a flat 3x3 cell and crash). That
    # subclass adds no instance state -- only geometry-aware method/property
    # overrides -- so we can rebless the already-loaded base instance in place.
    if (
        type(model.meanflow).__name__ == "MeanFlowCSPTransport"
        and not isinstance(model, MeanFlowCSPLitModule)
    ):
        model.__class__ = MeanFlowCSPLitModule
        print("Detected CSP transport -> using MeanFlowCSPLitModule geometry (encode/decode).")
    replace_legacy_transport_if_needed(model)
    # Hard-patch attributes that were added after old checkpoints were saved.
    # Using vars() / direct __dict__ write bypasses all class attribute
    # machinery (pickle, __getattr__, __setstate__) so this always works.
    _transport_compat_defaults = {
        "torus_coords": False,
        "torus_coord_dim": 3,
        "proj_lattice": False,
        "proj_coords_rotavg": False,
        "proj_coords_anchor": False,
        "sym_loss_weight": 0.0,
        "sym_coord_dim": 0,
        "sym_lattice_dim": 0,
        "sym_coords_are_fractional": False,
        "atom_loss_weight": 0.0,
        "coord_loss_weight": 1.0,
        "lattice_loss_weight": 1.0,
        "coord_dim": 0,
        "lattice_dim": 0,
        # Per-structure lattice geometry: absent on checkpoints saved before it
        # existed, where False is the correct reconstruction.
        "lattice_per_structure": False,
        "lattice_prior_sigma": 1.0,
        "lattice_prior_mean": None,
        "last_lattice_loss": None,
        "last_sample_atom_types": None,
        "last_sample_mask": None,
        "last_atom_ce": None,
        "last_sym_loss": None,
    }
    _transport_state = vars(model.meanflow)
    for _k, _v in _transport_compat_defaults.items():
        if _k not in _transport_state:
            _transport_state[_k] = _v
            print(f"  [compat] patched missing transport attr: {_k} = {_v!r}")
    print(f"Loaded module class: {type(model).__module__}.{type(model).__name__}")
    print(f"Denoiser class: {type(model.ema_denoiser).__module__}.{type(model.ema_denoiser).__name__}")
    print(f"Denoiser source: {inspect.getfile(type(model.ema_denoiser))}")
    print(f"MeanFlow class: {type(model.meanflow).__module__}.{type(model.meanflow).__name__}")
    print(f"MeanFlow source: {inspect.getfile(type(model.meanflow))}")
    if args.num_sampling_steps is not None:
        if args.num_sampling_steps <= 0:
            raise ValueError("num_sampling_steps must be positive")
        if "sampling" not in model.hparams or model.hparams.sampling is None:
            model.hparams.sampling = {}
        model.hparams.sampling["num_sampling_steps"] = int(args.num_sampling_steps)
        print(f"Overriding num_sampling_steps -> {args.num_sampling_steps}")
    if args.noise_scale != 1.0:
        print(f"Sampling noise_scale (diversity temperature): {args.noise_scale}")
    if args.enumerate_crystal_systems:
        print(f"Crystal-system ENUMERATION on: {args.num_samples_per_formula} samples "
              f"split over 7 systems (rep SGs {CRYSTAL_SYSTEM_REP_SGS}); "
              "CSV spacegroup ignored (formula-only).")
    model = model.to(device)
    model.eval()

    refiner = build_refiner(
        args.refine,
        steps=args.refine_steps,
        lr=args.refine_lr,
        radius_scale=args.refine_radius_scale,
        distance_floor=args.refine_distance_floor,
        use_valence=not args.refine_no_valence,
        allow_scale=not args.refine_no_scale,
    )
    if refiner is not None:
        print(f"Constraint refinement ENABLED: mode={args.refine} "
              f"steps={args.refine_steps} lr={args.refine_lr} "
              f"radius_scale={args.refine_radius_scale} "
              f"floor={args.refine_distance_floor} "
              f"valence={not args.refine_no_valence} scale={not args.refine_no_scale}")

    relaxer = None
    if args.relax:
        relax_device = args.relax_device
        if relax_device == "auto":
            if torch.cuda.is_available():
                relax_device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                relax_device = "mps"
            else:
                relax_device = "cpu"

        if args.relaxer == "orb":
            if not ORB_AVAILABLE:
                raise ImportError("\n".join([
                    f"could not import orb_models.forcefield.pretrained: {ORB_IMPORT_ERROR}",
                    "  If orb-models itself is missing:  pip install --no-deps orb-models==0.5.5",
                    "  If a DEPENDENCY is missing (the error above names it), install that",
                    "  package directly -- orb-models declares torch>=2.6, so resolving its",
                    "  dependencies normally would replace your torch build.",
                    "  Or relax with CHGNet instead:  --relaxer chgnet",
                ]))
            print(f"Loading ORB relaxer '{args.orb_model}' on device={relax_device} ...")
            relaxer = OrbRelaxer(device=relax_device, model_name=args.orb_model)
            print(f"ORB relaxer ready (model={args.orb_model}, fmax={args.relax_fmax}, "
                  f"steps={args.relax_steps}, max_relax_per_formula={args.max_relax_per_formula or 'all'}).")
        else:
            if not CHGNET_AVAILABLE:
                raise ImportError("CHGNet is not installed. Install with 'pip install chgnet' to use --relaxer chgnet.")
            print(f"Loading CHGNet relaxer on device={relax_device} ...")
            chgnet_model = CHGNet.load(use_device=relax_device)
            relaxer = StructOptimizer(model=chgnet_model, use_device=relax_device)
            print(f"CHGNet relaxer ready (fmax={args.relax_fmax}, steps={args.relax_steps}, "
                  f"max_relax_per_formula={args.max_relax_per_formula or 'all'}).")

    df = pd.read_csv(csv_path)

    # -- Timing benchmark (min/10k structures), matching CrystalFlow's 't' column --
    if args.time_trials > 0:
        import time as _time

        def _gen(formula, sg):
            if args.enumerate_crystal_systems:
                return generate_enumerated_over_systems(
                    model, formula, args.num_samples_per_formula,
                    args.batch_size, args.noise_scale)
            target_sg = sg if args.use_spacegroup else None
            return model.generate_conditioned_samples(
                target_formula=formula, target_spacegroup=target_sg,
                num_samples=args.num_samples_per_formula,
                batch_size=args.batch_size, noise_scale=args.noise_scale)

        rows = [r for _, r in df.iterrows()][: args.time_warmup + args.time_trials]
        def _sg(r):
            v = r.get(args.spacegroup_col, 1)
            return int(v) if not pd.isna(v) else 1

        for r in rows[: args.time_warmup]:          # warm-up (untimed)
            f = str(r.get(args.formula_col, "")).strip()
            if f:
                _gen(f, _sg(r))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = _time.perf_counter()
        n_struct = 0
        for r in rows[args.time_warmup:]:           # timed
            f = str(r.get(args.formula_col, "")).strip()
            if not f:
                continue
            n_struct += len(_gen(f, _sg(r)))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = _time.perf_counter() - t0
        per = dt / max(n_struct, 1)
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        eff_S = int((model.hparams.get("sampling", {}) or {}).get("num_sampling_steps", 1))
        print("\n" + "=" * 60)
        print("  SAMPLING TIMING (pure model.sample, no I/O)")
        print("=" * 60)
        print(f"  device: {gpu}")
        print(f"  S={eff_S}  k={args.num_samples_per_formula}  batch={args.batch_size}")
        print(f"  {n_struct} structures over {len(rows) - args.time_warmup} materials in {dt:.2f}s")
        print(f"  {per * 1000:.2f} ms/structure")
        print(f"  => {per * 10000 / 60:.3f} min / 10k structures   (CrystalFlow t-column units)")
        print("=" * 60)
        return

    summary = []
    summary_csv = gen_base_dir / "generation_summary.csv"

    for row_idx, row in df.iterrows():
        material_id = str(row.get(args.material_id_col, f"row_{row_idx}"))
        formula = str(row.get(args.formula_col, "")).strip()
        spacegroup_raw = row.get(args.spacegroup_col, None)
        if not formula or pd.isna(formula):
            summary.append({
                "row_idx": row_idx, "material_id": material_id,
                "formula": "", "spacegroup": "",
                "generated_count": 0, "valid_count": 0, "saved_count": 0,
                "status": "missing_formula",
            })
            continue
        # The spacegroup column is only needed when it is actually consumed
        # (--use_spacegroup). A formula-only CSV -- just a `primitive_formula`
        # column -- is valid input and must not be skipped.
        if pd.isna(spacegroup_raw):
            if args.use_spacegroup:
                summary.append({
                    "row_idx": row_idx, "material_id": material_id,
                    "formula": formula, "spacegroup": "",
                    "generated_count": 0, "valid_count": 0, "saved_count": 0,
                    "status": "missing_spacegroup",
                })
                continue
            spacegroup = None
        else:
            spacegroup = int(spacegroup_raw)
        print(f"[{row_idx + 1}/{len(df)}] material_id={material_id} "
              f"formula={formula} "
              f"spacegroup={spacegroup if spacegroup is not None else '-'}")

        try:
            if args.enumerate_crystal_systems:
                # Formula-only enumeration over the 7 crystal systems; the CSV
                # spacegroup is deliberately ignored.
                generated_arrays = generate_enumerated_over_systems(
                    model,
                    formula,
                    num_samples=args.num_samples_per_formula,
                    batch_size=args.batch_size,
                    noise_scale=args.noise_scale,
                )
            else:
                target_sg = spacegroup if args.use_spacegroup else None
                generated_arrays = model.generate_conditioned_samples(
                    target_formula=formula,
                    target_spacegroup=target_sg,
                    num_samples=args.num_samples_per_formula,
                    batch_size=args.batch_size,
                    noise_scale=args.noise_scale,
                )
        except Exception as ex:
            print(f"  generation failed: {ex}")
            print(traceback.format_exc())
            summary.append({
                "row_idx": row_idx, "material_id": material_id,
                "formula": formula, "spacegroup": spacegroup,
                "generated_count": 0, "valid_count": 0, "saved_count": 0,
                "status": f"generation_failed:{type(ex).__name__}:{ex}",
            })
            continue

        formula_tag = normalize_formula(formula)
        formula_dir = gen_base_dir / formula_tag
        formula_dir.mkdir(parents=True, exist_ok=True)

        valid_count = 0
        saved_count = 0
        relaxed_count = 0

        for sample in generated_arrays:
            crys = array_dict_to_crystal(sample, save=False)
            if refiner is None:
                passes = crys.valid if args.use_comp_validity else crys.struct_valid
                if not passes:
                    continue
                structure_to_save = crys.structure
            else:
                # Refine BEFORE the validity gate, not after. An overlapping
                # sample is DISCARDED by that gate (structure_validity rejects
                # any pair closer than 0.5 Ang), and at k=1 a discarded sample
                # is an automatic miss -- rescuing it is the entire point.
                # Composition is invariant under refinement, so comp_valid
                # carries over; only the structural half is recomputed.
                if not getattr(crys, "constructed", False):
                    continue
                structure_to_save = refiner.refine(crys.structure)
                passes = structure_validity(structure_to_save)
                if args.use_comp_validity:
                    passes = passes and bool(crys.comp_valid)
                if not passes:
                    continue
            valid_count += 1

            if relaxer is not None:
                if args.max_relax_per_formula > 0 and relaxed_count >= args.max_relax_per_formula:
                    continue
                try:
                    relaxed = relaxer.relax(
                        structure_to_save,
                        fmax=args.relax_fmax,
                        steps=args.relax_steps,
                        verbose=False,
                    )
                    structure_to_save = relaxed["final_structure"]
                except Exception as ex:
                    print(f"  {args.relaxer} relax failed for sample {sample.get('sample_idx', '?')}: {ex} — saving unrelaxed.")
                relaxed_count += 1

            if args.material_prefix:
                import re as _re
                _tag = _re.sub(r"[^A-Za-z0-9_.-]+", "", material_id) or f"row{row_idx}"
                out_path = formula_dir / f"{_tag}_sample_{int(sample['sample_idx'])}.cif"
            else:
                out_path = formula_dir / f"sample_{int(sample['sample_idx'])}.cif"
            structure_to_save.to(filename=str(out_path))
            saved_count += 1

        summary.append({
            "row_idx": row_idx, "material_id": material_id,
            "formula": formula, "spacegroup": spacegroup,
            "generated_count": len(generated_arrays),
            "valid_count": valid_count, "saved_count": saved_count,
            "status": "ok",
        })
        print(f"  -> generated={len(generated_arrays)} valid={valid_count} "
              f"saved={saved_count} (relaxed={relaxed_count})")

        # Persist incrementally so partial runs are usable
        pd.DataFrame(summary).to_csv(summary_csv, index=False)

    pd.DataFrame(summary).to_csv(summary_csv, index=False)
    ok = sum(1 for r in summary if r["status"] == "ok")
    print(f"Finished. Formulas processed OK: {ok}/{len(summary)}")
    if refiner is not None:
        print(refiner.summary())
    print(f"Generated CIFs under: {gen_base_dir}")
    print(f"Summary written to: {summary_csv}")


if __name__ == "__main__":
    main()
