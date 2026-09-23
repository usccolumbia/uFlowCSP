#!/usr/bin/env python3
"""
Create a folder of ground-truth CIF files for a CSP test split, named by
``<material_id>.cif`` (the naming ``select_top5_and_evaluate.py --per-material``
expects).

Three modes:

1) MP-20 (default, no args) -- reads the cif strings straight out of the
   shipped raw CSV, which is the only MP-20 file this repository carries:
     data/mp_20/raw/all.csv           -- 'material_id' + 'cif' columns (Git LFS)
     data/splits/difCSP_test_ids.json -- the 9046 test material_ids
     -> gt_cifs_test/<material_id>.cif

   That output directory is what slurm/submit_seed_replicates.sh and
   select_top5_and_evaluate.py --per-material expect. Takes a few minutes:
   all.csv is ~135 MB and every cif is written as its own file.

2) Benchmark (``--dataset perov|mpts``) -- same CSV path, for a prepared
   benchmark written by prepare_benchmark_data.py:
     data/<ds>/raw/all.csv         + data/splits/<ds>_test_ids.json
     -> data/gt_cifs_test_<ds>/<material_id>.cif

3) Legacy (``--from_pt``) -- reads cif strings from ``data/mp_20/raw/all.pt``
   and maps material_id -> index via all.csv. That .pt file is NOT part of this
   repository; the mode is kept only for older checkouts that still have one.

Examples:
    python make_gt_cifs.py                       # mp20 -> gt_cifs_test/
    python make_gt_cifs.py --dataset perov       # perov (paths inferred)
    python make_gt_cifs.py --all_csv X.csv --test_ids Y.json --out_dir Z/  # generic
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def _load_test_ids(ids_path: Path):
    with open(ids_path) as f:
        return set(str(x).strip() for x in json.load(f))


def _check_not_lfs_pointer(path: Path):
    """data/mp_20/raw/all.csv is stored in Git LFS. A clone made without LFS
    leaves a ~130-byte text pointer in its place, which otherwise fails much
    later as a missing-column or empty-dataset error."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. For MP-20 this file ships with the repo via Git "
            f"LFS — run `git lfs install && git lfs pull`."
        )
    if path.stat().st_size < 1024:
        with open(path, "r", errors="ignore") as f:
            head = f.read(64)
        if head.startswith("version https://git-lfs"):
            raise SystemExit(
                f"{path} is a Git LFS pointer, not the real file "
                f"({path.stat().st_size} bytes). Run:\n"
                f"    git lfs install && git lfs pull"
            )


def write_from_csv(all_csv: Path, test_ids_path: Path, out_dir: Path):
    """Write GT cifs from a CSV that already has material_id + cif columns."""
    _check_not_lfs_pointer(all_csv)
    out_dir.mkdir(parents=True, exist_ok=True)
    test_ids = _load_test_ids(test_ids_path)
    print(f"Reading cifs from {all_csv} ...")
    # The C parser handles the multi-line quoted cif fields and is minutes
    # faster on the 135 MB MP-20 csv; the python engine is the fallback for
    # CSVs it chokes on.
    try:
        df = pd.read_csv(all_csv)
    except Exception:
        df = pd.read_csv(all_csv, engine="python")
    cols = {c.lower(): c for c in df.columns}
    if "material_id" not in cols or "cif" not in cols:
        raise ValueError(
            f"{all_csv} must have material_id + cif columns (has {list(df.columns)})"
        )
    id_col, cif_col = cols["material_id"], cols["cif"]
    print(f"  {len(df)} rows; {len(test_ids)} test ids")

    found, missing = 0, 0
    seen = set()
    for _, row in df.iterrows():
        mid = str(row[id_col]).strip()
        if mid not in test_ids or mid in seen:
            continue
        seen.add(mid)
        (out_dir / f"{mid}.cif").write_text(str(row[cif_col]))
        found += 1
    missing = len(test_ids) - found
    print(f"\nDone. Wrote {found} CIFs to {out_dir}/")
    if missing:
        print(f"  Warning: {missing} test ids not found in {all_csv}")


def write_from_pt(pt_path: Path, csv_path: Path, test_ids_path: Path, out_dir: Path):
    """Legacy MP-20 path: cif strings live in all.pt, aligned with all.csv rows."""
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"  {len(df)} rows")
    print(f"Loading {pt_path} ...")
    all_data = torch.load(pt_path, map_location="cpu")
    print(f"  {len(all_data)} entries")
    assert len(df) == len(all_data), (
        f"Mismatch: CSV has {len(df)} rows but all.pt has {len(all_data)} entries"
    )
    test_ids = _load_test_ids(test_ids_path)
    print(f"  {len(test_ids)} test IDs")

    id_to_idx = {str(mid): i for i, mid in enumerate(df["material_id"])}
    found, missing = 0, 0
    for mid in test_ids:
        idx = id_to_idx.get(mid)
        if idx is None:
            missing += 1
            continue
        (out_dir / f"{mid}.cif").write_text(all_data[idx]["cif"])
        found += 1
    print(f"\nDone. Written {found} CIFs to {out_dir}/")
    if missing:
        print(f"  Warning: {missing} test IDs not found in all.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=None,
                    help="Benchmark name (perov|mpts). Infers CSV/test-ids/out paths. "
                         "Omit for MP-20.")
    ap.add_argument("--all_csv", default=None,
                    help="Generic CSV-cif mode: CSV with material_id + cif columns.")
    ap.add_argument("--test_ids", default=None, help="test ids JSON.")
    ap.add_argument("--out_dir", default=None, help="output dir for <material_id>.cif.")
    # Legacy .pt mode: that file is not shipped, so it must be asked for.
    ap.add_argument("--from_pt", action="store_true",
                    help="Read cifs from --pt_path instead of the raw CSV "
                         "(old checkouts only; all.pt is not in this repo).")
    ap.add_argument("--pt_path", default="data/mp_20/raw/all.pt")
    ap.add_argument("--csv_path", default="data/mp_20/raw/all.csv")
    args = ap.parse_args()

    if args.dataset:
        all_csv = Path(args.all_csv or f"data/{args.dataset}/raw/all.csv")
        test_ids = Path(args.test_ids or f"data/splits/{args.dataset}_test_ids.json")
        out_dir = Path(args.out_dir or f"data/gt_cifs_test_{args.dataset}")
        write_from_csv(all_csv, test_ids, out_dir)
    elif args.all_csv:
        if not args.test_ids or not args.out_dir:
            ap.error("--all_csv mode requires --test_ids and --out_dir")
        write_from_csv(Path(args.all_csv), Path(args.test_ids), Path(args.out_dir))
    elif args.from_pt:
        write_from_pt(
            Path(args.pt_path),
            Path(args.csv_path),
            Path(args.test_ids or "data/splits/difCSP_test_ids.json"),
            Path(args.out_dir or "gt_cifs_test"),
        )
    else:
        # Default MP-20 path: the shipped raw CSV already carries the cifs.
        write_from_csv(
            Path(args.csv_path),
            Path(args.test_ids or "data/splits/difCSP_test_ids.json"),
            Path(args.out_dir or "gt_cifs_test"),
        )


if __name__ == "__main__":
    main()
