#!/usr/bin/env python3
"""
Build a KNOWN-Z evaluation CSV: replace each row's formula with the TRUE
primitive-cell contents of its ground-truth structure.

This reproduces the conditioning convention of the DiffCSP / CrystalFlow papers
(the model is told the exact atom content of the cell, e.g. "Ti2 O4" for rutile,
not the reduced "TiO2"), making results directly comparable to published
numbers. The formula-only CSV remains the stricter, more realistic protocol —
report both, clearly labelled.

Input : --csv     source eval CSV (material_id, primitive_formula, spacegroup)
        --gt_dir  GT cifs named <material_id>.cif
Output: --out     same rows, but primitive_formula = full primitive-cell
                  composition (e.g. "Ti2 O4"); rows whose GT cif is missing or
                  unparsable are dropped (reported).

Usage (cluster, allatom env):
    python make_knownz_csv.py \
        --csv data/splits/difCSP_test_subset500.csv \
        --gt_dir gt_cifs_test \
        --out data/splits/difCSP_test_subset500_knownz.csv
"""

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd
from pymatgen.core import Structure
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--material_id_col", default="material_id")
    ap.add_argument("--formula_col", default="primitive_formula")
    args = ap.parse_args()

    gt_dir = Path(args.gt_dir)
    df = pd.read_csv(args.csv)

    rows, n_missing, n_bad = [], 0, 0
    n_bigger = 0  # cells with more atoms than the reduced formula (the freed cases)
    for _, row in tqdm(df.iterrows(), total=len(df), desc="known-Z"):
        mid = str(row[args.material_id_col]).strip()
        cif = gt_dir / f"{mid}.cif"
        if not cif.exists():
            n_missing += 1
            continue
        try:
            s = Structure.from_file(str(cif))
            prim = s.get_primitive_structure()
            # Full cell contents in COMPACT form, e.g. "Ti2O4" (no spaces --
            # the MeanFlow formula parser's character guard rejects spaces).
            # Composition preserves the given counts, so Z is retained.
            cell_formula = prim.composition.formula.replace(" ", "")
            if prim.composition.num_atoms > prim.composition.reduced_composition.num_atoms:
                n_bigger += 1
        except Exception as e:
            n_bad += 1
            if n_bad <= 10:
                print(f"  WARN bad GT cif {mid}: {e}")
            continue
        new = dict(row)
        new[args.formula_col] = cell_formula
        rows.append(new)

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}: {len(out)} rows "
          f"(dropped: {n_missing} missing GT, {n_bad} bad cif)")
    print(f"cells with Z>1 reduced units (previously UNREACHABLE with the "
          f"formula-only protocol): {n_bigger}/{len(out)} = "
          f"{100 * n_bigger / max(len(out), 1):.1f}%")


if __name__ == "__main__":
    main()
