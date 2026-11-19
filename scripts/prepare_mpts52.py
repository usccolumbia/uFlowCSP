#!/usr/bin/env python3
"""Build the MPTS-52 inputs from DiffCSP's canonical split CSVs.

Get the CSVs once (27,380 / 5,000 / 8,096 structures):
    git clone --depth 1 https://github.com/jiaor17/DiffCSP.git /tmp/DiffCSP      # data/mpts_52/{train,val,test}.csv

    python scripts/prepare_mpts52.py --diffcsp_dir /tmp/DiffCSP/data/mpts_52

Writes (relative to the repository root):
    data/mpts/raw/all.csv                    every structure (material_id, cif, ...), read by the datamodule
    data/splits/mpts_{train,val,test}_ids.json   the split, as the datamodule expects for split_type=mpts
    gt_cifs_mpts52_test/<material_id>.cif    reference structures of the 8,096 test materials
    data/splits/mpts52_test_knownz.csv       test targets: material_id, primitive_formula (primitive-cell contents,
                                             the DiffCSP / CrystalFlow known-Z convention), spacegroup
"""
import argparse
import json
import os
import warnings
from multiprocessing import Pool

import pandas as pd

warnings.filterwarnings("ignore")
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

SPLITS = ["train", "val", "test"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def target_row(args):
    mid, cif = args
    s = Structure.from_str(cif, fmt="cif")
    prim = s.get_primitive_structure()
    try:
        sg = SpacegroupAnalyzer(s, symprec=0.1).get_space_group_number()
    except Exception:
        sg = None
    return dict(material_id=mid, primitive_formula=prim.composition.formula.replace(" ", ""), spacegroup=sg)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--diffcsp_dir", required=True, help="folder with DiffCSP's mpts_52 train.csv, val.csv, test.csv")
    ap.add_argument("--nproc", type=int, default=os.cpu_count())
    args = ap.parse_args()

    dfs = {k: pd.read_csv(os.path.join(args.diffcsp_dir, f"{k}.csv")) for k in SPLITS}
    for k, d in dfs.items():
        assert d.material_id.is_unique, k
    allx = pd.concat([dfs[k] for k in SPLITS], ignore_index=True)
    assert allx.material_id.is_unique, "material_id collides across splits"
    print({k: len(d) for k, d in dfs.items()}, "total", len(allx), flush=True)

    os.makedirs(f"{ROOT}/data/mpts/raw", exist_ok=True)
    os.makedirs(f"{ROOT}/data/splits", exist_ok=True)
    cols = [c for c in ["material_id", "formation_energy_per_atom", "energy_above_hull", "year", "cif"] if c in allx]
    allx[cols].to_csv(f"{ROOT}/data/mpts/raw/all.csv")
    for k in SPLITS:
        with open(f"{ROOT}/data/splits/mpts_{k}_ids.json", "w") as f:
            json.dump(dfs[k].material_id.astype(str).tolist(), f)

    test = dfs["test"]
    gt = f"{ROOT}/gt_cifs_mpts52_test"
    os.makedirs(gt, exist_ok=True)
    for mid, cif in zip(test.material_id, test.cif):
        with open(f"{gt}/{mid}.cif", "w") as f:
            f.write(str(cif))
    with Pool(args.nproc) as p:
        rows = p.map(target_row, list(zip(test.material_id, test.cif)), chunksize=16)
    pd.DataFrame(rows).to_csv(f"{ROOT}/data/splits/mpts52_test_knownz.csv", index=False)
    print(f"wrote data/mpts/raw/all.csv, data/splits/mpts_*_ids.json, {gt}/ ({len(test)} cifs), "
          f"data/splits/mpts52_test_knownz.csv", flush=True)


if __name__ == "__main__":
    main()
