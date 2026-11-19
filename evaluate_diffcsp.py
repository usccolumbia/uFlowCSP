#!/usr/bin/env python3
"""Score generated structures with the DiffCSP / CrystalFlow CSP protocol (k = 1 and k = K).

For every target material, its own candidates (`<material_id>_sample_<i>.cif`, as written by
`src/uflowcsp.py --material_prefix`) are compared with the reference structure `<gt_dir>/<material_id>.cif`
using StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3); a candidate matches if `get_rms_dist` is not None.

Two numbers are reported for each k:
  filtered   -- the published protocol (DiffCSP compute_metrics.py, CrystalFlow README): a candidate counts only if it
                is valid (SMACT composition validity, structural validity, CrystalNN fingerprint computable); at k = 1
                the reference must be valid too. Targets whose formula fails SMACT are therefore counted as failures.
                Use these numbers to compare with published results.
  unfiltered -- structural validity only (no SMACT).
The denominator is always every target in --csv. RMSE is the mean, over matched targets, of the smallest
get_rms_dist among that target's matching candidates (k = 1: sample 0 only).

Sample 0 is the k = 1 candidate. Samples the generator did not save (structurally invalid) count as failures.

usage: python evaluate_diffcsp.py --gen_dir results/gen --gt_dir gt_cifs_test --csv data/splits/difCSP_test.csv \
           --output_dir results/eval [--k 20] [--nproc 16]
Writes <output_dir>/summary.json and per_sample.csv.
"""
import argparse
import glob
import json
import os
import re
import warnings
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pandas as pd
import rootutils

warnings.filterwarnings("ignore")
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure

from src.eval.crystal import smact_validity, structure_validity

MATCHER = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)
CRYSTALNN = CrystalNNFingerprint.from_preset("ops")


def validity(s):
    """(comp_valid, struct_valid, fp_ok) exactly as DiffCSP's Crystal class decides them."""
    cnt = Counter(site.specie.Z for site in s)
    elems = sorted(cnt)
    counts = np.array([cnt[e] for e in elems])
    counts = counts // np.gcd.reduce(counts)
    comp_valid = bool(smact_validity(tuple(elems), tuple(counts.tolist())))
    struct_valid = bool(s.volume >= 0.1 and structure_validity(s))
    try:
        for i in range(len(s)):
            CRYSTALNN.featurize(s, i)
        fp_ok = True
    except Exception:
        fp_ok = False
    return comp_valid, struct_valid, fp_ok


def rms(pred, gt):
    try:
        r = MATCHER.get_rms_dist(pred, gt)
        return None if r is None else float(r[0])
    except Exception:
        return None


def score_material(task):
    mid, gt_path, files = task
    gt = Structure.from_file(gt_path)
    g = validity(gt)
    rows = []
    for idx, path in files:
        try:
            s = Structure.from_file(path)
        except Exception:
            continue
        c = validity(s)
        rows.append(dict(material_id=mid, sample_idx=idx, comp_valid=c[0], struct_valid=c[1], fp_ok=c[2],
                         rms=rms(s, gt), gt_valid=all(g)))
    return mid, all(g), rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gen_dir", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--csv", required=True, help="target list with a material_id column (the denominator)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--k", type=int, default=20, help="candidates per target to score (sample_idx < k)")
    ap.add_argument("--nproc", type=int, default=os.cpu_count())
    args = ap.parse_args()

    mids = pd.read_csv(args.csv).material_id.astype(str).tolist()
    by_mid = {}
    for f in glob.glob(os.path.join(args.gen_dir, "**", "*_sample_*.cif"), recursive=True):
        m = re.match(r"(.+)_sample_(\d+)\.cif$", os.path.basename(f))
        if m and int(m.group(2)) < args.k:
            by_mid.setdefault(m.group(1), []).append((int(m.group(2)), f))
    tasks = []
    for mid in mids:
        gt = os.path.join(args.gt_dir, f"{mid}.cif")
        if not os.path.exists(gt):
            raise FileNotFoundError(f"reference CIF missing: {gt}")
        tasks.append((mid, gt, sorted(by_mid.get(mid, []))))
    print(f"{len(tasks)} targets, {sum(len(t[2]) for t in tasks)} saved candidates", flush=True)

    with Pool(args.nproc) as pool:
        res = pool.map(score_material, tasks, chunksize=1)
    gt_valid = {mid: v for mid, v, _ in res}
    df = pd.DataFrame([r for _, _, rr in res for r in rr],
                      columns=["material_id", "sample_idx", "comp_valid", "struct_valid", "fp_ok", "rms", "gt_valid"])
    os.makedirs(args.output_dir, exist_ok=True)
    df.to_csv(os.path.join(args.output_dir, "per_sample.csv"), index=False)

    n = len(mids)
    df["match_unf"] = df.struct_valid & df.rms.notna()
    df["match_filt"] = df.comp_valid & df.struct_valid & df.fp_ok & df.rms.notna()
    s0 = df[df.sample_idx == 0]
    out = dict(n_targets=n, k=args.k, n_targets_reference_invalid=int(sum(not v for v in gt_valid.values())))
    for tag, col, need_gt in (("filtered", "match_filt", True), ("unfiltered", "match_unf", False)):
        one = s0[s0[col] & (s0.gt_valid if need_gt else True)]
        many = df[df[col]].groupby("material_id").rms.min()
        out[f"{tag}_k1_match_rate"] = round(100 * len(one) / n, 2)
        out[f"{tag}_k1_rmse"] = round(float(one.rms.mean()), 4) if len(one) else None
        out[f"{tag}_k{args.k}_match_rate"] = round(100 * len(many) / n, 2)
        out[f"{tag}_k{args.k}_rmse"] = round(float(many.mean()), 4) if len(many) else None
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
