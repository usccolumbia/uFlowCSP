#!/usr/bin/env bash
# Benchmark a checkpoint on the MPTS-52 test set under the DiffCSP / CrystalFlow protocol:
# sample K candidates for each of the 8096 test materials (known-Z formulas, no relaxation) and score k=1 and k=K
# structure matches and RMSE, filtered (published protocol, with SMACT) and unfiltered.
# Needs the files written by scripts/prepare_mpts52.py.
#
#   bash scripts/benchmark_mpts52.sh model_mpts52.ckpt
#
# Environment variables: STEPS (5), K (20), BATCH (20), NPROC (CPU cores for scoring), OUT (results/mpts52_S<STEPS>_k<K>)
# Result: $OUT/eval/summary.json (same keys as scripts/benchmark_mp20.sh).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

CKPT=${1:?usage: bash scripts/benchmark_mpts52.sh <checkpoint file>}
STEPS=${STEPS:-5}; K=${K:-20}; BATCH=${BATCH:-20}; NPROC=${NPROC:-$(nproc)}
OUT=${OUT:-results/mpts52_S${STEPS}_k${K}}
GT_DIR=gt_cifs_mpts52_test
KNOWNZ_CSV=data/splits/mpts52_test_knownz.csv

[ -f "$CKPT" ] || { echo "Checkpoint not found: $CKPT" >&2; exit 1; }
[ -f "$KNOWNZ_CSV" ] && [ -n "$(ls -A "$GT_DIR" 2>/dev/null)" ] || {
    echo "MPTS-52 test files missing: run python scripts/prepare_mpts52.py --diffcsp_dir <DiffCSP>/data/mpts_52" >&2; exit 1; }
echo "checkpoint: $CKPT"

python src/uflowcsp.py --ckpt_path "$CKPT" --csv_path "$KNOWNZ_CSV" \
    --num_samples_per_formula "$K" --batch_size "$BATCH" --num_sampling_steps "$STEPS" \
    --material_prefix --save_generated_dir "$OUT/gen"

python evaluate_diffcsp.py --gen_dir "$OUT/gen" --gt_dir "$GT_DIR" --csv "$KNOWNZ_CSV" \
    --output_dir "$OUT/eval" --k "$K" --nproc "$NPROC"
echo "Results: $OUT/eval/summary.json"
