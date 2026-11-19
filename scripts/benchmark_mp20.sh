#!/usr/bin/env bash
# Benchmark a checkpoint on the MP-20 test set under the DiffCSP / CrystalFlow protocol:
# sample K candidates for each of the 9046 test materials (known-Z formulas, no relaxation) and score k=1 and k=K
# structure matches and RMSE, filtered (published protocol, with SMACT) and unfiltered.
#
#   bash scripts/benchmark_mp20.sh model_mp20.ckpt
#   STEPS=1 bash scripts/benchmark_mp20.sh model_mp20.ckpt
#
# Environment variables: STEPS (5), K (20), BATCH (20), NPROC (CPU cores for scoring), OUT (results/mp20_S<STEPS>_k<K>)
# Result: $OUT/eval/summary.json -- filtered_k1_match_rate, filtered_k<K>_match_rate (compare with published numbers),
#         the matching RMSEs, and the unfiltered numbers.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

CKPT=${1:?usage: bash scripts/benchmark_mp20.sh <checkpoint file>}
STEPS=${STEPS:-5}; K=${K:-20}; BATCH=${BATCH:-20}; NPROC=${NPROC:-$(nproc)}
OUT=${OUT:-results/mp20_S${STEPS}_k${K}}
GT_DIR=gt_cifs_test
KNOWNZ_CSV=data/splits/difCSP_test_knownz_full.csv
RAW_CSV=data/mp_20/raw/all.csv

[ -f "$CKPT" ] || { echo "Checkpoint not found: $CKPT" >&2; exit 1; }
echo "checkpoint: $CKPT"

if [ "$(wc -c < "$RAW_CSV")" -lt 1024 ]; then
    echo "$RAW_CSV is a Git LFS pointer. Run: git lfs install && git lfs pull" >&2
    exit 1
fi
if [ -z "$(ls -A "$GT_DIR" 2>/dev/null)" ]; then
    echo "Building reference CIFs in $GT_DIR/ (a few minutes)"
    python make_gt_cifs.py --test_ids data/splits/difCSP_test_ids.json --out_dir "$GT_DIR"
fi
if [ ! -f "$KNOWNZ_CSV" ]; then
    echo "Building $KNOWNZ_CSV"
    python make_knownz_csv.py --csv data/splits/difCSP_test.csv --gt_dir "$GT_DIR" --out "$KNOWNZ_CSV"
fi

python src/uflowcsp.py --ckpt_path "$CKPT" --csv_path "$KNOWNZ_CSV" \
    --num_samples_per_formula "$K" --batch_size "$BATCH" --num_sampling_steps "$STEPS" \
    --material_prefix --save_generated_dir "$OUT/gen"

# Each material is scored against only its own K candidates; no energy ranking.
python evaluate_diffcsp.py --gen_dir "$OUT/gen" --gt_dir "$GT_DIR" --csv "$KNOWNZ_CSV" \
    --output_dir "$OUT/eval" --k "$K" --nproc "$NPROC"
echo "Results: $OUT/eval/summary.json"
