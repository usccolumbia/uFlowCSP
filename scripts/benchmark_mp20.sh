#!/usr/bin/env bash
# Benchmark a checkpoint on the MP-20 test set under the DiffCSP protocol:
# sample K candidates for each of the 9046 test materials (known-Z formulas,
# no relaxation) and score any-of-K structure matches and RMSE.
#
#   bash scripts/benchmark_mp20.sh model.ckpt
#   bash scripts/benchmark_mp20.sh logs/<task>/runs/<run>/checkpoints   # picks the best-validation checkpoint
#   STEPS=1 bash scripts/benchmark_mp20.sh model.ckpt
#
# Environment variables: STEPS (5), K (20), BATCH (100), OUT (results/mp20_S<STEPS>_k<K>)
# Result: $OUT/eval/summary.json -- read diffcsp_match_rate and diffcsp_mean_rmse.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

CKPT=${1:?usage: bash scripts/benchmark_mp20.sh <checkpoint file or checkpoints directory>}
STEPS=${STEPS:-5}; K=${K:-20}; BATCH=${BATCH:-100}
OUT=${OUT:-results/mp20_S${STEPS}_k${K}}
GT_DIR=gt_cifs_test
KNOWNZ_CSV=data/splits/difCSP_test_knownz_full.csv
RAW_CSV=data/mp_20/raw/all.csv

if [ -d "$CKPT" ]; then
    # Highest valid_rate wins; it is the last @-field of the file name.
    best=$(ls -1 "$CKPT"/uflow-best_valid-*.ckpt 2>/dev/null \
           | awk -F 'valid_rate@' '{print $2"\t"$0}' | sort -k1,1 -gr | head -1 | cut -f2- || true)
    [ -n "$best" ] || { echo "No uflow-best_valid-*.ckpt in $CKPT" >&2; exit 1; }
    CKPT=$best
fi
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

# Each material is scored against only its own K candidates; no energy ranking,
# no space-group analysis.
python select_top5_and_evaluate.py --gen_dir "$OUT/gen" --gt_dir "$GT_DIR" --csv "$KNOWNZ_CSV" \
    --output_dir "$OUT/eval" --top_k "$K" --ranker none --skip_sg \
    --diffcsp --per-material --own-candidates-only
echo "Results: $OUT/eval/summary.json"
