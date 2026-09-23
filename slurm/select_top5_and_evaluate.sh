#!/bin/sh
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH -p gpu-A100
## #SBATCH --account <your-slurm-account>   # uncomment and set for your site
# Usage:
#   sbatch slurm/select_top5_and_evaluate.sh <gen_dir> <gt_dir> <csv> [output_dir] [top_k] [ranker] [matris_model]
# Example (CHGNet, default):
#   sbatch slurm/select_top5_and_evaluate.sh \
#     results_generate/generated_struc_30Apr ground_truth_180 180_primitive.csv
# Example (MatRIS):
#   sbatch slurm/select_top5_and_evaluate.sh \
#     results_generate/generated_struc_30Apr ground_truth_180 180_primitive.csv \
#     results_generate/top5_eval_matris 5 matris matris_10m_oam
#
# If the 'matris' env is not on the auto-detected path, set MATRIS_PYTHON manually:
#   MATRIS_PYTHON=/path/to/matris/env/bin/python sbatch ...

GEN_DIR="${1:?usage: gen_dir gt_dir csv [output_dir] [top_k] [ranker] [matris_model]}"
GT_DIR="${2:?usage: gen_dir gt_dir csv [output_dir] [top_k] [ranker] [matris_model]}"
CSV="${3:?usage: gen_dir gt_dir csv [output_dir] [top_k] [ranker] [matris_model]}"
OUTPUT_DIR="${4:-results_generate/top5_eval_$(date +%d%b)}"
TOP_K="${5:-5}"
RANKER="${6:-chgnet}"
MATRIS_MODEL="${7:-matris_10m_oam}"
# Override via env var or leave empty for auto-detection:
MATRIS_PYTHON="${MATRIS_PYTHON:-auto}"
# StructureMatcher tolerance defaults to the CSPBench / TCSP2 protocol
# (ltol=0.2, stol=0.3, angle_tol=5). DiffCSP CSP-task protocol (looser
# ltol=0.3, stol=0.5, angle_tol=10; any-of-k match rate over all candidates;
# + normalized RMSE) is enabled by DIFFCSP=1 OR by passing a `--diffcsp` /
# `diffcsp` token anywhere in the args. Examples:
#   DIFFCSP=1 sbatch slurm/select_top5_and_evaluate.sh <gen> <gt> <csv>
#   sbatch    slurm/select_top5_and_evaluate.sh <gen> <gt> <csv> <out> 5 --diffcsp

# Accept `--diffcsp`/`diffcsp` and `--per-material`/`per_material`/`per-material`
# tokens anywhere in the args as shorthand for DIFFCSP=1 / PER_MATERIAL=1. Passed
# as script args they are always propagated to the job, whereas env vars only
# propagate under sbatch's default --export=ALL.
for a in "$@"; do
    case "$a" in
        --diffcsp|diffcsp) DIFFCSP=1 ;;
        --per-material|per-material|per_material) PER_MATERIAL=1 ;;
        --own-candidates-only|own-candidates-only|own_candidates_only) OWN_CANDIDATES=1 ;;
    esac
done
# Guard: only known rankers pass. A stray token ('--', 'diffcsp', a typo)
# landing in the RANKER slot falls back to CHGNet instead of being passed on.
# 'none' = pure DiffCSP protocol (no energy model, dummy ordering).
case "$RANKER" in
    chgnet|matris|none) : ;;
    *) RANKER=chgnet ;;
esac

LOG_BASE="${OUTPUT_DIR}/slurm"
mkdir -p "$OUTPUT_DIR"

module load python3/anaconda/3.12 2>/dev/null || true
module load cuda/12.1 2>/dev/null || true
[ -f ~/.bashrc ] && . ~/.bashrc
conda activate "${CONDA_ENV:-meanflow}" 2>/dev/null || true
# Resolve the repo root: under SLURM use the directory you ran `sbatch` from
# (under SLURM $0 is a copy in the non-writable spool dir); otherwise the parent
# of this script's directory, since these scripts live in slurm/.
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
fi
PYTHON=${PYTHON:-python}
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
ulimit -n 8192

cd "$REPO_ROOT" || exit 1

exec 1>>"${LOG_BASE}.out"
exec 2>>"${LOG_BASE}.err"

echo "Time: $(date)"
echo "Gen dir:     $GEN_DIR"
echo "GT dir:      $GT_DIR"
echo "CSV:         $CSV"
echo "Output dir:  $OUTPUT_DIR"
echo "Top K:       $TOP_K"
echo "Ranker:      $RANKER"
if [ "$RANKER" = "matris" ]; then
echo "MatRIS model: $MATRIS_MODEL"
echo "MatRIS python: $MATRIS_PYTHON"
fi
echo ""

EXTRA_ARGS=""
if [ "$RANKER" = "matris" ]; then
    EXTRA_ARGS="--ranker matris --matris_model $MATRIS_MODEL --matris_python $MATRIS_PYTHON"
elif [ "$RANKER" = "none" ]; then
    EXTRA_ARGS="--ranker none"
fi
if [ "${SKIP_SG:-0}" = "1" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --skip_sg"
    echo "SG labelling: SKIPPED (pure-matching protocol)"
fi
if [ "${DIFFCSP:-0}" = "1" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --diffcsp"
    echo "Protocol:    diffcsp (ltol=0.3, stol=0.5, angle_tol=10; any-of-k + RMSE)"
fi
if [ "${PER_MATERIAL:-0}" = "1" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --per-material"
    echo "Eval unit:   per material_id (DiffCSP-faithful; denominator = CSV rows)"
fi
if [ "${OWN_CANDIDATES:-0}" = "1" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --own-candidates-only"
    echo "Cand. pool:  own candidates only (each material scored at exactly k)"
else
    echo "Cand. pool:  POOLED per formula folder (polymorphs share candidates; some"
    echo "             materials are scored against MORE than k — set OWN_CANDIDATES=1)"
fi

$PYTHON select_top5_and_evaluate.py \
    --gen_dir "$GEN_DIR" \
    --gt_dir "$GT_DIR" \
    --csv "$CSV" \
    --output_dir "$OUTPUT_DIR" \
    --top_k "$TOP_K" \
    --device cuda \
    $EXTRA_ARGS
