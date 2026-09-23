#!/bin/sh
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH -p gpu-A100
#SBATCH --job-name=mf_sample_test
#SBATCH --output=sample_test_%j.out
#SBATCH --error=sample_test_%j.err
## #SBATCH --account <your-slurm-account>   # uncomment and set for your site
# ============================================================================
#  Single-formula sampling test. Uses ./model.ckpt unless CKPT=<path> is set.
#
#      sbatch slurm/sample_test.sh
#
#  Overrides (all optional):
#      CKPT=<path>    CSV=<path>    OUT=<dir>
#      N=<samples per formula>      STEPS=<integration steps>
#      RELAX=0        skip relaxation (default 1 = relax with ORB v3)
# ============================================================================

module load python3/anaconda/3.12 2>/dev/null || true
module load cuda/12.1 2>/dev/null || true
[ -f ~/.bashrc ] && . ~/.bashrc
conda activate "${CONDA_ENV:-meanflow}" 2>/dev/null || true

# Repo root: under SLURM the dir you ran `sbatch` from, else the parent of slurm/
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
fi
cd "$REPO_ROOT" || exit 1
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
PYTHON=${PYTHON:-python}

# Repo-relative by default (see README, "Checkpoint"). Override with CKPT=<path>.
CKPT=${CKPT:-model.ckpt}
CSV=${CSV:-test.csv}
OUT=${OUT:-out2/}
N=${N:-5}
STEPS=${STEPS:-1}
RELAX=${RELAX:-1}

# Default one-formula CSV, created only if it does not already exist.
if [ ! -f "$CSV" ]; then
    printf 'primitive_formula\nSiO2\n' > "$CSV"
    echo "wrote $CSV"
fi

[ -f "$CKPT" ] || {
    echo "ABORT: checkpoint not found: $CKPT"
    echo "       Put one at ${REPO_ROOT}/model.ckpt, or pass CKPT=<path>."
    echo "       Trained checkpoints land in logs/<task>/runs/<run>/checkpoints/."
    exit 1
}

echo "Time:       $(date)"
echo "Repo root:  $REPO_ROOT"
echo "Checkpoint: $CKPT"
echo "CSV:        $CSV  ($(( $(grep -c '' "$CSV") - 1 )) formulas)"
echo "Samples:    $N per formula at $STEPS step(s)"
echo "Output:     $OUT"
echo "Relax:      $RELAX"
echo ""

set -- --ckpt_path "$CKPT" \
       --csv_path "$CSV" \
       --num_samples_per_formula "$N" \
       --num_sampling_steps "$STEPS" \
       --save_generated_dir "$OUT"

[ "$RELAX" = "1" ] && set -- "$@" --relax

exec "$PYTHON" src/match_meanflow_raw_all_formulas.py "$@"
