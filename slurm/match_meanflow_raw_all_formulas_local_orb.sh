#!/bin/sh
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH -p gpu-A100
## #SBATCH --account <your-slurm-account>   # uncomment and set for your site
# ============================================================================
#  ORB v3 copy of match_meanflow_raw_all_formulas.sh that relaxes
#  generated structures with ORB v3 instead of CHGNet.
#
#  Differences vs the cluster script:
#    * No #SBATCH / module load / conda activate — runs in your current env.
#    * Paths resolved RELATIVE to this repo (no absolute hard-codes).
#    * Uses --relaxer orb (ORB v3 foundation potential) for relaxation.
#    * Prints to the console instead of redirecting into results_generate/*.out.
#
#  Run under Git Bash (the repo's .sh convention). From anywhere:
#     sh slurm/match_meanflow_raw_all_formulas_local_orb.sh <ckpt_path> <csv_path> \
#         [num_samples_per_formula] [batch_size] [num_sampling_steps] \
#         [save_generated_dir] [max_relax_per_formula]
#
#  Example (200 samples/formula, 1-step MeanFlow, ORB relax top 50):
#     sh slurm/match_meanflow_raw_all_formulas_local_orb.sh \
#         path/to/model.ckpt 180_primitive.csv 200 100 1 "" 50
#
#  Env overrides:
#     RELAX=0                 skip relaxation entirely
#     RELAX_DEVICE=cpu|cuda|auto   (default: cuda, matching the cluster match_*.sh;
#                                   use cpu/auto if this box has no GPU)
#     ORB_MODEL=<name>        ORB pretrained model (default below; use
#                             orb_v3_conservative_20_omat for a faster/lighter run)
#     DISABLE_SPACEGROUP=0    feed the GT spacegroup (default 1 = off; leaks
#                             symmetry, not a formula-only evaluation)
#     PYTHON=<path>           python interpreter to use (default: python)
#
#  One-time install (in your local env):  pip install orb-models
# ============================================================================

CKPT_PATH="$1"
CSV_PATH="$2"
NUM_SAMPLES_PER_FORMULA=${3:-200}
BATCH_SIZE=${4:-100}
NUM_SAMPLING_STEPS=${5:-1}
SAVE_GENERATED_DIR=${6:-""}
MAX_RELAX_PER_FORMULA=${7:-50}
RELAX=${RELAX:-1}
NOISE_SCALE=${NOISE_SCALE:-1.0}   # sampling-diversity temperature (1.0 = training prior)
ENUM_SYSTEMS=${ENUM_SYSTEMS:-0}   # 1 = enumerate the 7 crystal systems (formula-only)
MATERIAL_PREFIX=${MATERIAL_PREFIX:-0}  # 1 = <material_id>_sample_N.cif filenames (known-Z runs)
RELAX_FMAX=${RELAX_FMAX:-0.1}
RELAX_STEPS=${RELAX_STEPS:-200}
RELAX_DEVICE=${RELAX_DEVICE:-cuda}
ORB_MODEL=${ORB_MODEL:-orb_v3_conservative_inf_omat}
DISABLE_SPACEGROUP=${DISABLE_SPACEGROUP:-1}
PY=${PYTHON:-python}

if [ -z "$CKPT_PATH" ] || [ -z "$CSV_PATH" ]; then
    echo "Usage: sh slurm/match_meanflow_raw_all_formulas_local_orb.sh <ckpt_path> <csv_path> [num_samples_per_formula] [batch_size] [num_sampling_steps] [save_generated_dir] [max_relax_per_formula]"
    exit 1
fi

# Cluster environment setup (matches the other match_meanflow_*.sh scripts).
module load python3/anaconda/3.12 2>/dev/null || true
module load cuda/12.1 2>/dev/null || true
[ -f ~/.bashrc ] && . ~/.bashrc
conda activate "${CONDA_ENV:-meanflow}" 2>/dev/null || true
ulimit -n 8192

# Resolve repo root. Under SLURM, $0 is a copy in the (non-writable) spool dir,
# so use $SLURM_SUBMIT_DIR (the dir you ran `sbatch` from); locally fall back to
# the parent of this script's directory (script lives in slurm/).
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
    REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
fi
cd "$REPO_ROOT" || exit 1

export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

RUN_DATE=$(date +"%d%b")
if [ -z "$SAVE_GENERATED_DIR" ]; then
    SAVE_GENERATED_DIR="results_generate/generated_struc_orb_${RUN_DATE}"
fi
mkdir -p results_generate

# Redirect stdout/stderr into fixed match.out / match.er in the current directory.
exec 1>>"match.out"
exec 2>>"match.er"

CMD="$PY src/match_meanflow_raw_all_formulas.py \
--ckpt_path \"$CKPT_PATH\" \
--csv_path \"$CSV_PATH\" \
--num_samples_per_formula \"$NUM_SAMPLES_PER_FORMULA\" \
--batch_size \"$BATCH_SIZE\" \
--num_sampling_steps \"$NUM_SAMPLING_STEPS\" \
--noise_scale \"$NOISE_SCALE\" \
--save_generated_dir \"$SAVE_GENERATED_DIR\""

# Spacegroup conditioning is OFF by default in the Python entry point, so only
# the opt-in case needs a flag.
if [ "$DISABLE_SPACEGROUP" = "0" ]; then
    CMD="$CMD --use_spacegroup"
fi

if [ "$ENUM_SYSTEMS" = "1" ]; then
    CMD="$CMD --enumerate_crystal_systems"
fi

if [ "$MATERIAL_PREFIX" = "1" ]; then
    CMD="$CMD --material_prefix"
fi

if [ "$RELAX" = "1" ]; then
    CMD="$CMD --relax --relaxer orb --orb_model \"$ORB_MODEL\" \
--relax_device \"$RELAX_DEVICE\" --relax_fmax $RELAX_FMAX \
--relax_steps $RELAX_STEPS --max_relax_per_formula $MAX_RELAX_PER_FORMULA"
fi

echo "Time:                 $(date)"
echo "Repo root:            $REPO_ROOT"
echo "Checkpoint:           $CKPT_PATH"
echo "CSV:                  $CSV_PATH"
echo "Samples per formula:  $NUM_SAMPLES_PER_FORMULA"
echo "Batch size:           $BATCH_SIZE"
echo "Num sampling steps:   $NUM_SAMPLING_STEPS"
echo "Noise scale:          $NOISE_SCALE"
echo "Enum crystal systems: $ENUM_SYSTEMS"
echo "Save generated dir:   $SAVE_GENERATED_DIR"
echo "Disable spacegroup:   $DISABLE_SPACEGROUP"
if [ "$RELAX" = "1" ]; then
    echo "Relaxer:              ORB ($ORB_MODEL) device=$RELAX_DEVICE fmax=$RELAX_FMAX steps=$RELAX_STEPS max_per_formula=$MAX_RELAX_PER_FORMULA"
else
    echo "Relaxer:              DISABLED"
fi
echo ""
echo "Executing: $CMD"
echo ""

eval $CMD
