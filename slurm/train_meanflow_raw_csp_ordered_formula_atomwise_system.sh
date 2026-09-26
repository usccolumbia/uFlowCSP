#!/bin/sh
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -p gpu-A100
## #SBATCH --account <your-slurm-account>   # uncomment and set for your site
#SBATCH --signal=B:TERM@180

# Usage: sbatch train_meanflow_raw_csp_ordered_formula_atomwise_system.sh [flow_ratio] [cfg_scale] [split_type] \
#                [sym_lattice] [sym_rotavg] [sym_anchor] [sym_loss_weight] \
#                [atom_ordering] [perm_augment] [modulo_translation] [use_formula_embedding] [use_atomwise_features] [use_crystal_system]
#
#   flow_ratio          (optional) fraction of batch where r=t, e.g. 0.7   (default: 0.25)
#   cfg_scale            (optional) CFG distillation scale, e.g. 2.0        (default: 1.5)
#   split_type           (optional) data split type, 'difCSP' or 'random'   (default: random)
#   sym_lattice          (optional) project lattice onto SG subspace        (default: false)
#   sym_rotavg           (optional) rotational averaging of coord channels  (default: false)
#   sym_anchor           (optional) Wyckoff anchor expansion (NotImpl)      (default: false)
#   sym_loss_weight      (optional) auxiliary symmetry MSE loss weight      (default: 0.0)
#   atom_ordering        (optional) 'none' | 'simple' | 'symmetry'         (default: symmetry)
#   perm_augment         (optional) hierarchical permutation augmentation   (default: false)
#   modulo_translation   (optional) global random modulo translation aug    (default: false)
#   use_formula_embedding (optional) add e_formula composition embedding    (default: true)
#   use_atomwise_features (optional) add per-token chemistry features       (default: true)
#   use_crystal_system    (optional) add coarse crystal-system token        (default: true)
#
# Example (all three signals on -- current best + crystal-system token):
#   sbatch train_meanflow_raw_csp_ordered_formula_atomwise_system.sh 0.25 1.5 difCSP false false false 0.0 symmetry false false true true true
# Example (crystal-system off, for direct A/B against train_meanflow_raw_csp_ordered_formula_atomwise.sh):
#   sbatch train_meanflow_raw_csp_ordered_formula_atomwise_system.sh 0.25 1.5 difCSP false false false 0.0 symmetry false false true true false
#
# ============================================================================
#  This is a NEW, ADDITIVE variant of train_meanflow_raw_csp_ordered_formula_atomwise.sh,
#  which itself is unchanged by this file. The only difference here is the
#  denoiser: MeanFlowSiTRawFormulaAtomwiseSystem additionally adds a COARSE
#  (7-class) crystal-system conditioning token to c:
#
#    c = e_t + e_r + e_formula + e_crystal_system     (fine 230-way SG NOT used)
#
#  The crystal system is DERIVED on the fly from the space-group number inside
#  the denoiser (SG number -> one of 7 systems), then the fine space group is
#  discarded. Crystal system is NOT derivable from formula alone, so the GT SG
#  label must reach the denoiser at TRAINING time -- hence this script sets
#  diffusion_module.conditioning.spacegroup=true (below). The fine 230-way SG
#  embedding is disabled (denoiser.use_spacegroup_conditioning=false), so the
#  model only ever sees the coarse crystal system, never the full space group.
#
#  At inference the space-group input is either null (unconditional) or an
#  enumeration over the 7 crystal systems across the per-formula samples; a
#  matched enumerate-sampling script is not included here.
#
#  With use_crystal_system=false this reproduces
#  train_meanflow_raw_csp_ordered_formula_atomwise.sh's conditioning behaviour
#  (crystal-system token disabled); note that conditioning.spacegroup is still
#  forced true here so the class/config path is consistent, but with fine-SG
#  embedding off and the crystal-system token off, no symmetry signal is used.
#
#  All other configs/splits/defaults are identical to
#  train_meanflow_raw_csp_ordered_formula_atomwise.sh -- unchanged.
# ============================================================================

RUN_DATE=$(date +"%d%b")

############################################################
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

############################################################

application="$PYTHON $REPO_ROOT/src/train_diffusion.py --config-path=$REPO_ROOT/configs --config-name=train_meanflow_raw_formula_atomwise_system_cfg_supervised"

#! Model hyperparameters
atom_type_vocab_size=120  # Z up to 118 + padding
atom_type_embed_dim=32
coord_dim=3
lattice_dim=9
depth=${DEPTH:-12}               # Transformer depth (env DEPTH; DiT-L=24)
hidden_size=${HIDDEN_SIZE:-768}  # Hidden size   (env HIDDEN_SIZE; DiT-L=1024)
num_heads=${NUM_HEADS:-12}       # Attention heads (env NUM_HEADS; DiT-L=16)

#! MeanFlow-specific hyperparameters
cfg_scale=${2:-1.5}    # Default CFG distillation scale (can be overridden by CFG_SCALE_OVERRIDE)
cfg_dropout=${CFG_DROPOUT_OVERRIDE:-0.8}  # Fraction of batch assigned null SG; drives BOTH cfg_ratio and class_dropout_prob
cfg_scale=${CFG_SCALE_OVERRIDE:-$cfg_scale}
split_type=${3:-difCSP}  # Data split type: 'difCSP' (fixed) or 'random'
# --- Multi-benchmark dataset selection (default mp20 => unchanged) -----------
# DATASET=perov|mpts trains on that benchmark: points data_root at data/<ds>,
# forces split_type=<ds> (loads data/splits/<ds>_*_ids.json), and drops the
# mp20-only 180-formula exclusion. Run slurm/prepare_benchmark.sh <ds> first.
dataset=${DATASET:-mp20}
if [ "$dataset" != "mp20" ]; then
    split_type="$dataset"
fi
flow_ratio=${1:-0.25}  # Fraction of batch where r=t (pure velocity supervision); paper optimal=0.25
jvp_api=funtorch # 'autograd' or 'funtorch' -- funtorch uses true forward-mode AD (more stable)
lr=0.0001        # Paper LR (constant schedule, matches paper Table 4)
num_steps=1      # Sampling steps at eval (1 = true one-step generation)
# NOTE: 'use_spacegroup' here only tags the run name (fine SG is always OFF in
# this variant). conditioning.spacegroup is forced true below so the GT SG label
# can reach the denoiser to DERIVE the coarse crystal system.
use_spacegroup=${USE_SPACEGROUP:-false}
# Global seed. Default 9 = the config's value, so unset SEED reproduces every
# existing run exactly. Override for replicate runs: --export=ALL,SEED=10
seed=${SEED:-9}

#! Spacegroup symmetry projection flags (CrystalFlow-style)
#  All default to false / 0.0 so existing runs are unchanged.
sym_lattice=${4:-false}      # stage 1: lattice metric-tensor projection
sym_rotavg=${5:-false}       # stage 2: coord rotational averaging
sym_anchor=${6:-false}       # stage 3: Wyckoff anchor (NotImplemented if true)
sym_loss_weight=${SYM_LOSS_WEIGHT:-${7:-0.0}}    # aux ||u - sym(u)||^2 weight (env SYM_LOSS_WEIGHT; train-time only, formula-only-safe)

#! MCFlow-style atom ordering flags (same as train_meanflow_raw_csp_ordered_formula_atomwise.sh)
atom_ordering=${8:-symmetry}          # 'none' | 'simple' | 'symmetry'
perm_augment=${9:-false}              # hierarchical inter-/intra-orbit permutation aug
modulo_translation=${10:-false}       # global random modulo translation aug
# Order the CSP formula atoms by electronegativity at inference to match the
# canonical training order. Defaults to true here (consistent with ordered
# training); override with FORMULA_ORDER_BY_EN=false.
formula_order_by_en=${FORMULA_ORDER_BY_EN:-true}
# With atom_ordering=none, force inference ordering off so the legacy pipeline
# is reproduced exactly.
if [ "$atom_ordering" = "none" ]; then
    formula_order_by_en=false
fi

#! Formula-global composition embedding flag (same as train_meanflow_raw_csp_ordered_formula_atomwise.sh).
#  Default true; set to false (11th positional arg or FORMULA_EMBED_OVERRIDE).
use_formula_embedding=${11:-true}
use_formula_embedding=${FORMULA_EMBED_OVERRIDE:-$use_formula_embedding}

#! Per-token atomic chemistry feature flag (same as train_meanflow_raw_csp_ordered_formula_atomwise.sh).
#  Default true; set to false (12th positional arg or ATOMWISE_EMBED_OVERRIDE).
use_atomwise_features=${12:-true}
use_atomwise_features=${ATOMWISE_EMBED_OVERRIDE:-$use_atomwise_features}

#! NEW: coarse crystal-system conditioning-token flag (this variant's whole point).
#  Default true; set to false (13th positional arg or CRYSTAL_SYSTEM_OVERRIDE)
#  to A/B against MeanFlowSiTRawFormulaAtomwise while still using this script.
use_crystal_system=${13:-true}
use_crystal_system=${CRYSTAL_SYSTEM_OVERRIDE:-$use_crystal_system}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True    # reduce fragmentation (OOM fix)
export WANDB_MODE=${WANDB_MODE:-online}
# Weights & Biases credentials are read from the ENVIRONMENT -- nothing is
# hardcoded here. Either run `wandb login` once on the cluster, or export
# WANDB_API_KEY (and optionally WANDB_ENTITY) before submitting. Set
# WANDB_MODE=offline, or logger=null on the command line, to train without W&B.
export WANDB_PROJECT=${WANDB_PROJECT:-meanflow-csp}

[ -n "$WANDB_API_KEY" ] && wandb login "$WANDB_API_KEY"
#! Alexandria dataset — override with: --export=ALL,USE_ALEXANDRIA=false (default: true)
use_alexandria=${USE_ALEXANDRIA:-false}
alex_csv="${ALEX_CSV:-$REPO_ROOT/alex_mp20_train.csv}"
run_name_prefix=${RUN_NAME_PREFIX:-}

#! Logging name
alex_suffix=""
if [ "$use_alexandria" = "true" ]; then
    alex_suffix="_alex"
fi
sym_suffix=""
[ "$sym_lattice"     = "true" ] && sym_suffix="${sym_suffix}_LP"
[ "$sym_rotavg"      = "true" ] && sym_suffix="${sym_suffix}_RA"
[ "$sym_anchor"      = "true" ] && sym_suffix="${sym_suffix}_AN"
# Tag the symmetry loss weight only when > 0 (avoid noise for default 0.0)
if [ "$(echo "$sym_loss_weight > 0" | bc -l 2>/dev/null)" = "1" ]; then
    sym_suffix="${sym_suffix}_sl${sym_loss_weight}"
fi
# Fine 230-way SG is always OFF in this variant; tag _noSG for consistency with siblings.
sg_suffix=""
if [ "$use_spacegroup" != "true" ]; then
    sg_suffix="_noSG"
fi
# Atom-ordering suffix for run name
ord_suffix="_ord-${atom_ordering}"
[ "$perm_augment"       = "true" ] && ord_suffix="${ord_suffix}_pa"
[ "$modulo_translation" = "true" ] && ord_suffix="${ord_suffix}_mt"
# Formula-embedding suffix for run name
formula_suffix="_fe"
[ "$use_formula_embedding" = "false" ] && formula_suffix="_fe-off"
# Atomwise-feature suffix for run name
atomwise_suffix="_aw"
[ "$use_atomwise_features" = "false" ] && atomwise_suffix="_aw-off"
# Crystal-system-token suffix for run name
cs_suffix="_cs"
[ "$use_crystal_system" = "false" ] && cs_suffix="_cs-off"
name="${run_name_prefix}train_meanflow_raw_${RUN_DATE}_csp_cfgscale${cfg_scale}_fr${flow_ratio}_${jvp_api}_split${split_type}_CFG${cfg_dropout}${sg_suffix}${alex_suffix}${sym_suffix}${ord_suffix}${formula_suffix}${atomwise_suffix}${cs_suffix}"

#! Checkpoint path (null = fresh training run). CKPT_PATH_OVERRIDE lets a campaign
#  force a fresh run (CKPT_PATH_OVERRIDE=null) -- required for a clean hyperparameter
#  sweep -- or point elsewhere, without editing this line. Unset => original resume.
ckpt_path=${CKPT_PATH_OVERRIDE:-"logs/train_meanflow_raw_csp_ordered_formula_atomwise_system/runs/_2026-07-10_03-35-27/checkpoints/last.ckpt"}

#! Per-dataset root + exclusion. mp20 keeps its exact absolute root and the
#  180-formula exclusion; benchmarks use data/<ds> (relative to the submit dir)
#  and no exclusion.
if [ "$dataset" = "mp20" ]; then
    data_root="$REPO_ROOT/data/mp_20"
    exclude_line="+data.datamodule.exclude_ids_csv=${REPO_ROOT}/180_primitive.csv"
else
    data_root="${REPO_ROOT}/data/${dataset}"
    exclude_line="+data.datamodule.exclude_ids_csv=null"
fi

#! Run configuration
options="
data.datamodule=null
+data.datamodule._target_=src.data.mp20_raw_datamodule.MP20RawDataModule
+data.datamodule.mp20_root=${data_root}
+data.datamodule.dataset=${dataset}
+data.datamodule.data_root=${data_root}
+data.datamodule.force_reload=False
${exclude_line}
+data.datamodule.split_type=${split_type}
+data.datamodule.use_alexandria=${use_alexandria}
+data.datamodule.alex_csv=${alex_csv}
+data.datamodule.alex_max_samples=40000
+data.datamodule.val_size=1200
+data.datamodule.test_size=1200
+data.datamodule.atom_ordering=${atom_ordering}
+data.datamodule.perm_augment=${perm_augment}
+data.datamodule.modulo_translation=${modulo_translation}
data=mp20_raw
callbacks=diffusion_mp20_only
trainer=default
trainer.check_val_every_n_epoch=${CHECK_VAL_EVERY:-100}
+data.datamodule.batch_size=${BATCH_SIZE:-256}
+trainer.accumulate_grad_batches=${GRAD_ACCUM:-1}
trainer.log_every_n_steps=100
trainer.max_epochs=${MAX_EPOCHS:-700}
+trainer.precision=bf16-mixed
logger=wandb
logger.wandb.name=${name}
logger.wandb.project=${WANDB_PROJECT}
task_name=train_meanflow_raw_csp_ordered_formula_atomwise_system
diffusion_module.denoiser.atom_type_vocab_size=${atom_type_vocab_size}
diffusion_module.denoiser.atom_type_embed_dim=${atom_type_embed_dim}
diffusion_module.denoiser.coord_dim=${coord_dim}
diffusion_module.denoiser.lattice_dim=${lattice_dim}
diffusion_module.denoiser.depth=${depth}
diffusion_module.denoiser.hidden_size=${hidden_size}
diffusion_module.denoiser.num_heads=${num_heads}
diffusion_module.denoiser.predict_atom_types=false
diffusion_module.denoiser.atom_type_condition_dropout=0.0
diffusion_module.denoiser.class_dropout_prob=${cfg_dropout}
diffusion_module.denoiser.use_formula_embedding=${use_formula_embedding}
diffusion_module.denoiser.use_atomwise_features=${use_atomwise_features}
diffusion_module.denoiser.use_crystal_system_embedding=${use_crystal_system}
diffusion_module.denoiser.use_spacegroup_conditioning=false
diffusion_module.meanflow.cfg_scale=${cfg_scale}
diffusion_module.meanflow.cfg_ratio=${cfg_dropout}
diffusion_module.meanflow.flow_ratio=${flow_ratio}
diffusion_module.meanflow.jvp_api=${jvp_api}
diffusion_module.meanflow.atom_loss_weight=0.0
diffusion_module.meanflow.proj_lattice=${sym_lattice}
diffusion_module.meanflow.proj_coords_rotavg=${sym_rotavg}
diffusion_module.meanflow.proj_coords_anchor=${sym_anchor}
diffusion_module.meanflow.sym_loss_weight=${sym_loss_weight}
diffusion_module.conditioning.formula_order_by_en=${formula_order_by_en}
diffusion_module.optimizer.lr=${lr}
+diffusion_module.optimizer.betas=[0.9,0.95]
diffusion_module.sampling.num_sampling_steps=${num_steps}
diffusion_module.sampling.generate_atom_types=false
diffusion_module.conditioning.spacegroup=true
seed=${seed}
ckpt_path=${ckpt_path}
"

workdir="$REPO_ROOT"
export OMP_NUM_THREADS=1
np=$(( ${SLURM_NNODES:-1} * ${SLURM_NTASKS_PER_NODE:-1} ))

# ── Campaign integration ─────────────────────────────────────────────────────
CAMPAIGN_RUN_DIR=""
if [ -n "${CAMPAIGN_LABEL:-}" ]; then
    CAMPAIGN_RUN_DIR="${workdir}/logs/train_meanflow_raw_csp_ordered_formula_atomwise_system/runs/campaign_${CAMPAIGN_LABEL}"
    mkdir -p "${CAMPAIGN_RUN_DIR}/checkpoints"
    options="${options}
hydra.run.dir=${CAMPAIGN_RUN_DIR}
"
fi

CMD="$application $options"

###############################################################
### You should not have to change anything below this line ####
###############################################################

cd $workdir
echo -e "Changed directory to `pwd`.\n"

JOBID=$SLURM_JOB_ID

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"

if [ "$SLURM_JOB_NODELIST" ]; then
        export NODEFILE=`generate_pbs_nodefile`
        cat $NODEFILE | uniq > machine.file.$JOBID
        echo -e "\nNodes allocated:\n================"
        echo `cat machine.file.$JOBID | sed -e 's/\..*$//g'`
fi

echo -e "\nnumtasks=$np, numnodes=$SLURM_NNODES, mpi_tasks_per_node=$SLURM_NTASKS_PER_NODE (OMP_NUM_THREADS=$OMP_NUM_THREADS)"

echo -e "\nExecuting command:\n==================\n$CMD\n"

# ── Crash-safe registry: append the final best-valid ckpt on ANY exit ────────
_register_best_ckpt() {
    local rc=$?
    if [ -n "${CAMPAIGN_LABEL:-}" ] && [ -n "${CAMPAIGN_CKPT_REGISTRY:-}" ] \
       && [ -n "${CAMPAIGN_RUN_DIR}" ]; then
        local ckdir="${CAMPAIGN_RUN_DIR}/checkpoints"
        local best
        best=$(ls -1 "${ckdir}"/uflow-best_valid-*.ckpt 2>/dev/null \
               | awk -F 'valid_rate@' '{print $2"\t"$0}' \
               | sort -k1,1 -gr | head -1 | cut -f2-)
        if [ -z "$best" ] && [ -f "${ckdir}/last.ckpt" ]; then
            best="${ckdir}/last.ckpt"
        fi
        if [ -n "$best" ]; then
            echo "CKPT_${CAMPAIGN_LABEL}=\"${best}\"  # exit_rc=${rc}" >> "${CAMPAIGN_CKPT_REGISTRY}"
            echo "Registry updated (exit rc=${rc}): CKPT_${CAMPAIGN_LABEL} = ${best}"
        else
            echo "WARNING: no checkpoint found in ${ckdir} for ${CAMPAIGN_LABEL} (exit rc=${rc})" >&2
        fi
    fi
}
trap _register_best_ckpt EXIT

eval $CMD
