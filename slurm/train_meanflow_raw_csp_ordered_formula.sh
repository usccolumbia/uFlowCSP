#!/bin/sh
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -p gpu-H200
## #SBATCH --account <your-slurm-account>   # uncomment and set for your site
#SBATCH --signal=B:TERM@180

# Usage: sbatch train_meanflow_raw_csp_ordered_formula.sh [flow_ratio] [cfg_scale] [split_type] \
#                [sym_lattice] [sym_rotavg] [sym_anchor] [sym_loss_weight] \
#                [atom_ordering] [perm_augment] [modulo_translation] [use_formula_embedding]
#
#   flow_ratio          (optional) fraction of batch where r=t, e.g. 0.7   (default: 0.25)
#   cfg_scale            (optional) CFG distillation scale, e.g. 2.0        (default: 1.5)
#   split_type           (optional) data split type, 'difCSP' or 'random'   (default: difCSP)
#   sym_lattice          (optional) project lattice onto SG subspace        (default: false)
#   sym_rotavg           (optional) rotational averaging of coord channels  (default: false)
#   sym_anchor           (optional) Wyckoff anchor expansion (NotImpl)      (default: false)
#   sym_loss_weight      (optional) auxiliary symmetry MSE loss weight      (default: 0.0)
#   atom_ordering        (optional) 'none' | 'simple' | 'symmetry'         (default: symmetry)
#   perm_augment         (optional) hierarchical permutation augmentation   (default: false)
#   modulo_translation   (optional) global random modulo translation aug    (default: false)
#   use_formula_embedding (optional) add e_formula composition embedding    (default: true)
#
# Example (formula embedding on, ordering-only A/B vs baseline):
#   sbatch train_meanflow_raw_csp_ordered_formula.sh 0.25 1.5 difCSP false false false 0.0 symmetry false false true
# Example (formula embedding off, for direct A/B against train_meanflow_raw_csp_ordered.sh):
#   sbatch train_meanflow_raw_csp_ordered_formula.sh 0.25 1.5 difCSP false false false 0.0 symmetry false false false
#
# ============================================================================
#  This is a NEW, ADDITIVE variant of train_meanflow_raw_csp_ordered.sh, which
#  itself is unchanged by this file. The only difference here is the denoiser:
#  MeanFlowSiTRawFormula adds an explicit formula-global composition embedding
#  e_formula (derived from normalized element counts) to the conditioning
#  vector c = e_t + e_r + e_formula, so the model has direct access to the
#  whole composition instead of inferring it via attention over repeated
#  atom-type tokens. This does not require or reintroduce spacegroup (SG) as
#  a required inference input -- it stays purely formula-derived.
#
#  With use_formula_embedding=false this reproduces train_meanflow_raw_csp_ordered.sh's
#  conditioning exactly (still routed through the new denoiser class/config,
#  but with e_formula disabled).
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

application="$PYTHON $REPO_ROOT/src/train_diffusion.py --config-path=$REPO_ROOT/configs --config-name=train_meanflow_raw_formula_cfg_supervised"

#! Model hyperparameters
atom_type_vocab_size=120  # Z up to 118 + padding
atom_type_embed_dim=32
coord_dim=3
lattice_dim=9
depth=12         # Transformer depth
hidden_size=768  # Hidden size
num_heads=12     # Number of attention heads

#! MeanFlow-specific hyperparameters
cfg_scale=${2:-1.5}    # Default CFG distillation scale (can be overridden by CFG_SCALE_OVERRIDE)
cfg_dropout=${CFG_DROPOUT_OVERRIDE:-0.8}  # Fraction of batch assigned null SG; drives BOTH cfg_ratio and class_dropout_prob
cfg_scale=${CFG_SCALE_OVERRIDE:-$cfg_scale}
split_type=${3:-random}  # Data split type: 'difCSP' (fixed) or 'random'
flow_ratio=${1:-0.25}  # Fraction of batch where r=t (pure velocity supervision); paper optimal=0.25
jvp_api=funtorch # 'autograd' or 'funtorch' -- funtorch uses true forward-mode AD (more stable)
lr=0.0001        # Paper LR (constant schedule, matches paper Table 4)
num_steps=1      # Sampling steps at eval (1 = true one-step generation)
use_spacegroup=${USE_SPACEGROUP:-false}  # true: SG-conditioned training; false: force null SG everywhere

#! Spacegroup symmetry projection flags (CrystalFlow-style)
#  All default to false / 0.0 so existing runs are unchanged.
sym_lattice=${4:-false}      # stage 1: lattice metric-tensor projection
sym_rotavg=${5:-false}       # stage 2: coord rotational averaging
sym_anchor=${6:-false}       # stage 3: Wyckoff anchor (NotImplemented if true)
sym_loss_weight=${7:-0.0}    # aux ||u - sym(u)||^2 weight

#! MCFlow-style atom ordering flags (same as train_meanflow_raw_csp_ordered.sh)
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

#! NEW: formula-global composition embedding flag (this variant's whole point).
#  Default true; set to false (11th positional arg or FORMULA_EMBED_OVERRIDE)
#  to A/B against the baseline conditioning while still using this script.
use_formula_embedding=${11:-true}
use_formula_embedding=${FORMULA_EMBED_OVERRIDE:-$use_formula_embedding}

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
name="${run_name_prefix}train_meanflow_raw_${RUN_DATE}_csp_cfgscale${cfg_scale}_fr${flow_ratio}_${jvp_api}_split${split_type}_CFG${cfg_dropout}${sg_suffix}${alex_suffix}${sym_suffix}${ord_suffix}${formula_suffix}"

#! Checkpoint path (null = fresh training run)
ckpt_path=null

#! Run configuration
options="
data.datamodule=null
+data.datamodule._target_=src.data.mp20_raw_datamodule.MP20RawDataModule
+data.datamodule.mp20_root=$REPO_ROOT/data/mp_20
+data.datamodule.force_reload=False
+data.datamodule.exclude_ids_csv=${REPO_ROOT}/180_primitive.csv
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
trainer.check_val_every_n_epoch=100
+data.datamodule.batch_size=256
+trainer.accumulate_grad_batches=1
trainer.log_every_n_steps=100
trainer.max_epochs=${MAX_EPOCHS:-700}
+trainer.precision=bf16-mixed
logger=wandb
logger.wandb.name=${name}
logger.wandb.project=${WANDB_PROJECT}
task_name=train_meanflow_raw_csp_ordered_formula
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
diffusion_module.conditioning.spacegroup=${use_spacegroup}
ckpt_path=${ckpt_path}
"

workdir="$REPO_ROOT"
export OMP_NUM_THREADS=1
np=$(( ${SLURM_NNODES:-1} * ${SLURM_NTASKS_PER_NODE:-1} ))

# ── Campaign integration ─────────────────────────────────────────────────────
CAMPAIGN_RUN_DIR=""
if [ -n "${CAMPAIGN_LABEL:-}" ]; then
    CAMPAIGN_RUN_DIR="${workdir}/logs/train_meanflow_raw_csp_ordered_formula/runs/campaign_${CAMPAIGN_LABEL}"
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
