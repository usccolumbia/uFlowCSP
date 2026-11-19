#!/usr/bin/env bash
# Train uFlowCSP (periodic coordinates + (h,k,l) plane features) on MPTS-52 with one GPU.
# Prepare the data first: python scripts/prepare_mpts52.py --diffcsp_dir <DiffCSP>/data/mpts_52
#
#   bash scripts/train_mpts52.sh                  # the reported configuration (1200 epochs, seed 9)
#   WANDB_MODE=offline bash scripts/train_mpts52.sh   # log locally instead of to Weights & Biases
#   bash scripts/train_mpts52.sh <hydra overrides>    # extra Hydra overrides go at the end
# Keep a logger: with logger=null no checkpoints are written.
#
# Environment variables:
#   SEED (9), MAX_EPOCHS (1200), BATCH_SIZE (128) x ACCUM (2) = effective batch 256
#     (128 x 2 fits a 24 GB GPU; on a 40 GB+ GPU BATCH_SIZE=256 ACCUM=1 is equivalent)
#   RESUME=<path/to/last.ckpt>             continue a run
#   PLANE_FEATURES (true)                  false = ablation with the 48 per-axis Fourier features only
#
# Same model as scripts/train.sh (MP-20); two differences: the MeanFlow time pairs are drawn uniformly on [0, 1]
# (MP-20: logistic-normal), and no formulas are excluded from training.
# Model (formula embedding, per-atom chemistry features, crystal-system token with dropout 0.8, MeanFlow flow ratio
# 0.25, AdamW lr 1e-4 constant):
#   - positions are fractional coordinates on the torus [0,1)^3: uniform prior, shortest-way interpolation, wrapped
#     after every sampler step                                                 (meanflow.torus_coords)
#   - the network sees sin/cos(2 pi k f), k = 1..8, per axis (48 numbers/atom) (denoiser.coord_fourier_bands=8)
#     plus sin/cos(2 pi G.f) for the 56 mixed plane directions G=(h,k,l), |h|,|k|,|l| <= 2
#                                                                              (denoiser.coord_fourier_mode=axis+cross)
#   - one lattice (9 numbers, Niggli cell) per crystal, loss 10 x positions + 1 x lattice
#   - padding slots zeroed at the network input (identical in training and sampling)
#
# Checkpoints: logs/train_uflowcsp_mpts52/runs/<run>/checkpoints/ (every 100th epoch in periodic/). The reported model
# is periodic/uflow-periodic-epoch@1199.ckpt (training length chosen on the validation split).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export WANDB_PROJECT=${WANDB_PROJECT:-uflowcsp}

[ -f data/mpts/raw/all.csv ] || { echo "data/mpts/raw/all.csv missing: run scripts/prepare_mpts52.py first" >&2; exit 1; }
SEED=${SEED:-9}
PLANE=()
[ "${PLANE_FEATURES:-true}" = true ] && PLANE=(+diffusion_module.denoiser.coord_fourier_mode=axis+cross)

python src/train_diffusion.py \
    --config-path="$(pwd)/configs" \
    --config-name=train_meanflow_raw_formula_atomwise_system_cfg_supervised \
    data.datamodule=null \
    +data.datamodule._target_=src.data.mp20_raw_datamodule.MP20RawDataModule \
    +data.datamodule.mp20_root="$(pwd)/data/mpts" \
    +data.datamodule.dataset=mpts \
    +data.datamodule.data_root="$(pwd)/data/mpts" \
    +data.datamodule.force_reload=False \
    +data.datamodule.exclude_ids_csv=null \
    +data.datamodule.split_type=mpts \
    +data.datamodule.use_alexandria=false \
    +data.datamodule.alex_max_samples=40000 \
    +data.datamodule.val_size=1200 \
    +data.datamodule.test_size=1200 \
    +data.datamodule.atom_ordering=symmetry \
    +data.datamodule.perm_augment=false \
    +data.datamodule.modulo_translation=false \
    data=mp20_raw \
    callbacks=diffusion_periodic \
    trainer=default \
    trainer.check_val_every_n_epoch=100 \
    +data.datamodule.batch_size="${BATCH_SIZE:-128}" \
    +trainer.accumulate_grad_batches="${ACCUM:-2}" \
    trainer.log_every_n_steps=100 \
    trainer.max_epochs="${MAX_EPOCHS:-1200}" \
    +trainer.precision=bf16-mixed \
    logger=wandb \
    logger.wandb.name="uflowcsp_mpts52_seed${SEED}" \
    logger.wandb.project="${WANDB_PROJECT}" \
    task_name=train_uflowcsp_mpts52 \
    diffusion_module.denoiser.atom_type_vocab_size=120 \
    diffusion_module.denoiser.atom_type_embed_dim=32 \
    diffusion_module.denoiser.coord_dim=3 \
    diffusion_module.denoiser.lattice_dim=9 \
    diffusion_module.denoiser.depth=12 \
    diffusion_module.denoiser.hidden_size=768 \
    diffusion_module.denoiser.num_heads=12 \
    diffusion_module.denoiser.predict_atom_types=false \
    diffusion_module.denoiser.atom_type_condition_dropout=0.0 \
    diffusion_module.denoiser.class_dropout_prob=0.8 \
    diffusion_module.denoiser.use_formula_embedding=true \
    diffusion_module.denoiser.use_atomwise_features=true \
    diffusion_module.denoiser.use_crystal_system_embedding=true \
    diffusion_module.denoiser.use_spacegroup_conditioning=false \
    diffusion_module.meanflow.cfg_scale=1.5 \
    diffusion_module.meanflow.cfg_ratio=0.8 \
    diffusion_module.meanflow.flow_ratio=0.25 \
    "diffusion_module.meanflow.time_dist=[uniform,0.0,1.0]" \
    diffusion_module.meanflow.jvp_api=funtorch \
    diffusion_module.meanflow.atom_loss_weight=0.0 \
    diffusion_module.meanflow.proj_lattice=false \
    diffusion_module.meanflow.proj_coords_rotavg=false \
    diffusion_module.meanflow.proj_coords_anchor=false \
    diffusion_module.meanflow.sym_loss_weight=0.0 \
    diffusion_module.conditioning.formula_order_by_en=true \
    diffusion_module.optimizer.lr=0.0001 \
    '+diffusion_module.optimizer.betas=[0.9,0.95]' \
    diffusion_module.sampling.num_sampling_steps=1 \
    diffusion_module.sampling.generate_atom_types=false \
    diffusion_module.sampling.visualize=false \
    diffusion_module.conditioning.spacegroup=true \
    diffusion_module.meanflow.torus_coords=true \
    diffusion_module.meanflow.torus_coord_dim=3 \
    diffusion_module.denoiser.coord_fourier_bands=8 \
    diffusion_module.meanflow.coord_dim=3 \
    diffusion_module.meanflow.lattice_dim=9 \
    diffusion_module.meanflow.coord_loss_weight=10.0 \
    diffusion_module.meanflow.lattice_loss_weight=1.0 \
    +diffusion_module.meanflow.lattice_per_structure=true \
    +diffusion_module.denoiser.zero_padding_state=true \
    "${PLANE[@]}" \
    seed="${SEED}" \
    ckpt_path="${RESUME:-null}" \
    "$@"
