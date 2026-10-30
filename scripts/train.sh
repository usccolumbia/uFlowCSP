#!/usr/bin/env bash
# Train the released uFlowCSP model on MP-20 with one GPU.
#
#   bash scripts/train.sh                  # the released configuration
#   bash scripts/train.sh logger=null      # extra Hydra overrides go at the end
#
# Environment variables:
#   SEED (9), MAX_EPOCHS (700), BATCH_SIZE (256)
#   RESUME=<path/to/last.ckpt>             continue a run
#   FORMULA_EMBEDDING, ATOMWISE_FEATURES, CRYSTAL_SYSTEM (all true): set one to
#   false to train without that conditioning signal (ablation)
#
# Checkpoints are written to logs/train_meanflow_raw_csp_ordered_formula_atomwise_system/runs/<run>/checkpoints/.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export WANDB_PROJECT=${WANDB_PROJECT:-meanflow-csp}

SEED=${SEED:-9}
FORMULA_EMBEDDING=${FORMULA_EMBEDDING:-true}
ATOMWISE_FEATURES=${ATOMWISE_FEATURES:-true}
CRYSTAL_SYSTEM=${CRYSTAL_SYSTEM:-true}

python src/train_diffusion.py \
    --config-path="$(pwd)/configs" \
    --config-name=train_meanflow_raw_formula_atomwise_system_cfg_supervised \
    data.datamodule=null \
    +data.datamodule._target_=src.data.mp20_raw_datamodule.MP20RawDataModule \
    +data.datamodule.mp20_root="$(pwd)/data/mp_20" \
    +data.datamodule.dataset=mp20 \
    +data.datamodule.data_root="$(pwd)/data/mp_20" \
    +data.datamodule.force_reload=False \
    +data.datamodule.exclude_ids_csv="$(pwd)/180_primitive.csv" \
    +data.datamodule.split_type=difCSP \
    +data.datamodule.use_alexandria=false \
    +data.datamodule.alex_max_samples=40000 \
    +data.datamodule.val_size=1200 \
    +data.datamodule.test_size=1200 \
    +data.datamodule.atom_ordering=symmetry \
    +data.datamodule.perm_augment=false \
    +data.datamodule.modulo_translation=false \
    data=mp20_raw \
    callbacks=diffusion_mp20_only \
    trainer=default \
    trainer.check_val_every_n_epoch=100 \
    +data.datamodule.batch_size="${BATCH_SIZE:-256}" \
    +trainer.accumulate_grad_batches=1 \
    trainer.log_every_n_steps=100 \
    trainer.max_epochs="${MAX_EPOCHS:-700}" \
    +trainer.precision=bf16-mixed \
    logger=wandb \
    logger.wandb.name="uflowcsp_seed${SEED}" \
    logger.wandb.project="${WANDB_PROJECT}" \
    task_name=train_meanflow_raw_csp_ordered_formula_atomwise_system \
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
    diffusion_module.denoiser.use_formula_embedding="${FORMULA_EMBEDDING}" \
    diffusion_module.denoiser.use_atomwise_features="${ATOMWISE_FEATURES}" \
    diffusion_module.denoiser.use_crystal_system_embedding="${CRYSTAL_SYSTEM}" \
    diffusion_module.denoiser.use_spacegroup_conditioning=false \
    diffusion_module.meanflow.cfg_scale=1.5 \
    diffusion_module.meanflow.cfg_ratio=0.8 \
    diffusion_module.meanflow.flow_ratio=0.25 \
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
    diffusion_module.conditioning.spacegroup=true \
    seed="${SEED}" \
    ckpt_path="${RESUME:-null}" \
    "$@"
