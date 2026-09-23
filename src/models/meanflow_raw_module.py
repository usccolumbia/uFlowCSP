"""MeanFlowRaw Lightning Module for one-step generative modelling of raw atomic data."""

import copy
import inspect
import os
import time
import re
from typing import Dict

import torch
from lightning import LightningModule
from omegaconf import DictConfig
from src.eval.crystal_generation import CrystalGenerationEvaluator
from src.models.meanflow_transport import MeanFlowTransport
from src.utils import pylogger
from torch.nn import ModuleDict
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch
from torchmetrics import MeanMetric
import pandas as pd

log = pylogger.RankedLogger(__name__)


class MeanFlowRawLitModule(LightningModule):
    """
    LightningModule for training MeanFlow on raw atomic data (no VAE).

    Training:  JVP self-consistency loss with implicit CFG distillation.
    Inference: 1-step (or configurable N-step) mean flow sampling.
    Evaluation: CrystalGenerationEvaluator metrics.
    """

    def __init__(
        self,
        denoiser: torch.nn.Module,
        meanflow: MeanFlowTransport,
        augmentations: DictConfig,
        sampling: DictConfig,
        conditioning: DictConfig,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        scheduler_frequency: str = "1",
        compile: bool = False,
        ema_decay: float = 0.9999,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        # No autoencoder for raw data
        self.autoencoder = None

        # Trainable denoiser
        self.denoiser = denoiser

        # MeanFlow transport (training loss + sampler)
        self.meanflow = meanflow

        # ── Target EMA denoiser (stabilises CFG training targets) ──────────
        self.ema_decay = ema_decay
        self.ema_denoiser = copy.deepcopy(self.denoiser)
        for param in self.ema_denoiser.parameters():
            param.requires_grad_(False)
        log.info(f"Created target EMA denoiser (decay={ema_decay})")

        # ── Metrics ─────────────────────────────────────────────────────────
        self.train_metrics = ModuleDict({
            "loss": MeanMetric(),
            "mse_val": MeanMetric(),
        })

        _per_dataset_metrics = lambda: ModuleDict({
            "loss": MeanMetric(),
            "mse_val": MeanMetric(),
            "valid_rate": MeanMetric(),
            "struct_valid_rate": MeanMetric(),
            "comp_valid_rate": MeanMetric(),
            "unique_rate": MeanMetric(),
            "novel_rate": MeanMetric(),
            "sampling_time": MeanMetric(),
        })
        self.val_metrics = ModuleDict({
            "mp20": _per_dataset_metrics(),
        })
        self.test_metrics = copy.deepcopy(self.val_metrics)

        # ── Augmentations ───────────────────────────────────────────────────
        self.augmentations = augmentations

        # ── Sampling ────────────────────────────────────────────────────────
        self.sampling = sampling
        self._sampling_generates_atom_types = bool(self.hparams.sampling.get("generate_atom_types", False))
        self._predict_atom_types = bool(getattr(self.denoiser, "predict_atom_types", False))

        # ── Conditioning ────────────────────────────────────────────────────
        self.conditioning = conditioning

        # ── Evaluators ──────────────────────────────────────────────────────
        try:
            mp20_cif_list = pd.read_csv(
                os.path.join(self.hparams.sampling.data_dir, "mp_20/raw/all.csv")
            )["cif"].tolist()
        except (FileNotFoundError, pd.errors.EmptyDataError):
            mp20_cif_list = []

        self.evaluators = {
            "mp20": CrystalGenerationEvaluator(
                dataset_cif_list=mp20_cif_list,
                compute_novelty=self.hparams.sampling.compute_novelty,
                relax_structures=self.hparams.sampling.relax_structures,
            ),
        }
        self._val_sampling_time = 0.0
        self._test_sampling_time = 0.0
        self._target_num_samples = int(self.hparams.sampling.get("num_samples", 0))
        self._val_generated_samples = 0
        self._test_generated_samples = 0
        self._target_formula = str(self.hparams.sampling.get("target_formula", "")).strip()
        self._target_spacegroup = self.hparams.sampling.get("target_spacegroup", None)
        if self._target_spacegroup is not None:
            self._target_spacegroup = int(self._target_spacegroup)
        self._formula_atom_types_cpu = None
        self._formula_num_atoms = 0
        # Optional: order the conditioning formula's atoms by Pauling
        # electronegativity so inference matches the canonical training order
        # (MCFlow-style). Defaults to False -> legacy alphabetical order.
        self._formula_order_by_en = bool(
            self.hparams.conditioning.get("formula_order_by_en", False)
        )
        # The torus transport needs a periodic coordinate embedding on the
        # denoiser side; a plain Linear on raw fractional coords is
        # discontinuous across the cell seam. Warn loudly if the two halves
        # disagree — this combination trains but cannot fit its own target.
        _fourier_bands = int(getattr(self.denoiser, "coord_fourier_bands", 0))
        if getattr(self.meanflow, "torus_coords", False) and _fourier_bands <= 0:
            log.warning(
                "torus_coords=True but denoiser.coord_fourier_bands=0: the model "
                "sees raw fractional coords through a non-periodic Linear, so it "
                "cannot represent the periodic target (discontinuity at f=0/1). "
                "Set diffusion_module.denoiser.coord_fourier_bands=8 (or similar)."
            )
        elif _fourier_bands > 0 and not getattr(self.meanflow, "torus_coords", False):
            log.warning(
                f"denoiser.coord_fourier_bands={_fourier_bands} but torus_coords=False: "
                "Fourier features assume fractional coords and are meaningless for "
                "Cartesian positions in Angstrom. Set coord_fourier_bands=0."
            )
        # The dataset carries ONE lattice per structure, broadcast onto every
        # token by encode_batch, and _generated_batch_to_array_dicts pools it
        # back with a mean over the ACTIVE tokens. A transport that treats the
        # lattice channels as N independent per-token quantities therefore (a)
        # loses the pooled lattice of its t=1 prior to the across-token mean
        # subtraction and (b) trains on a lattice the decode never reads.
        # PMFRawTransport can be told about the invariant; warn if it wasn't.
        if hasattr(self.meanflow, "lattice_per_structure") and not getattr(
            self.meanflow, "lattice_per_structure", False
        ):
            if int(getattr(self.denoiser, "lattice_dim", 0)) > 0:
                log.warning(
                    "meanflow.lattice_per_structure=False with a lattice head: the "
                    "lattice prior is drawn i.i.d. per token and then mean-centred "
                    "across tokens, so the pooled lattice the decode reads back at "
                    "t=1 is ~0 (no per-structure randomness), and the network's "
                    "per-token lattice output is never pooled during training. Set "
                    "+diffusion_module.meanflow.lattice_per_structure=true (plus "
                    "meanflow.coord_dim / meanflow.lattice_dim) for the corrected "
                    "geometry. Left False only to reproduce existing checkpoints."
                )

        if self._target_formula:
            formula_atom_types = self._parse_formula_to_atom_types(
                self._target_formula, order_by_en=self._formula_order_by_en
            )
            if formula_atom_types.numel() > int(self.denoiser.input_size):
                raise ValueError(
                    f"target_formula={self._target_formula!r} has {formula_atom_types.numel()} atoms "
                    f"which exceeds denoiser.input_size={self.denoiser.input_size}."
                )
            self._formula_atom_types_cpu = formula_atom_types
            self._formula_num_atoms = int(formula_atom_types.numel())
            log.info(
                f"Using target formula conditioning: {self._target_formula} "
                f"(num_atoms={self._formula_num_atoms})."
            )
        if self._target_spacegroup is not None:
            log.info(f"Using target spacegroup conditioning: {self._target_spacegroup}")
        if self._predict_atom_types and self._sampling_generates_atom_types:
            log.info("Joint A/X/L generation is enabled for raw MeanFlow sampling.")

    def forward(self, batch: Data) -> torch.Tensor:
        """Forward pass through denoiser."""
        return self.denoiser(
            x=batch.x,  # (B*N, coord_dim + lattice_dim)
            atom_types=batch.atom_types,  # (B*N,)
            t=batch.t,
            r=batch.r,
            spacegroup=batch.spacegroup,
            mask=batch.mask,
        )

    def training_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        """Training step with MeanFlow loss."""
        # Encode raw data (no VAE)
        x, atom_types, mask = self.encode_batch(batch)

        batch_size = x.shape[0]
        spacegroup = getattr(
            batch,
            "spacegroup",
            torch.zeros(batch_size, dtype=torch.long, device=x.device),
        )
        if not self.hparams.conditioning.spacegroup:
            spacegroup = torch.zeros(batch_size, dtype=torch.long, device=x.device)

        # Set batch attributes for forward pass
        batch.x = x
        batch.atom_types = atom_types
        batch.mask = mask
        batch.spacegroup = spacegroup

        # Prepare conditioning dict
        cond_dict = {
            "mask": mask,
            "spacegroup": spacegroup,
            "atom_types": atom_types,
            "generate_atom_types": self._sampling_generates_atom_types,
        }

        # Compute loss
        loss, mse_val = self.meanflow.loss(self.denoiser, x, cond_dict, ema_model=self.ema_denoiser)

        # Log metrics
        self.train_metrics["loss"](loss)
        self.train_metrics["mse_val"](mse_val)

        # EMA update
        self.update_ema()

        self.log_dict(
            self.train_metrics,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return loss

    def on_validation_epoch_start(self) -> None:
        """Reset validation metrics and evaluator buffers."""
        for metric in self.val_metrics["mp20"].values():
            metric.reset()
        if self.evaluators["mp20"] is not None:
            self.evaluators["mp20"].clear()
        self._val_sampling_time = 0.0
        self._val_generated_samples = 0

    def validation_step(self, batch: Data, batch_idx: int) -> None:
        """Compute validation loss and run raw sampling for generation metrics."""
        x, atom_types, mask = self.encode_batch(batch)

        batch_size = x.shape[0]
        spacegroup = getattr(
            batch,
            "spacegroup",
            torch.zeros(batch_size, dtype=torch.long, device=x.device),
        )
        if not self.hparams.conditioning.spacegroup:
            spacegroup = torch.zeros(batch_size, dtype=torch.long, device=x.device)

        cond_dict = {
            "mask": mask,
            "spacegroup": spacegroup,
            "atom_types": atom_types,
        }

        loss, mse_val = self.meanflow.loss(self.denoiser, x, cond_dict, ema_model=self.ema_denoiser)
        self.val_metrics["mp20"]["loss"](loss)
        self.val_metrics["mp20"]["mse_val"](mse_val)

        if self._target_num_samples <= 0 or self._val_generated_samples >= self._target_num_samples:
            return

        num_steps = int(self.hparams.sampling.get("num_sampling_steps", 1))
        while self._val_generated_samples < self._target_num_samples:
            remaining = self._target_num_samples - self._val_generated_samples
            cur_batch = min(batch_size, remaining)
            atom_types_cur, mask_cur, spacegroup_cur = self._build_sampling_conditioning(
                atom_types=atom_types,
                mask=mask,
                spacegroup=spacegroup,
                cur_batch=cur_batch,
            )

            cond_dict_cur = {
                "mask": mask_cur,
                "spacegroup": spacegroup_cur,
                "atom_types": atom_types_cur,
                "generate_atom_types": self._sampling_generates_atom_types,
            }

            t_start = time.time()
            z = self.meanflow.sample_prior(cur_batch, x.shape[1], x.shape[2], x.device)
            x_gen = self.meanflow.sample(self.ema_denoiser, z, cond_dict_cur, num_steps=num_steps)
            self._val_sampling_time += time.time() - t_start

            sample_atom_types, sample_mask = self._resolve_sample_atom_state(atom_types_cur, mask_cur)
            self._append_generated_batch_to_evaluator(
                x_gen=x_gen,
                atom_types=sample_atom_types,
                mask=sample_mask,
                sample_offset=self._val_generated_samples,
            )
            self._val_generated_samples += cur_batch

            # One generation pass per step unless the batch is smaller than remaining.
            if cur_batch == batch_size:
                break

    def on_validation_epoch_end(self) -> None:
        """Compute and log generation metrics (validity, uniqueness, novelty)."""
        evaluator = self.evaluators.get("mp20")
        if evaluator is None:
            return

        gen_metrics = evaluator.get_metrics(
            save=self.hparams.sampling.visualize,
            save_dir=self.hparams.sampling.save_dir + f"/mp20_val_epoch{self.current_epoch}_{self.global_rank}",
        )
        gen_metrics["sampling_time"] = self._val_sampling_time

        for key, value in gen_metrics.items():
            if key in self.val_metrics["mp20"]:
                self.val_metrics["mp20"][key](value)

        for key, metric in self.val_metrics["mp20"].items():
            self.log(
                f"val_mp20/{key}",
                metric,
                on_step=False,
                on_epoch=True,
                prog_bar=(key == "valid_rate"),
                logger=True,
                sync_dist=True,
            )

    def on_test_epoch_start(self) -> None:
        """Reset test metrics and evaluator buffers."""
        for metric in self.test_metrics["mp20"].values():
            metric.reset()
        if self.evaluators["mp20"] is not None:
            self.evaluators["mp20"].clear()
        self._test_sampling_time = 0.0
        self._test_generated_samples = 0

    def test_step(self, batch: Data, batch_idx: int) -> None:
        """Compute test loss and run raw sampling for generation metrics."""
        x, atom_types, mask = self.encode_batch(batch)

        batch_size = x.shape[0]
        spacegroup = getattr(
            batch,
            "spacegroup",
            torch.zeros(batch_size, dtype=torch.long, device=x.device),
        )
        if not self.hparams.conditioning.spacegroup:
            spacegroup = torch.zeros(batch_size, dtype=torch.long, device=x.device)

        cond_dict = {
            "mask": mask,
            "spacegroup": spacegroup,
            "atom_types": atom_types,
        }

        loss, mse_val = self.meanflow.loss(self.denoiser, x, cond_dict, ema_model=self.ema_denoiser)
        self.test_metrics["mp20"]["loss"](loss)
        self.test_metrics["mp20"]["mse_val"](mse_val)

        if self._target_num_samples <= 0 or self._test_generated_samples >= self._target_num_samples:
            return

        num_steps = int(self.hparams.sampling.get("num_sampling_steps", 1))
        while self._test_generated_samples < self._target_num_samples:
            remaining = self._target_num_samples - self._test_generated_samples
            cur_batch = min(batch_size, remaining)
            atom_types_cur, mask_cur, spacegroup_cur = self._build_sampling_conditioning(
                atom_types=atom_types,
                mask=mask,
                spacegroup=spacegroup,
                cur_batch=cur_batch,
            )

            cond_dict_cur = {
                "mask": mask_cur,
                "spacegroup": spacegroup_cur,
                "atom_types": atom_types_cur,
                "generate_atom_types": self._sampling_generates_atom_types,
            }

            t_start = time.time()
            z = self.meanflow.sample_prior(cur_batch, x.shape[1], x.shape[2], x.device)
            x_gen = self.meanflow.sample(self.ema_denoiser, z, cond_dict_cur, num_steps=num_steps)
            self._test_sampling_time += time.time() - t_start

            sample_atom_types, sample_mask = self._resolve_sample_atom_state(atom_types_cur, mask_cur)
            self._append_generated_batch_to_evaluator(
                x_gen=x_gen,
                atom_types=sample_atom_types,
                mask=sample_mask,
                sample_offset=self._test_generated_samples,
            )
            self._test_generated_samples += cur_batch

            # One generation pass per step unless the batch is smaller than remaining.
            if cur_batch == batch_size:
                break

    def on_test_epoch_end(self) -> None:
        """Compute and log test-time generation metrics."""
        evaluator = self.evaluators.get("mp20")
        if evaluator is None:
            return

        gen_metrics = evaluator.get_metrics(
            save=self.hparams.sampling.visualize,
            save_dir=self.hparams.sampling.save_dir + f"/mp20_test_epoch{self.current_epoch}_{self.global_rank}",
        )
        gen_metrics["sampling_time"] = self._test_sampling_time

        for key, value in gen_metrics.items():
            if key in self.test_metrics["mp20"]:
                self.test_metrics["mp20"][key](value)

        for key, metric in self.test_metrics["mp20"].items():
            self.log(
                f"test_mp20/{key}",
                metric,
                on_step=False,
                on_epoch=True,
                prog_bar=(key == "valid_rate"),
                logger=True,
                sync_dist=True,
            )

    @torch.no_grad()
    def generate_conditioned_samples(
        self,
        target_formula: str | None = None,
        target_spacegroup: int | None = None,
        num_samples: int = 800,
        batch_size: int | None = None,
        noise_scale: float = 1.0,
    ) -> list[dict]:
        """Generate raw crystal samples, optionally conditioned on a fixed formula and spacegroup.

        ``noise_scale`` (default 1.0 = training prior) is a sampling-diversity
        temperature forwarded to ``transport.sample_prior``; it widens only the
        Gaussian part of the prior (see the transport docstrings).
        """
        formula = (target_formula or "").strip()
        atom_types_seq = None
        num_atoms = 0
        max_nodes = int(self.denoiser.input_size)

        if formula:
            atom_types_seq = self._parse_formula_to_atom_types(
                formula, order_by_en=self._formula_order_by_en
            ).to(self.device)
            num_atoms = int(atom_types_seq.numel())
            if num_atoms > max_nodes:
                raise ValueError(
                    f"target_formula={formula!r} has {num_atoms} atoms, exceeds input_size={max_nodes}."
                )
        elif not self._sampling_generates_atom_types:
            raise ValueError(
                "target_formula must be provided unless sampling.generate_atom_types=true for joint A/X/L generation."
            )

        batch_size = int(batch_size or self.hparams.sampling.get("batch_size", 100))
        num_steps = int(self.hparams.sampling.get("num_sampling_steps", 1))

        generated = 0
        samples = []
        while generated < num_samples:
            cur_batch = min(batch_size, num_samples - generated)

            atom_types = torch.zeros(cur_batch, max_nodes, dtype=torch.long, device=self.device)
            mask = torch.ones(cur_batch, max_nodes, dtype=torch.bool, device=self.device) if not formula else torch.zeros(cur_batch, max_nodes, dtype=torch.bool, device=self.device)

            if atom_types_seq is not None:
                atom_types[:, :num_atoms] = atom_types_seq.unsqueeze(0).expand(cur_batch, -1)
                mask[:, :num_atoms] = True

            if target_spacegroup is None:
                spacegroup = torch.zeros(cur_batch, dtype=torch.long, device=self.device)
            else:
                spacegroup = torch.full(
                    (cur_batch,),
                    int(target_spacegroup),
                    dtype=torch.long,
                    device=self.device,
                )

            cond_dict = {
                "mask": mask,
                "spacegroup": spacegroup,
                "atom_types": atom_types,
                "generate_atom_types": self._sampling_generates_atom_types and not formula,
            }

            z = self.meanflow.sample_prior(
                cur_batch, max_nodes, self.denoiser.in_channels, self.device,
                noise_scale=noise_scale,
            )
            x_gen = self.meanflow.sample(self.ema_denoiser, z, cond_dict, num_steps=num_steps)
            sample_atom_types, sample_mask = self._resolve_sample_atom_state(atom_types, mask)
            samples.extend(
                self._generated_batch_to_array_dicts(
                    x_gen=x_gen,
                    atom_types=sample_atom_types,
                    mask=sample_mask,
                    sample_offset=generated,
                )
            )
            generated += cur_batch

        return samples

    @torch.no_grad()
    def generate_conditioned_trajectory(
        self,
        target_formula: str,
        target_spacegroup: int | None = None,
        num_samples: int = 32,
        num_steps: int = 5,
        noise_scale: float = 1.0,
    ) -> list[list[dict]]:
        """Sample ONE batch and decode the structures at *every* integration step.

        Additive sibling of :meth:`generate_conditioned_samples`, used only by
        the trajectory-figure tooling (``src/export_trajectory_meanflow.py``).
        Training, validation and the benchmark samplers never call it, and
        :meth:`generate_conditioned_samples` is untouched — the only shared
        state is the opt-in ``record_traj`` flag on the transport's sampler.

        Returns ``num_steps + 1`` lists of array dicts (the same dicts
        :meth:`generate_conditioned_samples` returns). Index 0 is the prior
        (pure noise, t=1) and index ``num_steps`` is the final sample (t=0);
        the inner lists are aligned by position, so ``traj[s][i]`` is sample
        ``i`` at step ``s``.

        The whole batch is drawn in one ``sample()`` call — ``num_samples`` must
        fit in memory, which is the point: every sample shares the trajectory
        so the caller can pick whichever one succeeds and export its path.
        """
        formula = (target_formula or "").strip()
        if not formula:
            raise ValueError("generate_conditioned_trajectory requires a target_formula.")

        max_nodes = int(self.denoiser.input_size)
        atom_types_seq = self._parse_formula_to_atom_types(
            formula, order_by_en=self._formula_order_by_en
        ).to(self.device)
        num_atoms = int(atom_types_seq.numel())
        if num_atoms > max_nodes:
            raise ValueError(
                f"target_formula={formula!r} has {num_atoms} atoms, exceeds input_size={max_nodes}."
            )

        if "record_traj" not in inspect.signature(self.meanflow.sample).parameters:
            raise RuntimeError(
                f"{type(self.meanflow).__name__}.sample() has no record_traj argument; "
                "trajectory export is only implemented for MeanFlowTransport and "
                "MeanFlowCSPTransport."
            )

        cur_batch = int(num_samples)
        atom_types = torch.zeros(cur_batch, max_nodes, dtype=torch.long, device=self.device)
        mask = torch.zeros(cur_batch, max_nodes, dtype=torch.bool, device=self.device)
        atom_types[:, :num_atoms] = atom_types_seq.unsqueeze(0).expand(cur_batch, -1)
        mask[:, :num_atoms] = True

        if target_spacegroup is None:
            spacegroup = torch.zeros(cur_batch, dtype=torch.long, device=self.device)
        else:
            spacegroup = torch.full(
                (cur_batch,), int(target_spacegroup), dtype=torch.long, device=self.device
            )

        cond_dict = {
            "mask": mask,
            "spacegroup": spacegroup,
            "atom_types": atom_types,
            # Formula is fixed, so atom types are never generated — same branch
            # generate_conditioned_samples takes for a non-empty formula.
            "generate_atom_types": False,
        }

        z = self.meanflow.sample_prior(
            cur_batch, max_nodes, self.denoiser.in_channels, self.device,
            noise_scale=noise_scale,
        )
        x_gen = self.meanflow.sample(
            self.ema_denoiser, z, cond_dict, num_steps=int(num_steps), record_traj=True
        )
        traj = getattr(self.meanflow, "last_sample_traj", None)
        if not traj:
            raise RuntimeError("Transport did not record a trajectory despite record_traj=True.")
        if not torch.equal(traj[-1], x_gen):
            raise RuntimeError("Recorded trajectory does not end on the returned sample.")

        sample_atom_types, sample_mask = self._resolve_sample_atom_state(atom_types, mask)
        return [
            self._generated_batch_to_array_dicts(
                x_gen=x_step,
                atom_types=sample_atom_types,
                mask=sample_mask,
                sample_offset=0,
            )
            for x_step in traj
        ]

    def _resolve_sample_atom_state(
        self,
        atom_types: torch.Tensor,
        mask: torch.Tensor,
    ):
        """Use the latest sampled atom predictions when available, otherwise fall back to conditioning."""
        sample_atom_types = getattr(self.meanflow, "last_sample_atom_types", None)
        sample_mask = getattr(self.meanflow, "last_sample_mask", None)

        if sample_atom_types is None:
            sample_atom_types = atom_types
        if sample_mask is None:
            sample_mask = mask

        return sample_atom_types, sample_mask

    def _append_generated_batch_to_evaluator(
        self,
        x_gen: torch.Tensor,
        atom_types: torch.Tensor,
        mask: torch.Tensor,
        sample_offset: int = 0,
    ) -> None:
        """Convert generated raw tensors to crystal arrays expected by evaluator."""
        evaluator = self.evaluators.get("mp20")
        if evaluator is None:
            return

        for sample in self._generated_batch_to_array_dicts(
            x_gen=x_gen,
            atom_types=atom_types,
            mask=mask,
            sample_offset=sample_offset,
        ):
            evaluator.append_pred_array(sample)

    def _generated_batch_to_array_dicts(
        self,
        x_gen: torch.Tensor,
        atom_types: torch.Tensor,
        mask: torch.Tensor,
        sample_offset: int = 0,
    ) -> list[dict]:
        """Convert a generated batch into array dictionaries compatible with Crystal evaluators."""
        samples = []

        eps = 1e-8
        batch_size = x_gen.shape[0]

        for i in range(batch_size):
            atom_types_row = atom_types[i].long()
            atom_types_row = torch.where(
                (atom_types_row > 0) & (atom_types_row <= 118),
                atom_types_row,
                torch.zeros_like(atom_types_row),
            )
            active_mask = mask[i].bool() & (atom_types_row > 0)
            num_atoms = int(active_mask.sum().item())
            if num_atoms <= 0:
                continue

            atom_types_i = atom_types_row[active_mask]
            # Slice by the denoiser's declared geometry rather than hardcoded
            # offsets.
            c_dim = int(getattr(self.denoiser, "coord_dim", 3))
            l_dim = int(getattr(self.denoiser, "lattice_dim", 9))
            coord_tokens_i = x_gen[i, active_mask, :c_dim]
            lattice_tokens_i = x_gen[i, active_mask, c_dim:c_dim + l_dim]
            lattice_flat_i = lattice_tokens_i.mean(dim=0)
            cell_i = lattice_flat_i.view(3, 3)

            if getattr(self.meanflow, "torus_coords", False):
                # Generated coordinate channel is already fractional (and
                # wrapped onto [0, 1) at every sampler step) - recover
                # Cartesian positions for the "pos" field.
                frac_coords_i = torch.remainder(coord_tokens_i, 1.0)
                pos_i = frac_coords_i @ cell_i
            else:
                pos_i = coord_tokens_i
                inv_cell_i = torch.linalg.pinv(cell_i)
                frac_coords_i = (pos_i @ inv_cell_i) % 1.0

            v1, v2, v3 = cell_i[0], cell_i[1], cell_i[2]
            l1 = torch.linalg.norm(v1).clamp(min=eps)
            l2 = torch.linalg.norm(v2).clamp(min=eps)
            l3 = torch.linalg.norm(v3).clamp(min=eps)

            alpha = torch.acos(torch.clamp(torch.dot(v2, v3) / (l2 * l3), -1.0, 1.0))
            beta = torch.acos(torch.clamp(torch.dot(v1, v3) / (l1 * l3), -1.0, 1.0))
            gamma = torch.acos(torch.clamp(torch.dot(v1, v2) / (l1 * l2), -1.0, 1.0))

            lengths_i = torch.stack([l1, l2, l3])
            angles_i = torch.rad2deg(torch.stack([alpha, beta, gamma]))

            samples.append({
                "atom_types": atom_types_i.detach().cpu().numpy(),
                "pos": pos_i.detach().float().cpu().numpy(),
                "frac_coords": frac_coords_i.detach().float().cpu().numpy(),
                "lengths": lengths_i.detach().float().cpu().numpy(),
                "angles": angles_i.detach().float().cpu().numpy(),
                "sample_idx": sample_offset + self.global_rank * batch_size + i,
            })

        return samples

    @staticmethod
    def _parse_formula_to_atom_types(formula: str, order_by_en: bool = False) -> torch.Tensor:
        """Parse chemical formula like Fe2O3 into a 1D atom-type tensor (atomic numbers).

        When ``order_by_en`` is True the atoms are ordered by Pauling
        electronegativity (ascending), matching the canonical training order;
        otherwise the legacy element-dict (alphabetical) order is preserved.
        """
        try:
            from pymatgen.core import Composition, Element
        except Exception as ex:
            raise RuntimeError(
                "pymatgen is required for target_formula conditioning. "
                "Install pymatgen or leave sampling.target_formula empty."
            ) from ex

        clean = formula.strip()
        if not clean:
            return torch.empty(0, dtype=torch.long)

        # Basic guard to fail fast on clearly invalid strings.
        if re.search(r"[^A-Za-z0-9().]", clean):
            raise ValueError(f"Invalid target_formula: {formula!r}")

        comp = Composition(clean)
        elem_blocks = []  # list of (en_sort_key, atomic_number, count)
        for elem, amount in comp.get_el_amt_dict().items():
            cnt = int(round(float(amount)))
            if cnt <= 0 or abs(float(amount) - cnt) > 1e-6:
                raise ValueError(
                    f"target_formula={formula!r} must contain integer positive counts; got {elem}:{amount}."
                )
            z = int(Element(elem).Z)
            en = Element(elem).X
            if en is None or (isinstance(en, float) and en != en):  # nan-safe
                en = 100.0 + z * 1e-3
            elem_blocks.append((float(en), z, cnt))

        if not elem_blocks:
            raise ValueError(f"Parsed empty formula from target_formula={formula!r}")

        if order_by_en:
            elem_blocks.sort(key=lambda b: (b[0], b[1]))

        atom_types = []
        for _en, z, cnt in elem_blocks:
            atom_types.extend([z] * cnt)

        return torch.tensor(atom_types, dtype=torch.long)

    def _build_sampling_conditioning(
        self,
        atom_types: torch.Tensor,
        mask: torch.Tensor,
        spacegroup: torch.Tensor,
        cur_batch: int,
    ):
        """Build per-batch sampling conditioning, optionally overriding with target formula."""
        if not self._target_formula:
            if self._sampling_generates_atom_types:
                # Generative mode: zero out atom types so the model generates them from scratch.
                # Keep the mask (num atoms per structure) from the batch but clear identities.
                return (
                    torch.zeros_like(atom_types[:cur_batch]),
                    mask[:cur_batch],
                    spacegroup[:cur_batch],
                )
            return (
                atom_types[:cur_batch],
                mask[:cur_batch],
                spacegroup[:cur_batch],
            )

        max_nodes = int(self.denoiser.input_size)
        atom_types_formula = torch.zeros(cur_batch, max_nodes, dtype=torch.long, device=atom_types.device)
        mask_formula = torch.zeros(cur_batch, max_nodes, dtype=torch.bool, device=mask.device)

        atom_types_seq = self._formula_atom_types_cpu.to(atom_types.device)
        atom_types_formula[:, :self._formula_num_atoms] = atom_types_seq.unsqueeze(0).expand(cur_batch, -1)
        mask_formula[:, :self._formula_num_atoms] = True

        if self._target_spacegroup is None:
            spacegroup_formula = spacegroup[:cur_batch]
        else:
            spacegroup_formula = torch.full(
                (cur_batch,),
                self._target_spacegroup,
                dtype=spacegroup.dtype,
                device=spacegroup.device,
            )

        return (
            atom_types_formula,
            mask_formula,
            spacegroup_formula,
        )

    def encode_batch(self, batch: Data):
        """Encode batch to raw data format.

        When ``meanflow.torus_coords`` is enabled, the coordinate channel is
        built from fractional coordinates (periodic on [0, 1)^3) instead of
        Cartesian ``pos``, so the transport's torus interpolation operates
        on the correct representation. Default behaviour (Cartesian ``pos``)
        is unchanged.
        """
        if getattr(self.meanflow, "torus_coords", False):
            coord_source = torch.remainder(batch.frac_coords, 1.0)
        else:
            coord_source = batch.pos
        coords, mask = to_dense_batch(coord_source, batch.batch, max_num_nodes=self.denoiser.input_size)
        atom_types, _ = to_dense_batch(batch.atom_types, batch.batch, max_num_nodes=self.denoiser.input_size)
        # lattice is the per-atom (N, 9) matrix in the dataset, used unchanged.
        lat_src = batch.lattice
        lattice_dense, _ = to_dense_batch(lat_src, batch.batch, max_num_nodes=self.denoiser.input_size)  # (B, max_N, 9)

        # Combine coords and lattice per token
        x = torch.cat([coords, lattice_dense], dim=-1)  # (B, max_N, coord_dim + lattice_dim)

        return x, atom_types, mask

    def update_ema(self):
        """Update EMA denoiser parameters."""
        with torch.no_grad():
            for param, ema_param in zip(self.denoiser.parameters(), self.ema_denoiser.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Strip keys from old checkpoints that no longer exist in the model."""
        state_dict = checkpoint.get("state_dict", {})
        removed_prefixes = ["denoiser.dataset_embedder", "denoiser.validity_embedder",
                            "ema_denoiser.dataset_embedder", "ema_denoiser.validity_embedder"]
        keys_to_remove = [k for k in state_dict if any(k.startswith(p) for p in removed_prefixes)]
        for k in keys_to_remove:
            log.info(f"on_load_checkpoint: dropping obsolete key '{k}'")
            del state_dict[k]

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_mp20/loss",
                    "interval": "epoch",
                    "frequency": self.hparams.scheduler_frequency,
                },
            }
        return {"optimizer": optimizer}