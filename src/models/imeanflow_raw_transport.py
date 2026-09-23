"""
Improved MeanFlow (iMF) transport for RAW atomic data (no VAE).

Key difference from MeanFlowTransport (meanflow_transport.py):
  Instead of regressing the mean velocity u against a self-referential target
  u_tgt = v_hat - (t-r)*dudt  (which depends on the network via dudt),
  iMF derives the network's implied instantaneous velocity:

      v_pred = u + (t - r) * dudt

  and regresses it against the purely data-derived v_hat (network-independent).
  This eliminates the self-referential loop and turns training into a standard
  regression problem, improving stability.

This file is the RAW-data counterpart of imeanflow_transport.py.  It uses the
same model call signature as MeanFlowTransport in meanflow_transport.py, i.e.
  model(x=z_, atom_types=..., t=t_, r=r_, spacegroup=..., mask=...)
and is therefore a drop-in replacement for MeanFlowRawLitModule.

Reference: "Improved Mean Flows: On the Challenges of Fastforward Generative
            Models" (Geng et al., arXiv:2512.02012, Dec 2025)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def stopgrad(x: torch.Tensor) -> torch.Tensor:
    """Stop-gradient (detach) — shorthand."""
    return x.detach()


def adaptive_l2_loss(
    error: torch.Tensor,  # (B, N, d)
    mask: torch.Tensor,   # (B, N) bool — True for valid tokens
    gamma: float = 0.0,
    c: float = 1e-3,
) -> torch.Tensor:
    """
    Adaptive L2 loss from MeanFlow paper, adapted for (B, N, d) tensors.

    Loss = mean_over_batch[ sg(w_b) * delta_b^2 ]
    where  delta_b^2 = masked MSE for sample b,
           w_b       = 1 / (delta_b^2 + c)^(1 - gamma)
    """
    num_tokens = mask.float().sum(dim=1).clamp(min=1.0)  # (B,)
    d = error.shape[-1]

    masked_error = error * mask[..., None].float()  # (B, N, d)

    # Per-sample mean squared error
    delta_sq = (masked_error ** 2).sum(dim=(1, 2)) / (num_tokens * d)  # (B,)

    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)  # (B,)

    loss = (stopgrad(w) * delta_sq).mean()
    return loss


class IMeanFlowRawTransport:
    """
    Improved MeanFlow (iMF) transport for RAW atomic data: JVP loss + multi-step sampler.

    Loss formulation (iMF, Geng et al. 2025):
        v_pred = u_theta(z, t, r) + (t - r) * d/dt[u_theta(z(t), t, r)]
        loss   = adaptive_L2(v_pred - v_hat, mask)

    where v_hat is the (optionally CFG-guided) data velocity — fully
    network-independent, so the target never chases itself across iterations.

    All other aspects (time sampling, CFG distillation, atom-type loss, sampler)
    are identical to MeanFlowTransport (meanflow_transport.py).

    Args:
        flow_ratio:      Fraction of samples where r = t (pure velocity supervision).
                         Default: 0.5.
        time_dist:       [mode, mu, sigma] for sampling t and r.
                         mode='lognorm': sigmoid of N(mu, sigma). Default: lognorm,-0.4,1.0.
                         mode='uniform': uniform on [0, 1].
        cfg_ratio:       Fraction of batch where spacegroup is nullified (CFG dropout).
                         Default: 0.1.
        cfg_scale:       Scale for CFG distillation target. None or 1.0 disables.
                         Default: 2.0.
        jvp_api:         'funtorch' (torch.func.jvp, recommended) or
                         'autograd' (torch.autograd.functional.jvp).
        detach_dudt_in_loss: If True, stop-gradient is applied to dudt before
                         computing v_pred = u + (t-r)*dudt.  This removes the
                         higher-order gradient path through dudt, significantly
                         reducing GPU memory at the cost of a slightly looser
                         iMF update.  Useful when running on A100-40 GB.
                         Default: False (strict iMF, full gradient).
        atom_loss_weight: Cross-entropy weight for joint atom-type generation loss.
                         Default: 1.0 (matches meanflow_raw.yaml).
    """

    def __init__(
        self,
        flow_ratio: float = 0.50,
        time_dist: list = None,
        cfg_ratio: float = 0.10,
        cfg_scale: float = 2.0,
        jvp_api: str = "funtorch",
        detach_dudt_in_loss: bool = False,
        atom_loss_weight: float = 1.0,
        coord_loss_weight: float = 1.0,
        lattice_loss_weight: float = 1.0,
        coord_dim: int = 0,
        lattice_dim: int = 0,
    ):
        if time_dist is None:
            time_dist = ["lognorm", -0.4, 1.0]
        self.flow_ratio = flow_ratio
        self.time_dist = time_dist
        self.cfg_ratio = cfg_ratio
        self.cfg_scale = cfg_scale
        self.jvp_api = jvp_api
        self.detach_dudt_in_loss = bool(detach_dudt_in_loss)
        self.atom_loss_weight = float(atom_loss_weight)
        self.coord_loss_weight = float(coord_loss_weight)
        self.lattice_loss_weight = float(lattice_loss_weight)
        self.coord_dim = int(coord_dim)
        self.lattice_dim = int(lattice_dim)
        self.last_sample_atom_types = None
        self.last_sample_mask = None
        self.last_atom_ce = None
        self.last_sym_loss = None  # kept for API compatibility with MeanFlowTransport

        assert jvp_api in ("autograd", "funtorch"), (
            "jvp_api must be 'autograd' or 'funtorch'"
        )

    # ------------------------------------------------------------------
    # Time sampling  (identical to MeanFlowTransport)
    # ------------------------------------------------------------------

    def sample_t_r(self, batch_size: int, device: torch.device):
        """
        Sample time pairs (t, r) with t >= r, both in (0, 1).

        For flow_ratio fraction of the batch r is set equal to t, which
        supervises on the pure (instantaneous) velocity and helps stabilise
        early training.
        """
        mode = self.time_dist[0]
        mu, sigma = float(self.time_dist[1]), float(self.time_dist[2])

        if mode == "uniform":
            samples = np.random.rand(batch_size, 2).astype(np.float32)
        elif mode == "lognorm":
            raw = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
            samples = 1.0 / (1.0 + np.exp(-raw))  # logistic sigmoid → (0, 1)
        else:
            raise ValueError(f"Unknown time_dist mode: {mode!r}")

        # Ensure t >= r
        t_np = np.maximum(samples[:, 0], samples[:, 1])
        r_np = np.minimum(samples[:, 0], samples[:, 1])

        # Pure-velocity samples: set r = t
        num_flow = int(self.flow_ratio * batch_size)
        if num_flow > 0:
            idx = np.random.permutation(batch_size)[:num_flow]
            r_np[idx] = t_np[idx]

        t = torch.tensor(t_np, device=device)
        r = torch.tensor(r_np, device=device)
        return t, r

    # ------------------------------------------------------------------
    # Training loss  (iMF formulation — key difference vs MeanFlowTransport)
    # ------------------------------------------------------------------

    def loss(
        self,
        model: nn.Module,
        x_1: torch.Tensor,  # (B, N, d)  clean raw crystal tensors
        cond_dict: dict,
        ema_model: nn.Module = None,
    ):
        """
        iMF self-consistency adaptive L2 loss.

        Args:
            model:     Training denoiser (MeanFlowSiTRaw).
            x_1:       Clean raw crystal data (B, N, d).
            cond_dict: Conditioning dict with keys: mask, spacegroup, atom_types.
            ema_model: If provided, used for the u_uncond CFG target computation
                       instead of the training model. Stabilises training targets.

        Returns:
            loss:    scalar training loss
            mse_val: scalar raw MSE for monitoring (no backprop)
        """
        B, N, d = x_1.shape
        device = x_1.device
        mask = cond_dict["mask"]           # (B, N) bool

        def _split_output(output):
            if isinstance(output, tuple):
                return output[0], output[1]
            return output, None

        t, r = self.sample_t_r(B, device)
        t_ = t.view(B, 1, 1)
        r_ = r.view(B, 1, 1)

        # --- Build noisy input z at time t along the linear flow path ---
        # Linear path: z(t) = (1-t)*x_1 + t*e  where e ~ N(0, I)
        # Center noise to preserve translational invariance (crystals/molecules)
        e = torch.randn_like(x_1)
        e = e - e.mean(dim=-2, keepdim=True)

        z = (1.0 - t_) * x_1 + t_ * e    # (B, N, d)  noisy latent at time t
        v = e - x_1                        # (B, N, d)  true instantaneous velocity

        # --- CFG distillation: build guided target velocity v_hat ---
        # v_hat is fully network-independent — no stopgrad needed on it (iMF key property).
        use_cfg = (self.cfg_scale is not None) and (self.cfg_scale > 1.0)
        _atom_types = cond_dict.get("atom_types")

        if use_cfg:
            null_spacegroup = torch.zeros_like(cond_dict["spacegroup"])

            # Use EMA model for u_uncond if available — stabilises training targets
            uncond_model = ema_model if ema_model is not None else model
            with torch.no_grad():
                # Unconditional prediction at null spacegroup; r=t collapses the
                # self-consistency interval to zero (pure velocity prediction)
                if _atom_types is None:
                    uncond_output = uncond_model(
                        z, t, t,
                        spacegroup=null_spacegroup,
                        mask=mask,
                    )
                else:
                    uncond_output = uncond_model(
                        x=z,
                        atom_types=_atom_types,
                        t=t,
                        r=t,
                        spacegroup=null_spacegroup,
                        mask=mask,
                    )

                u_uncond, _ = _split_output(uncond_output)

            # Guided velocity target
            v_hat = self.cfg_scale * v + (1.0 - self.cfg_scale) * u_uncond

            # CFG dropout mask: cfg_ratio fraction of samples use raw v as target
            cfg_mask = (torch.rand(B, device=device) < self.cfg_ratio)  # (B,) bool
            v_hat = torch.where(cfg_mask.view(B, 1, 1), v, v_hat)

            # Null out spacegroup for CFG-masked samples in the main forward pass
            spacegroup_cond = torch.where(
                cfg_mask, null_spacegroup, cond_dict["spacegroup"]
            )
        else:
            v_hat = v
            spacegroup_cond = cond_dict["spacegroup"]

        # --- JVP: get u and du/dt for iMF ---
        # Differentiate model(z, t, r, ...) along direction (v_hat, 1, 0):
        #   du/dt ≈ J_z·v_hat + J_t·1 + J_r·0
        _mask = mask

        def model_fn(z_, t_, r_):
            if _atom_types is None:
                output = model(
                    z_, t_, r_,
                    spacegroup=spacegroup_cond,
                    mask=_mask,
                )
            else:
                output = model(
                    x=z_,
                    atom_types=_atom_types,
                    t=t_,
                    r=r_,
                    spacegroup=spacegroup_cond,
                    mask=_mask,
                )
            vel, _ = _split_output(output)
            return vel

        jvp_inputs = (z, t, r)
        jvp_tangents = (v_hat, torch.ones_like(t), torch.zeros_like(r))

        if self.jvp_api == "autograd":
            u, dudt = torch.autograd.functional.jvp(
                model_fn, jvp_inputs, jvp_tangents, create_graph=True
            )
            dudt = dudt.detach()
        else:  # funtorch
            u, dudt = torch.func.jvp(model_fn, jvp_inputs, jvp_tangents)

        # --- iMF loss ---
        # Derive the network's implied instantaneous velocity and regress against v_hat.
        # When r = t: (t_ - r_) = 0 so v_pred = u — pure velocity supervision.
        # Target v_hat is fully data-derived (no stopgrad needed — that's the iMF advantage).
        # detach_dudt_in_loss=True cuts the higher-order gradient path through dudt,
        # saving ~30-40% peak VRAM on A100 at minor theoretical cost.
        _dudt = dudt.detach() if self.detach_dudt_in_loss else dudt
        v_pred = u + (t_ - r_) * _dudt   # network's implied instantaneous velocity
        error  = v_pred - v_hat           # network-independent target

        # CrystalFlow-style per-component loss weights (coord λ=10, lattice λ=1 in CSP mode).
        # Only activates when both dims are set and weights differ; otherwise falls back to
        # uniform adaptive_l2_loss — identical to the original iMF formulation.
        _split = (
            self.coord_dim > 0
            and self.lattice_dim > 0
            and (self.coord_loss_weight != self.lattice_loss_weight)
        )
        if _split:
            e_coord   = error[..., :self.coord_dim]
            e_lattice = error[..., self.coord_dim:self.coord_dim + self.lattice_dim]
            loss = (
                self.coord_loss_weight   * adaptive_l2_loss(e_coord,   mask)
                + self.lattice_loss_weight * adaptive_l2_loss(e_lattice, mask)
            )
        else:
            loss = adaptive_l2_loss(error, mask)

        # Optional categorical loss for joint atom-type generation (same as MeanFlowTransport).
        self.last_atom_ce = None
        if self.atom_loss_weight > 0.0 and _atom_types is not None:
            atom_output = model(
                x=z,
                atom_types=_atom_types,
                t=t,
                r=r,
                spacegroup=spacegroup_cond,
                mask=_mask,
            )
            _, atom_logits = _split_output(atom_output)
            if atom_logits is not None:
                atom_targets = torch.where(mask, _atom_types, torch.zeros_like(_atom_types))
                atom_ce = F.cross_entropy(
                    atom_logits.reshape(-1, atom_logits.shape[-1]),
                    atom_targets.reshape(-1),
                    reduction="mean",
                )
                loss = loss + self.atom_loss_weight * atom_ce
                self.last_atom_ce = atom_ce.detach()

        # Raw MSE for monitoring (no backprop)
        with torch.no_grad():
            num_tok = mask.float().sum(dim=1).clamp(min=1.0)  # (B,)
            mse_val = (
                ((error.detach() ** 2) * mask[..., None].float()).sum(dim=(1, 2))
                / (num_tok * d)
            ).mean()

        return loss, mse_val

    # ------------------------------------------------------------------
    # Inference sampler  (identical to MeanFlowTransport)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        z: torch.Tensor,        # (B, N, d)  initial noise (centered)
        cond_dict: dict,
        num_steps: int = 1,
    ) -> torch.Tensor:
        """
        Multi-step MeanFlow sampler (unchanged from MeanFlowTransport).

        num_steps=1  → true one-step generation.
        num_steps>1  → iterative refinement.
        """
        model.eval()
        self.last_sample_atom_types = None
        self.last_sample_mask = None

        def _split_output(output):
            if isinstance(output, tuple):
                return output[0], output[1]
            return output, None

        # Linearly space timesteps from 1 (pure noise) to 0 (data)
        t_vals = torch.linspace(1.0, 0.0, num_steps + 1, device=z.device)

        for i in range(num_steps):
            t = torch.full((z.size(0),), t_vals[i].item(), device=z.device)
            r = torch.full((z.size(0),), t_vals[i + 1].item(), device=z.device)

            atom_types = cond_dict.get("atom_types")
            if atom_types is None:
                output = model(
                    z, t, r,
                    spacegroup=cond_dict["spacegroup"],
                    mask=cond_dict["mask"],
                )
            else:
                output = model(
                    x=z,
                    atom_types=atom_types,
                    t=t,
                    r=r,
                    spacegroup=cond_dict["spacegroup"],
                    mask=cond_dict["mask"],
                )

            u, atom_logits = _split_output(output)
            if atom_logits is not None:
                predicted_atom_types = atom_logits.argmax(dim=-1)
                predicted_atom_types = torch.where(
                    predicted_atom_types > 118,
                    torch.zeros_like(predicted_atom_types),
                    predicted_atom_types,
                )
                if cond_dict.get("generate_atom_types", False):
                    # Joint A/X/L generation: expose predictions and update conditioning.
                    # In CSP mode (generate_atom_types=False) do NOT overwrite
                    # last_sample_atom_types — _resolve_sample_atom_state must fall
                    # back to the formula atom types supplied as conditioning.
                    self.last_sample_atom_types = predicted_atom_types.detach()
                    self.last_sample_mask = (predicted_atom_types > 0).detach()
                    cond_dict = dict(cond_dict)
                    cond_dict["atom_types"] = predicted_atom_types
                    cond_dict["mask"] = predicted_atom_types > 0

            # Euler step: move z along mean velocity u
            dt = t_vals[i].item() - t_vals[i + 1].item()
            z = z - dt * u

        return z
