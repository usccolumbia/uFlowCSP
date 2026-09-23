"""
MeanFlow training loss and sampling for 1D sequence latents (Crystal / Molecule VAE latents).

Implements:
  - Lognormal (t, r) time-pair sampling
  - JVP-based self-consistency loss with adaptive L2 weighting
  - Implicit CFG distillation baked into the training target (fixed cfg_scale)
  - Multi-step sampler (num_steps=1 for true one-step generation)

Reference: "Mean Flows for One-step Generative Modeling" (Geng et al., 2025)
           Unofficial PyTorch code: https://github.com/haidog-yaqub/MeanFlow
"""

from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def stopgrad(x: torch.Tensor) -> torch.Tensor:
    """Stop-gradient (detach) — shorthand used throughout MeanFlow."""
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

    The stop-gradient on w makes it an adaptive reweighting without
    introducing second-order terms into the backward graph.
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


class MeanFlowTransport:
    """
    MeanFlow transport: JVP self-consistency training loss + multi-step sampler.

    CFG distillation is baked into training via cfg_scale:
      v_hat = cfg_scale * v_true + (1 - cfg_scale) * u_uncond
    where u_uncond is the model output at null spacegroup (index 0).

    At inference, a single forward pass with the real conditioning suffices —
    no double-forward CFG needed.

    Args:
        flow_ratio:   Fraction of samples where r = t (pure velocity supervision,
                      no self-consistency term).  Default: 0.5.
        time_dist:    [mode, mu, sigma] for sampling t and r.
                      mode='lognorm': sigmoid of N(mu, sigma). Default: lognorm, -0.4, 1.0.
                      mode='uniform': uniform on [0, 1].
        cfg_ratio:    Fraction of batch where spacegroup is nullified (CFG dropout).
                      These samples use raw velocity v as the target. Default: 0.1.
        cfg_scale:    Scale for CFG distillation target. Set to None or 1.0 to disable.
                      Default: 2.0.
        jvp_api:      'autograd' (torch.autograd.functional.jvp, default).
                      'funtorch' (torch.func.jvp) — needs functorch, Flash Attn-free.
        coord_loss_weight / lattice_loss_weight / coord_dim / lattice_dim:
                      CrystalFlow-style per-component loss weighting, mirroring
                      PMFRawTransport. Only activates when both dims are set and
                      the two weights differ; otherwise the loss falls back to a
                      single pooled adaptive_l2_loss over all channels, exactly
                      as before. REQUIRED for torus_coords=True: fractional
                      coordinate velocities live in [-0.5, 0.5) while lattice
                      channels are in Angstrom, so a single pooled per-sample MSE
                      is dominated by the lattice by ~2 orders of magnitude and
                      the coordinate head receives almost no gradient. Splitting
                      gives each component its own adaptive weight
                      1/(delta^2 + c), which self-normalises the two scales.
        lattice_per_structure / lattice_prior_sigma / lattice_prior_mean:
                      Treat the lattice channels as ONE quantity per structure
                      rather than N independent per-token ones - which is what
                      the dataset stores and what the decode reads back. OFF by
                      default, so existing runs are bit-identical. Same flags,
                      semantics and defaults as PMFRawTransport. Recommended for
                      any new raw-CSP run, and effectively required for the
                      lattice to carry per-structure randomness at NFE > 1.
    """

    def __init__(
        self,
        flow_ratio: float = 0.50,
        time_dist: list = None,
        cfg_ratio: float = 0.10,
        cfg_scale: float = 2.0,
        jvp_api: str = "autograd",
        atom_loss_weight: float = 0.0,
        # ------------------------------------------------------------------
        # Per-component (coord vs lattice) loss weighting - inert by default
        # (dims 0 and equal weights) so existing runs are bit-identical.
        # ------------------------------------------------------------------
        coord_loss_weight: float = 1.0,
        lattice_loss_weight: float = 1.0,
        coord_dim: int = 0,
        lattice_dim: int = 0,
        # ------------------------------------------------------------------
        # Spacegroup symmetry projection (CrystalFlow-style) - OFF by default
        # so behaviour of existing runs is unchanged.
        # ------------------------------------------------------------------
        proj_lattice: bool = False,           # stage 1: lattice metric-tensor projection
        proj_coords_rotavg: bool = False,     # stage 2: coord rotational averaging
        proj_coords_anchor: bool = False,     # stage 3: Wyckoff anchor (NotImplemented)
        sym_loss_weight: float = 0.0,         # aux loss ||u - sym(u)||^2
        sym_coord_dim: int = 0,               # raw mode: 3; latent mode: 0 (disabled)
        sym_lattice_dim: int = 0,             # raw mode: 9; latent mode: 0 (disabled)
        sym_coords_are_fractional: bool = False,
        # ------------------------------------------------------------------
        # Torus (fractional-coordinate) transport - OFF by default so
        # behaviour of existing runs is unchanged. When enabled, the first
        # ``torus_coord_dim`` channels of x are treated as fractional
        # coordinates living on [0, 1)^3 with periodic boundary conditions:
        # the noising path uses a wrapped (minimum-image) displacement and a
        # uniform prior instead of the plain Euclidean straight-line path
        # with a Gaussian prior. Remaining channels (lattice, etc.) are
        # unaffected and keep using the original Euclidean path.
        # ------------------------------------------------------------------
        torus_coords: bool = False,
        torus_coord_dim: int = 3,
        # ------------------------------------------------------------------
        # Per-structure lattice geometry - OFF by default so every existing
        # MeanFlowTransport run/checkpoint is bit-identical.
        #
        # The dataset stores ONE lattice per structure; ``encode_batch``
        # broadcasts it onto every token and ``_generated_batch_to_array_dicts``
        # pools it back with a mean over the ACTIVE tokens. Treating the lattice
        # channels as N independent per-token quantities therefore breaks the
        # invariant in three places:
        #
        #   1. ``sample_prior`` subtracts the across-token mean from EVERY
        #      channel - a translational-invariance device that only makes sense
        #      for Cartesian coordinates. Applied to the lattice channels it
        #      drives the POOLED lattice of the t=1 prior to ~0, so the prior
        #      carries no per-structure lattice randomness at all (only the
        #      residual from padding tokens, which the decode never sees).
        #   2. the training noise is drawn i.i.d. per token, so z(t) carries N
        #      different lattices for a structure whose clean target carries one.
        #   3. the network's per-token lattice output is never pooled during
        #      training or sampling, but IS pooled on decode - a train/sample
        #      mismatch on the lattice channels.
        #
        # Identical in semantics, defaults and channel layout to
        # ``PMFRawTransport``'s flags of the same name, so the pMF-vs-velocity
        # A/B stays apples-to-apples. Requires coord_dim > 0 and lattice_dim > 0
        # (the channel split must be known).
        # ------------------------------------------------------------------
        lattice_per_structure: bool = False,
        lattice_prior_sigma: float = 1.0,
        lattice_prior_mean: list = None,
    ):
        if time_dist is None:
            time_dist = ["lognorm", -0.4, 1.0]
        self.flow_ratio = flow_ratio
        self.time_dist = time_dist
        self.cfg_ratio = cfg_ratio
        self.cfg_scale = cfg_scale
        self.jvp_api = jvp_api
        self.atom_loss_weight = float(atom_loss_weight)
        self.coord_loss_weight = float(coord_loss_weight)
        self.lattice_loss_weight = float(lattice_loss_weight)
        self.coord_dim = int(coord_dim)
        self.lattice_dim = int(lattice_dim)
        self.last_sample_atom_types = None
        self.last_sample_mask = None
        self.last_atom_ce = None

        # Symmetry projection config
        self.proj_lattice = bool(proj_lattice)
        self.proj_coords_rotavg = bool(proj_coords_rotavg)
        self.proj_coords_anchor = bool(proj_coords_anchor)
        self.sym_loss_weight = float(sym_loss_weight)
        self.sym_coord_dim = int(sym_coord_dim)
        self.sym_lattice_dim = int(sym_lattice_dim)
        self.sym_coords_are_fractional = bool(sym_coords_are_fractional)
        self.last_sym_loss = None

        # Torus transport config
        self.torus_coords = bool(torus_coords)
        self.torus_coord_dim = int(torus_coord_dim)

        # Per-structure lattice geometry config (see the __init__ note above).
        self.lattice_per_structure = bool(lattice_per_structure)
        self.lattice_prior_sigma = float(lattice_prior_sigma)
        self.lattice_prior_mean = (
            None if lattice_prior_mean is None
            else [float(v) for v in lattice_prior_mean]
        )
        self.last_lattice_loss = None  # API parity with PMFRawTransport
        if self.lattice_per_structure:
            assert self.coord_dim > 0 and self.lattice_dim > 0, (
                "lattice_per_structure=True needs coord_dim and lattice_dim set "
                "so the lattice channel range is known (got coord_dim="
                f"{self.coord_dim}, lattice_dim={self.lattice_dim})."
            )
            if self.lattice_prior_mean is not None:
                assert len(self.lattice_prior_mean) == self.lattice_dim, (
                    "lattice_prior_mean needs lattice_dim entries (got "
                    f"{len(self.lattice_prior_mean)} for lattice_dim="
                    f"{self.lattice_dim})."
                )

        if self.proj_coords_anchor:
            # Fail loudly at construction time rather than silently mid-training.
            from src.models.symmetry import project_coords_anchor  # noqa: F401
            raise NotImplementedError(
                "proj_coords_anchor=True is not yet supported; Wyckoff "
                "preprocessing must be added to the raw datasets first."
            )

        if (self.proj_lattice or self.proj_coords_rotavg) and (
            self.sym_coord_dim <= 0 or self.sym_lattice_dim <= 0
        ):
            raise ValueError(
                "Symmetry projection enabled but sym_coord_dim / "
                "sym_lattice_dim are not set (raw mode requires 3 / 9)."
            )

        assert jvp_api in ("autograd", "funtorch"), (
            "jvp_api must be 'autograd' or 'funtorch'"
        )

    # ------------------------------------------------------------------
    # Backward-compatibility shim for old pickled checkpoints
    # ------------------------------------------------------------------

    # Attributes added after the initial release that may be absent on
    # MeanFlowTransport instances restored from old checkpoints via
    # Lightning's save_hyperparameters / pickle mechanism.
    _COMPAT_DEFAULTS: dict = {
        "torus_coords": False,
        "torus_coord_dim": 3,
        # Per-structure lattice geometry: absent on every checkpoint saved
        # before it existed, and False there is the correct reconstruction -
        # those runs were trained under the per-token lattice prior.
        "lattice_per_structure": False,
        "lattice_prior_sigma": 1.0,
        "lattice_prior_mean": None,
        "last_lattice_loss": None,
        "proj_lattice": False,
        "proj_coords_rotavg": False,
        "proj_coords_anchor": False,
        "sym_loss_weight": 0.0,
        "sym_coord_dim": 0,
        "sym_lattice_dim": 0,
        "sym_coords_are_fractional": False,
        "atom_loss_weight": 0.0,
        "coord_loss_weight": 1.0,
        "lattice_loss_weight": 1.0,
        "coord_dim": 0,
        "lattice_dim": 0,
        "last_sample_atom_types": None,
        "last_sample_mask": None,
        "last_atom_ce": None,
        "last_sym_loss": None,
    }

    def __setstate__(self, state: dict):
        # Called by pickle when restoring the instance. Inject defaults for
        # any attributes that did not exist when the checkpoint was saved, so
        # that old pickled instances work with new code without AttributeErrors.
        for k, v in self._COMPAT_DEFAULTS.items():
            if k not in state:
                state[k] = v
        self.__dict__.update(state)

    # ------------------------------------------------------------------
    # Symmetry projection helper
    # ------------------------------------------------------------------

    def _sym_enabled(self) -> bool:
        return self.proj_lattice or self.proj_coords_rotavg

    def _apply_sym(self, x: torch.Tensor, spacegroup, mask,
                   velocity: bool = False,
                   lattice_source: torch.Tensor = None) -> torch.Tensor:
        """Apply enabled symmetry projections to x = [coords | lattice | extras].

        ``velocity=True``: x is a velocity / noise field, not a structure tensor.
        In that case stage 1 (lattice metric-tensor Cholesky projection) is
        intentionally skipped — it requires a valid lattice matrix and is
        undefined for velocity/noise.  Stage 2 (coord rotational averaging)
        is valid for vector fields and is still applied.

        ``lattice_source`` (only consulted when ``velocity=True``): the
        structure-shaped tensor whose lattice should be used for the
        cart↔frac conversion in stage 2.  Without this, stage 2 would invert
        the velocity's own lattice channels — which is meaningless and was
        the cause of the RA-mode training collapse.

        Fast no-op when no flags are set.  ``spacegroup`` may be ``None`` for
        unconditional paths — in that case projection is skipped.
        """
        if not self._sym_enabled() or spacegroup is None:
            return x
        # Lazy import to avoid the pymatgen dependency at module load when
        # symmetry projection is off (which is the default).
        from src.models import symmetry as _sym

        if self.proj_lattice and not velocity:
            # Stage 1 only makes sense for actual structure tensors (x_1, z).
            # Never apply to noise or velocity fields.
            x = _sym.project_lattice_to_spacegroup(
                x, spacegroup, mask,
                coord_dim=self.sym_coord_dim,
                lattice_dim=self.sym_lattice_dim,
            )
        if self.proj_coords_rotavg:
            x = _sym.project_coords_rotavg(
                x, spacegroup, mask,
                coord_dim=self.sym_coord_dim,
                lattice_dim=self.sym_lattice_dim,
                coords_are_fractional=self.sym_coords_are_fractional,
                # When projecting a velocity, x's own lattice channels are a
                # velocity-of-lattice, not a lattice.  Use the structure
                # tensor's lattice if the caller supplied one.
                lattice_source=lattice_source if velocity else None,
            )
        return x

    # ------------------------------------------------------------------
    # Torus (fractional-coordinate) transport helpers
    # ------------------------------------------------------------------

    def _coord_mask(self, dim: int, device: torch.device) -> torch.Tensor:
        """Boolean (1, 1, dim) mask: True on the leading ``torus_coord_dim`` channels.

        Used to apply the torus treatment to the coordinate channels via
        ``torch.where`` while keeping x as a single tensor throughout -
        coordinates and lattice channels are never split into separate
        tensors / concatenated back together.
        """
        idx = torch.arange(dim, device=device)
        return (idx < self.torus_coord_dim).view(1, 1, dim)

    # ------------------------------------------------------------------
    # Per-structure lattice helpers (all inert unless lattice_per_structure).
    # Mirrors PMFRawTransport's implementation exactly so the two transports
    # can be A/B'd at a matched setting.
    # ------------------------------------------------------------------

    def _lattice_mask(self, dim: int, device: torch.device) -> torch.Tensor:
        """Boolean (1, 1, dim) mask: True on the lattice channel range.

        The lattice occupies channels ``[coord_dim, coord_dim + lattice_dim)``,
        matching the layout ``encode_batch`` packs and the split used by the
        per-component loss weights.
        """
        idx = torch.arange(dim, device=device)
        in_range = (idx >= self.coord_dim) & (idx < self.coord_dim + self.lattice_dim)
        return in_range.view(1, 1, dim)

    def _draw_lattice_noise(
        self,
        batch_size: int,
        num_nodes: int,
        dim: int,
        device: torch.device,
        dtype: torch.dtype,
        noise_scale: float = 1.0,
    ) -> torch.Tensor:
        """One lattice noise vector per STRUCTURE, broadcast over the tokens.

        Never mean-subtracted across tokens: the lattice is a per-structure
        quantity, so removing its across-token mean would remove the quantity
        itself (that centering only belongs to Cartesian coordinates).

        Returns a (B, N, dim) view; only its lattice channels are ever read
        (see ``_lattice_mask``).
        """
        noise = torch.randn(batch_size, 1, dim, device=device, dtype=dtype)
        noise = noise * (self.lattice_prior_sigma * noise_scale)
        if self.lattice_prior_mean is not None:
            offset = torch.zeros(dim, device=device, dtype=dtype)
            offset[self.coord_dim:self.coord_dim + self.lattice_dim] = torch.as_tensor(
                self.lattice_prior_mean, device=device, dtype=dtype
            )
            noise = noise + offset.view(1, 1, dim)
        return noise.expand(batch_size, num_nodes, dim)

    def _apply_lattice_noise(self, e: torch.Tensor, noise_scale: float = 1.0) -> torch.Tensor:
        """Overwrite the lattice channels of a noise tensor with a per-structure draw."""
        if not self.lattice_per_structure:
            return e
        B, N, d = e.shape
        lat = self._draw_lattice_noise(B, N, d, e.device, e.dtype, noise_scale)
        return torch.where(self._lattice_mask(d, e.device), lat, e)

    def _pool_lattice_channels(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Replace the lattice channels of ``x`` by their mask-mean over tokens.

        The decode (``meanflow_raw_module``) recovers the lattice as the mean
        over ACTIVE tokens, so the network's effective lattice prediction is
        that pooled value. Applying the pooling inside the transport makes
        training, the JVP and the sampler all operate on the same quantity the
        decode reads, instead of on N loosely-coupled per-token copies.

        Applied to the predicted VELOCITY here (MeanFlow's network output),
        which is the exact analogue of pooling pMF's x-prediction: the Euler
        step is linear, so pooling u keeps z's lattice constant across tokens
        whenever the prior's already is.

        Pooling is linear, hence smooth - safe inside ``torch.func.jvp``.
        """
        if not self.lattice_per_structure:
            return x
        m = mask[..., None].to(x.dtype)                          # (B, N, 1)
        denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)        # (B, 1, 1)
        pooled = (x * m).sum(dim=1, keepdim=True) / denom        # (B, 1, d)
        lattice_mask = self._lattice_mask(x.shape[-1], x.device)
        return torch.where(lattice_mask, pooled.expand_as(x), x)

    def sample_prior(
        self,
        batch_size: int,
        num_nodes: int,
        dim: int,
        device: torch.device,
        noise_scale: float = 1.0,
    ) -> torch.Tensor:
        """Sample the initial noise tensor z used at t=1 (training noise / sampling start).

        Default (``torus_coords=False``): standard centered Gaussian over the
        full channel dimension - identical to the previous inline code.

        When ``torus_coords=True``: the leading ``torus_coord_dim`` channels
        (fractional coordinates) are drawn uniformly on [0, 1) - the natural
        prior on the torus - while the remaining channels keep the centered
        Gaussian prior. Both candidates are computed over the full tensor and
        combined with a mask, so x is never split into separate pieces.

        ``noise_scale`` is a sampling-diversity temperature (default 1.0 =
        training-time prior); it scales only the Gaussian channels, never the
        torus-uniform coordinate channels.
        """
        gauss = torch.randn(batch_size, num_nodes, dim, device=device)
        gauss = gauss - gauss.mean(dim=-2, keepdim=True)
        gauss = gauss * noise_scale
        # Lattice channels: one draw per structure, broadcast, NOT mean-
        # subtracted (inert unless lattice_per_structure). Without this the
        # centering above cancels exactly the pooled lattice the decode reads.
        gauss = self._apply_lattice_noise(gauss, noise_scale)
        if not self.torus_coords:
            return gauss

        uniform = torch.rand(batch_size, num_nodes, dim, device=device)
        coord_mask = self._coord_mask(dim, device)
        return torch.where(coord_mask, uniform, gauss)

    @staticmethod
    def _wrap_disp(diff: torch.Tensor) -> torch.Tensor:
        """Wrap a fractional-coordinate displacement to the minimum image in [-0.5, 0.5)."""
        return diff - torch.round(diff)

    def _build_noisy_and_velocity(self, x_1: torch.Tensor, t_: torch.Tensor):
        """Build the noisy latent z(t) and its true instantaneous velocity v.

        Default path (``torus_coords=False``): unchanged plain Euclidean
        straight-line interpolation with a centered Gaussian noise prior.

        Torus path (``torus_coords=True``): the leading ``torus_coord_dim``
        channels use a wrapped (minimum-image) displacement and a uniform
        prior on [0, 1), so the interpolation is a geodesic on the torus
        instead of a chord through Euclidean space, while the remaining
        channels (lattice, etc.) keep the original Euclidean path. Both
        candidates are computed over the full ``x_1`` tensor and combined
        with a mask - x_1 is never split into separate coordinate/lattice
        tensors.
        """
        e_gauss = torch.randn_like(x_1)
        e_gauss = e_gauss - e_gauss.mean(dim=-2, keepdim=True)
        # Same per-structure lattice draw as sample_prior, so the training
        # noise and the t=1 sampling prior are the same distribution.
        e_gauss = self._apply_lattice_noise(e_gauss)
        diff_plain = e_gauss - x_1

        if not self.torus_coords:
            z = x_1 + t_ * diff_plain
            return z, diff_plain

        coord_mask = self._coord_mask(x_1.shape[-1], x_1.device)
        e_uniform = torch.rand_like(x_1)
        diff_wrapped = self._wrap_disp(e_uniform - x_1)

        diff = torch.where(coord_mask, diff_wrapped, diff_plain)
        z = x_1 + t_ * diff
        # Only the coordinate channels need wrapping back onto [0, 1).
        z = torch.where(coord_mask, torch.remainder(z, 1.0), z)
        return z, diff

    def _euler_step_coords(self, z: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
        """Apply an Euler step, wrapping the coordinate channels back onto [0, 1).

        No-op change vs. plain ``z - dt * u`` when ``torus_coords=False``.
        Computed over the full tensor and combined with a mask, rather than
        slicing z into separate coordinate/lattice tensors.
        """
        z_next = z - dt * u
        if not self.torus_coords:
            return z_next
        coord_mask = self._coord_mask(z.shape[-1], z.device)
        return torch.where(coord_mask, torch.remainder(z_next, 1.0), z_next)

    # ------------------------------------------------------------------
    # Time sampling
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
    # Training loss
    # ------------------------------------------------------------------

    def loss(
        self,
        model: nn.Module,
        x_1: torch.Tensor,  # (B, N, d)  clean latents from VAE
        cond_dict: dict,
        ema_model: nn.Module = None,
    ):
        """
        MeanFlow self-consistency adaptive L2 loss.

        Args:
            ema_model: If provided, used for the u_uncond CFG target computation
                       instead of the training model. Stabilises training targets.

        Returns:
            loss:    scalar training loss (adaptive L2 with stop-grad weights)
            mse_val: scalar raw MSE for monitoring (not backpropagated)
        """
        B, N, d = x_1.shape
        device = x_1.device
        mask = cond_dict["mask"]           # (B, N) bool
        sg_for_sym = cond_dict.get("spacegroup")  # (B,) long or None

        # Project the clean target onto the SG-allowed subspace (no-op when
        # all symmetry flags are off — preserves existing behaviour).
        x_1 = self._apply_sym(x_1, sg_for_sym, mask)

        def _split_output(output):
            if isinstance(output, tuple):
                return output[0], output[1]
            return output, None

        t, r = self.sample_t_r(B, device)
        t_ = t.view(B, 1, 1)              # broadcast shape
        r_ = r.view(B, 1, 1)

        # --- Build noisy input z at time t along the flow path ---
        # Default: linear path z(t) = (1-t)*x_1 + t*e, e ~ N(0, I), centered
        # noise to preserve translational invariance (crystals/molecules).
        # When torus_coords=True, the coordinate channels instead use a
        # wrapped (minimum-image) geodesic on [0, 1)^3 with a uniform prior
        # (see _build_noisy_and_velocity). Do NOT project noise e with stage
        # 1 (lattice): e is not a lattice matrix so computing G = e^T e and
        # Cholesky is undefined / unstable. Stage 2 (coord rotavg) on a
        # noise vector is also questionable, so skip symmetry projection for
        # e entirely.
        z, v = self._build_noisy_and_velocity(x_1, t_)  # (B, N, d) each

        # --- CFG distillation: build guided target velocity v_hat ---
        use_cfg = (self.cfg_scale is not None) and (self.cfg_scale > 1.0)

        if use_cfg:
            null_spacegroup = torch.zeros_like(cond_dict["spacegroup"])  # index 0 = null
            atom_types = cond_dict.get("atom_types")

            # Use EMA model for u_uncond if available — stabilises training targets
            uncond_model = ema_model if ema_model is not None else model
            with torch.no_grad():
                # Unconditional prediction at null spacegroup; r=t collapses the
                # self-consistency interval to zero (pure velocity prediction)
                if atom_types is None:
                    uncond_output = uncond_model(
                        z, t, t,
                        spacegroup=null_spacegroup,
                        mask=mask,
                    )
                else:
                    uncond_output = uncond_model(
                        x=z,
                        atom_types=atom_types,
                        t=t,
                        r=t,
                        spacegroup=null_spacegroup,
                        mask=mask,
                    )

                u_uncond, _ = _split_output(uncond_output)
                # One lattice per structure: pool the unconditional velocity the
                # same way the conditional one is pooled below, so the guided
                # target v_hat mixes two quantities on the same footing.
                u_uncond = self._pool_lattice_channels(u_uncond, mask)

            # Guided velocity target
            v_hat = self.cfg_scale * v + (1.0 - self.cfg_scale) * u_uncond

            # CFG dropout mask: cfg_ratio fraction of samples use raw v as target
            # (they are also assigned null spacegroup in the main forward)
            cfg_mask = (torch.rand(B, device=device) < self.cfg_ratio)  # (B,) bool
            v_hat = torch.where(cfg_mask.view(B, 1, 1), v, v_hat)

            # Null out spacegroup for CFG-masked samples in the main forward pass
            spacegroup_cond = torch.where(
                cfg_mask, null_spacegroup, cond_dict["spacegroup"]
            )
        else:
            v_hat = v
            spacegroup_cond = cond_dict["spacegroup"]

        # --- JVP: get u and du/dt for self-consistency ---
        # We differentiate model(z, t, r, ...) along the direction (v_hat, 1, 0):
        #   d/dt[ model(z(t), t, r) ] ≈ J_z·v_hat + J_t·1 + J_r·0
        # This is the total time-derivative of the mean velocity along the path.
        #
        # Captured (non-differentiated) context variables:
        _mask = mask
        _atom_types = cond_dict.get("atom_types")
        _sg_for_sym = sg_for_sym
        # NOTE: we project the *cond* spacegroup, but symmetry projection
        # itself uses the per-sample SG (CFG dropouts get SG=0 = no-op).
        # Pre-extract as a plain Python list so it can be used inside
        # torch.func.jvp — dual tensors inside jvp have no storage and
        # cannot be accessed via .detach().cpu().tolist().
        _sg_proj = spacegroup_cond if self._sym_enabled() else None
        _sg_proj_list = (
            _sg_proj.detach().cpu().tolist() if _sg_proj is not None else None
        )

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

            continuous_out, _ = _split_output(output)
            # One lattice per structure: pool the predicted velocity's lattice
            # channels so the JVP differentiates the same quantity the decode
            # reads back. Pooling is linear, hence smooth - safe inside
            # torch.func.jvp. Inert unless lattice_per_structure.
            continuous_out = self._pool_lattice_channels(continuous_out, _mask)
            # Project predicted velocity (velocity=True: skip lattice Cholesky,
            # only apply coord rotavg if enabled) so dudt is consistent.
            # Pass z_ as lattice_source so stage-2 cart↔frac uses the
            # *structure*'s lattice, not the velocity's lattice channels
            # (pinv on a velocity tensor is numerical garbage).
            if _sg_proj_list is not None:
                continuous_out = self._apply_sym(continuous_out, _sg_proj_list, _mask,
                                                 velocity=True,
                                                 lattice_source=z_)
            return continuous_out

        jvp_inputs = (z, t, r)
        jvp_tangents = (v_hat, torch.ones_like(t), torch.zeros_like(r))

        if self.jvp_api == "autograd":
            u, dudt = torch.autograd.functional.jvp(
                model_fn, jvp_inputs, jvp_tangents, create_graph=True
            )
            dudt = dudt.detach()  # free the unused 2nd-order graph; only u needs grad_fn
        else:  # funtorch
            u, dudt = torch.func.jvp(model_fn, jvp_inputs, jvp_tangents)

        # --- Self-consistency target ---
        # u_tgt: the mean velocity that self-consistently maps z(t) to x_1
        # When r = t: (t_ - r_) = 0 so u_tgt = v_hat (pure velocity supervision)
        u_tgt = v_hat - (t_ - r_) * dudt   # (B, N, d)

        # --- Adaptive L2 loss ---
        error = u - stopgrad(u_tgt)

        # CrystalFlow-style per-component loss weights (same convention as
        # PMFRawTransport). Only activates when both dims are set and the
        # weights differ; otherwise this is the original single pooled
        # adaptive_l2_loss over all channels — bit-identical to before.
        #
        # Splitting matters most under torus_coords=True: each component then
        # gets its own adaptive weight 1/(delta^2 + c), so the fractional
        # coordinate channels (|v| <= 0.5) are no longer drowned out by the
        # Angstrom-scale lattice channels inside one pooled per-sample MSE.
        _split = (
            self.coord_dim > 0
            and self.lattice_dim > 0
            and (
                self.coord_loss_weight != self.lattice_loss_weight
                # The pooled lattice loss below is a different normalisation,
                # so it must engage even when the two weights are equal.
                or self.lattice_per_structure
            )
        )
        self.last_lattice_loss = None
        if _split:
            e_coord = error[..., :self.coord_dim]
            e_lattice = error[..., self.coord_dim:self.coord_dim + self.lattice_dim]
            if self.lattice_per_structure:
                # One lattice per structure -> score it ONCE per structure.
                # Mask-mean the per-token error into a single (B, lattice_dim)
                # vector and run adaptive-L2 over a single all-ones token,
                # identical to PMFRawTransport / MeanFlowCSPTransport. Without
                # this the lattice delta^2 is a per-token mean here and a
                # per-structure mean there, so the same coord:lattice ratio
                # means two different things across transports.
                _tok = mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)   # (B, 1)
                lattice_global_error = (
                    e_lattice * mask[..., None].float()
                ).sum(dim=1) / _tok                                           # (B, lattice_dim)
                lattice_loss = adaptive_l2_loss(
                    lattice_global_error.unsqueeze(1),
                    torch.ones((B, 1), dtype=mask.dtype, device=device),
                )
            else:
                lattice_loss = adaptive_l2_loss(e_lattice, mask)
            loss = (
                self.coord_loss_weight * adaptive_l2_loss(e_coord, mask)
                + self.lattice_loss_weight * lattice_loss
            )
            self.last_lattice_loss = lattice_loss.detach()
        else:
            loss = adaptive_l2_loss(error, mask)

        # Optional symmetry consistency auxiliary loss.
        # Even when projections are applied inside model_fn, this term acts
        # as a stronger soft constraint encouraging u to stay symmetric.
        self.last_sym_loss = None
        if self.sym_loss_weight > 0.0 and self._sym_enabled() and sg_for_sym is not None:
            # velocity=True + lattice_source=z together fix both the
            # velocity=False Cholesky bug and the pinv-on-velocity bug.
            u_sym = self._apply_sym(u, sg_for_sym, mask, velocity=True,
                                    lattice_source=z)
            sym_diff = u - u_sym
            sym_loss = (
                (sym_diff.pow(2) * mask[..., None].to(sym_diff.dtype)).sum()
                / mask.to(sym_diff.dtype).sum().clamp(min=1.0)
                / d
            )
            loss = loss + self.sym_loss_weight * sym_loss
            self.last_sym_loss = sym_loss.detach()

        # Optional categorical loss for joint atom-type generation.
        self.last_atom_ce = None
        if self.atom_loss_weight > 0.0 and _atom_types is not None:
            if _atom_types is None:
                atom_output = None
            else:
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
    # Inference sampler
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        z: torch.Tensor,        # (B, N, d)  initial noise (centered)
        cond_dict: dict,
        num_steps: int = 1,
        record_traj: bool = False,
    ) -> torch.Tensor:
        """
        Multi-step MeanFlow sampler.

        num_steps=1  → true one-step generation (the MeanFlow promise).
        num_steps>1  → iterative refinement (better quality, more compute).

        No double-forward CFG needed: guidance is already encoded in the
        model weights via the training-time CFG distillation.

        ``record_traj`` (default False, opt-in) additionally stores the state
        after every Euler step on ``self.last_sample_traj`` as a list of
        ``num_steps + 1`` detached clones — index 0 is the prior (t=1), index
        ``num_steps`` is the returned sample (t=0). It is read only by the
        figure tooling (``src/export_trajectory_meanflow.py``); with the
        default it costs two branch tests and changes nothing about the
        returned tensor.
        """
        model.eval()
        self.last_sample_atom_types = None
        self.last_sample_mask = None
        self.last_sample_traj = None
        traj = [] if record_traj else None

        def _split_output(output):
            if isinstance(output, tuple):
                return output[0], output[1]
            return output, None

        sg_for_sym = cond_dict.get("spacegroup") if self._sym_enabled() else None
        mask_for_sym = cond_dict["mask"]

        # Initial z is pure noise; skip stage 1 (Cholesky) on noise.
        # Stage 2 (coord rotavg) on noise is also skipped via velocity=True.
        z = self._apply_sym(z, sg_for_sym, mask_for_sym, velocity=True)
        if traj is not None:
            traj.append(z.detach().clone())

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
            # Pool the lattice channels exactly as training does, so every token
            # carries the one lattice the decode will average back out.
            u = self._pool_lattice_channels(u, cond_dict["mask"])
            if atom_logits is not None:
                predicted_atom_types = atom_logits.argmax(dim=-1)
                predicted_atom_types = torch.where(
                    predicted_atom_types > 118,
                    torch.zeros_like(predicted_atom_types),
                    predicted_atom_types,
                )

                if cond_dict.get("generate_atom_types", False):
                    # Joint A/X/L generation: update conditioning and expose
                    # predictions so the caller can build the output structure.
                    self.last_sample_atom_types = predicted_atom_types.detach()
                    self.last_sample_mask = (predicted_atom_types > 0).detach()
                    cond_dict = dict(cond_dict)
                    cond_dict["atom_types"] = predicted_atom_types
                    cond_dict["mask"] = predicted_atom_types > 0
                # else: formula/conditioning mode — atom types are fixed by the
                # caller; do NOT overwrite last_sample_atom_types, which stays
                # None so _resolve_sample_atom_state falls back to the formula.

            # Euler step: move z along mean velocity u
            dt = t_vals[i].item() - t_vals[i + 1].item()
            # u is a velocity field: skip stage 1 (Cholesky), apply stage 2 only.
            # Pass z as lattice_source so stage-2 inverts the current
            # structure's lattice, not the velocity's lattice channels.
            u = self._apply_sym(u, sg_for_sym, mask_for_sym, velocity=True,
                                lattice_source=z)
            # Euler step; wraps the coordinate channels back onto [0, 1)
            # when torus_coords=True (no-op otherwise).
            z = self._euler_step_coords(z, u, dt)
            # z is now an accumulated structure tensor; apply full projection
            # (stage 1 + 2) to keep the lattice in the SG-allowed subspace.
            z = self._apply_sym(z, sg_for_sym, mask_for_sym)
            if traj is not None:
                traj.append(z.detach().clone())

        if traj is not None:
            self.last_sample_traj = traj
        return z
