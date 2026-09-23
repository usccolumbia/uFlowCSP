"""
MeanFlowSiTRaw: Sequence Transformer backbone for MeanFlow generative modelling on raw atomic data.

Extends MeanFlowSiT to handle raw atom types, coordinates, and lattice parameters directly,
instead of VAE latents. Implements separate processing within a unified model.

Key differences from MeanFlowSiT:
    - in_channels tracks the flowed raw state dimensions only
  - Separate embedders for atom types (categorical), coordinates (continuous), lattice (global)
  - Unified processing with component-specific handling
"""

import math

import torch
import torch.nn as nn

# Re-use the shared building blocks from EqM's SiT to stay consistent
from src.models.denoisers.dit import (
    FinalLayer,
    LabelEmbedder,
    SiTBlock,
    TimestepEmbedder,
    get_pos_embedding,
)


class MeanFlowSiTRaw(nn.Module):
    """
    Sequence Transformer for MeanFlow on raw atomic data.

    Processes atom types, coordinates, and lattice separately within a unified model.

    Input x shape: (B, N, d) where d = 3 (coords) + 9 (lattice flattened)

    Args:
        input_size:         Max number of tokens (atoms) in a sequence.
        atom_type_vocab_size: Number of possible atom types (elements).
        atom_type_embed_dim: Embedding dimension for atom types.
        coord_dim:          Dimension of coordinates (default 3 for 3D).
        lattice_dim:        Dimension of flattened lattice (default 9 for 3x3 matrix).
        hidden_size:        Transformer hidden dimension.
        depth:              Number of SiTBlocks.
        num_heads:          Number of attention heads.
        mlp_ratio:          MLP expansion ratio in each block.
        class_dropout_prob: Dropout probability for label embedders (CFG).
        num_spacegroups:    Number of spacegroups for spacegroup conditioning.
                            Index 0 is the *null* (unconditional) class.
        coord_fourier_bands: Number of Fourier bands K used to featurise the
                            coordinate channels on input. 0 (default) = OFF:
                            the raw coordinate values are passed straight into
                            x_embedder, exactly as before.

                            K > 0 replaces the raw coords with
                            [sin(2*pi*k*f), cos(2*pi*k*f)] for k = 1..K, which
                            is *periodic by construction*. This is REQUIRED for
                            meanflow.torus_coords=True: with fractional coords
                            the target field is periodic in f, but a plain
                            nn.Linear on raw f is discontinuous across the cell
                            seam (f=0.999 and f=0.001 are adjacent on the torus
                            yet maximally distant in input space), so the model
                            cannot represent the function it is being trained
                            on. Same trick as DiffCSP / CrystalFlow.

                            ONLY valid with fractional coordinates. Leave at 0
                            for Cartesian runs — sin/cos of an Angstrom-valued
                            coordinate is meaningless.
    """

    def __init__(
        self,
        input_size: int = 100,
        atom_type_vocab_size: int = 100,  # Z up to ~118, but pad to 100+
        atom_type_embed_dim: int = 32,
        coord_dim: int = 3,
        lattice_dim: int = 9,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.1,
        num_spacegroups: int = 230,
        predict_atom_types: bool = False,
        atom_type_condition_dropout: float = 0.0,
        use_spacegroup_conditioning: bool = True,
        coord_fourier_bands: int = 0,
    ):
        super().__init__()
        self.input_size = input_size
        self.atom_type_vocab_size = atom_type_vocab_size
        self.atom_type_embed_dim = atom_type_embed_dim
        self.coord_dim = coord_dim
        self.lattice_dim = lattice_dim
        self.in_channels = coord_dim + lattice_dim

        # Periodic input featurisation of the coordinate channels. Affects the
        # INPUT width only — in_channels (the flowed state / output width) is
        # unchanged, so the transport and sampler are untouched.
        self.coord_fourier_bands = int(coord_fourier_bands)
        if self.coord_fourier_bands > 0:
            # 2*pi*k for k = 1..K, held as a constant (non-learned) buffer.
            self.register_buffer(
                "coord_fourier_freqs",
                2.0 * math.pi * torch.arange(1, self.coord_fourier_bands + 1, dtype=torch.float32),
                persistent=False,
            )
            coord_feat_dim = 2 * coord_dim * self.coord_fourier_bands
        else:
            coord_feat_dim = coord_dim
        self.coord_feat_dim = coord_feat_dim

        self.model_input_dim = atom_type_embed_dim + coord_feat_dim + lattice_dim
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.predict_atom_types = predict_atom_types
        self.atom_type_condition_dropout = float(atom_type_condition_dropout)
        self.use_spacegroup_conditioning = bool(use_spacegroup_conditioning)

        # Separate embedders for components
        self.atom_type_embedder = nn.Embedding(atom_type_vocab_size, atom_type_embed_dim)
        # Coordinates are continuous, so no embedder needed - pass through
        # Lattice is continuous, pass through

        # Combined input projection: [atom embedding | flowed state] -> hidden_size
        self.x_embedder = nn.Linear(self.model_input_dim, hidden_size, bias=True)

        # Two separate timestep embedders: current time t and target time r
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.r_embedder = TimestepEmbedder(hidden_size)

        # Conditioning embedders (same NUM_CLASSES logic as EqM: index 0 = null class)
        self.spacegroup_embedder = LabelEmbedder(num_spacegroups, hidden_size, class_dropout_prob)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            SiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

        # CRITICAL: disable fused/Flash attention in ALL blocks.
        # torch.autograd.functional.jvp is incompatible with Flash Attention.
        for block in self.blocks:
            block.attn.fused_attn = False

        # Output layer predicts velocity in the flowed continuous state and,
        # optionally, atom-type logits for joint A/X/L generation.
        final_out_dim = self.in_channels + (self.atom_type_vocab_size if self.predict_atom_types else 0)
        self.final_layer = FinalLayer(hidden_size, final_out_dim)

        self.initialize_weights()

    def initialize_weights(self):
        """Xavier-uniform init for linears, zero-init for adaLN output layers."""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, std=0.02)

        self.apply(_basic_init)

        # Embedding tables
        nn.init.normal_(self.spacegroup_embedder.embedding_table.weight, std=0.02)

        # Timestep MLPs (both t and r)
        for embedder in [self.t_embedder, self.r_embedder]:
            nn.init.normal_(embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers (standard DiT init)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def _featurize_coords(self, coords: torch.Tensor) -> torch.Tensor:
        """Coordinate channels -> input features.

        Identity pass-through when ``coord_fourier_bands == 0`` (default), so
        the Cartesian path is bit-identical to the original implementation.
        Otherwise returns [sin(2*pi*k*f), cos(2*pi*k*f)]_{k=1..K} flattened,
        which is exactly periodic in f with period 1 — the representation the
        torus transport needs. sin/cos are smooth, so this stays JVP-safe.
        """
        # getattr, not self.x: Lightning pickles the denoiser OBJECT into hparams,
        # so checkpoints saved before this attribute existed are unpickled without
        # __init__ ever running and have no `coord_fourier_bands`. Defaulting to 0
        # keeps every pre-existing checkpoint loadable on the unchanged path.
        bands = int(getattr(self, "coord_fourier_bands", 0))
        if bands <= 0:
            return coords
        # (B, N, coord_dim, 1) * (K,) -> (B, N, coord_dim, K)
        angles = coords.unsqueeze(-1) * self.coord_fourier_freqs
        feats = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        # (B, N, coord_dim, 2K) -> (B, N, coord_dim * 2K)
        return feats.flatten(start_dim=-2).to(coords.dtype)

    def forward(
        self,
        x: torch.Tensor,               # (B, N, coord_dim + lattice_dim)  coords and lattice
        atom_types: torch.Tensor,      # (B, N)  atom type indices
        t: torch.Tensor,               # (B,)          current time ∈ [0, 1]
        r: torch.Tensor,               # (B,)          target time ∈ [0, t]
        spacegroup: torch.Tensor,      # (B,)          0 = null (unconditional)
        mask: torch.Tensor,            # (B, N)        True for valid tokens
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Returns u(x, t, r): mean velocity field, shape (B, N, coord_dim + lattice_dim).
        If predict_atom_types is enabled, also returns token-wise atom logits with
        shape (B, N, atom_type_vocab_size).
        Padding tokens are zeroed out in the continuous branch via mask.
        """
        # --- Separate processing of components ---
        coords = x[..., :self.coord_dim]  # First coord_dim are coordinates
        lattice = x[..., self.coord_dim:self.coord_dim+self.lattice_dim]  # Next lattice_dim are lattice

        # Embed atom types. For joint A/X/L generation we can optionally drop
        # the atom-type conditioning during training so the model cannot simply
        # copy atom identities from the input context.
        atom_types = atom_types.clamp(min=0, max=self.atom_type_vocab_size - 1)
        atom_types_cond = atom_types
        if self.predict_atom_types and self.atom_type_condition_dropout > 0 and self.training:
            drop_mask = torch.rand(atom_types.shape, device=atom_types.device) < self.atom_type_condition_dropout
            atom_types_cond = torch.where(drop_mask, torch.zeros_like(atom_types), atom_types)

        atom_emb = self.atom_type_embedder(atom_types_cond)  # (B, N, atom_type_embed_dim)

        # Concatenate atom-type conditioning with the flowed continuous state.
        # Coordinates go through the (optional) periodic featurisation first;
        # with coord_fourier_bands=0 this is an identity pass-through.
        coord_feats = self._featurize_coords(coords)
        x_combined = torch.cat([atom_emb, coord_feats, lattice], dim=-1)  # (B, N, model_input_dim)

        # --- Input embedding ---
        x_emb = self.x_embedder(x_combined)  # (B, N, hidden_size)

        # Sinusoidal positional embedding based on token indices within each sequence
        token_index = torch.cumsum(mask, dim=-1, dtype=torch.int64) - 1
        pos_emb = get_pos_embedding(token_index, self.hidden_size, max_len=2048)
        x_emb = x_emb + pos_emb

        # --- Conditioning vector c (broadcast over tokens inside each block) ---
        t_emb = self.t_embedder(t)  # (B, hidden_size)
        r_emb = self.r_embedder(r)  # (B, hidden_size)
        if self.use_spacegroup_conditioning:
            s_emb = self.spacegroup_embedder(spacegroup, self.training)
        else:
            # Hard-disable SG pathway so labels are ignored even if provided.
            s_emb = torch.zeros_like(t_emb)

        c = t_emb + r_emb + s_emb

        # --- Transformer blocks ---
        for block in self.blocks:
            x_emb = block(x_emb, c)

        # --- Output projection ---
        x_out = self.final_layer(x_emb, c)

        if self.predict_atom_types:
            continuous_out = x_out[..., :self.in_channels]
            atom_type_logits = x_out[..., self.in_channels:]
            continuous_out = continuous_out * mask[..., None]
            return continuous_out, atom_type_logits

        x_out = x_out * mask[..., None]     # zero out padding positions
        return x_out