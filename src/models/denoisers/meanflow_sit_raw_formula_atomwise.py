"""
MeanFlowSiTRawFormulaAtomwise: MeanFlowSiTRawFormula + explicit per-token chemistry features.

This is a NEW, ADDITIVE copy of meanflow_sit_raw_formula.py. It does not modify
MeanFlowSiTRawFormula, MeanFlowSiTRaw, or any of their behavior; existing
configs/checkpoints referencing those classes are completely unaffected.

Motivation
----------
MeanFlowSiTRawFormula gives the model an explicit *global* composition signal
(e_formula) but each atom's own token is still only a learned embedding keyed
by its atomic number Z -- no explicit chemistry is attached to the token
itself. This class additionally concatenates a small set of basic, fixed
periodic-table properties (Pauling electronegativity, atomic radius, period,
group; see atomic_features.py) onto each atom's token, alongside its learned
Z-embedding, before the input projection:

    token_input = [Z-embedding | atomic chemistry features | coords | lattice]

So every token carries a bit of real chemistry about itself, not just an
opaque learned ID plus a global composition summary.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Re-use the shared building blocks from EqM's SiT to stay consistent
from src.models.denoisers.atomic_features import ATOMIC_FEATURE_DIM, build_atomic_feature_table
from src.models.denoisers.dit import (
    FinalLayer,
    LabelEmbedder,
    SiTBlock,
    TimestepEmbedder,
    get_pos_embedding,
)


class FormulaEmbedder(nn.Module):
    """
    Embeds a normalized element-count composition vector (B, atom_type_vocab_size)
    into the transformer's hidden size, for use as an additive conditioning term.
    """

    def __init__(self, atom_type_vocab_size: int, hidden_size: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(atom_type_vocab_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, composition: torch.Tensor) -> torch.Tensor:
        return self.mlp(composition)


class MeanFlowSiTRawFormulaAtomwise(nn.Module):
    """
    Sequence Transformer for MeanFlow on raw atomic data, with both:
      - an optional formula-global composition embedding added to `c`
        (same as MeanFlowSiTRawFormula), and
      - optional explicit per-token chemistry features (electronegativity,
        atomic radius, period, group) concatenated onto each atom's token.

    Identical to MeanFlowSiTRawFormula except for the additional
    `use_atomwise_features` option (default True in this class).

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
        use_spacegroup_conditioning: Whether the SG pathway is active at all.
        use_formula_embedding: Whether to compute and add e_formula to `c`.
                            Default True.
        use_atomwise_features: Whether to concatenate fixed per-token chemistry
                            features onto each atom's token. Default True; set
                            False to reproduce MeanFlowSiTRawFormula exactly
                            while still using this class.
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
        use_formula_embedding: bool = True,
        use_atomwise_features: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.atom_type_vocab_size = atom_type_vocab_size
        self.atom_type_embed_dim = atom_type_embed_dim
        self.coord_dim = coord_dim
        self.lattice_dim = lattice_dim
        self.in_channels = coord_dim + lattice_dim
        self.use_atomwise_features = bool(use_atomwise_features)
        atomwise_dim = ATOMIC_FEATURE_DIM if self.use_atomwise_features else 0
        self.model_input_dim = atom_type_embed_dim + atomwise_dim + self.in_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.predict_atom_types = predict_atom_types
        self.atom_type_condition_dropout = float(atom_type_condition_dropout)
        self.use_spacegroup_conditioning = bool(use_spacegroup_conditioning)
        self.use_formula_embedding = bool(use_formula_embedding)

        # Separate embedders for components
        self.atom_type_embedder = nn.Embedding(atom_type_vocab_size, atom_type_embed_dim)
        # Coordinates are continuous, so no embedder needed - pass through
        # Lattice is continuous, pass through

        # NEW: fixed (non-trainable) per-element chemistry feature table,
        # indexed by atomic number Z (same indexing as atom_type_embedder).
        # Concatenated onto each atom's token alongside its learned embedding.
        if self.use_atomwise_features:
            self.register_buffer(
                "atomic_feature_table",
                build_atomic_feature_table(atom_type_vocab_size),
                persistent=False,
            )

        # Combined input projection: [atom embedding | (atomic features) | flowed state] -> hidden_size
        self.x_embedder = nn.Linear(self.model_input_dim, hidden_size, bias=True)

        # Two separate timestep embedders: current time t and target time r
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.r_embedder = TimestepEmbedder(hidden_size)

        # Conditioning embedders (same NUM_CLASSES logic as EqM: index 0 = null class)
        self.spacegroup_embedder = LabelEmbedder(num_spacegroups, hidden_size, class_dropout_prob)

        # Formula-global composition embedder. Consumes a normalized
        # element-count vector (fraction of each atom type in the structure,
        # over valid/non-padding tokens) derived from the ground-truth atom
        # types + mask -- i.e. purely from the formula, not from SG labels.
        if self.use_formula_embedding:
            self.formula_embedder = FormulaEmbedder(atom_type_vocab_size, hidden_size)

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

    def _composition_vector(self, atom_types: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Build the normalized element-count composition vector from the
        ground-truth atom types and the valid-token mask.

        atom_types: (B, N) integer element indices (already clamped to vocab range)
        mask:       (B, N) True/1 for valid (non-padding) tokens

        Returns: (B, atom_type_vocab_size) with each row summing to 1 over
        valid tokens (formula fractions), zeros for structures with no valid
        tokens (degenerate/padding-only, should not occur in practice).
        """
        valid = mask.to(dtype=torch.float32)
        one_hot = F.one_hot(atom_types, num_classes=self.atom_type_vocab_size).to(dtype=valid.dtype)
        counts = (one_hot * valid.unsqueeze(-1)).sum(dim=1)  # (B, atom_type_vocab_size)
        totals = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
        return counts / totals

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

        # Concatenate atom-type conditioning (+ optional per-token chemistry
        # features, using the same possibly-dropped-out atom_types_cond so the
        # atom_type_condition_dropout mechanism isn't leaked through) with the
        # flowed continuous state.
        if self.use_atomwise_features:
            atom_features = self.atomic_feature_table[atom_types_cond]  # (B, N, ATOMIC_FEATURE_DIM)
            x_combined = torch.cat([atom_emb, atom_features, coords, lattice], dim=-1)
        else:
            x_combined = torch.cat([atom_emb, coords, lattice], dim=-1)  # (B, N, model_input_dim)

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

        if self.use_formula_embedding:
            # NOTE: uses the *ground-truth* atom_types (not atom_types_cond),
            # since the formula (composition) is the given/known quantity in
            # CSP -- it should stay available even when per-token atom-type
            # conditioning is being dropped out for joint generation.
            composition = self._composition_vector(atom_types, mask)  # (B, atom_type_vocab_size)
            f_emb = self.formula_embedder(composition)  # (B, hidden_size)
            c = c + f_emb

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
