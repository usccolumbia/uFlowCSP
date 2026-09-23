"""Lightning module for MeanFlow CSP (CrystalFlow-style geometry).

Subclasses :class:`src.models.meanflow_raw_module.MeanFlowRawLitModule` and
overrides only the geometry-specific glue:

* :meth:`encode_batch` packs **fractional coordinates** (not Cartesian) and the
  **6-D polar lattice representation** into the dense ``x`` tensor.
* :meth:`_sample_initial_noise` draws CrystalFlow's prior — uniform fractional
  coordinates plus an identity-centred Gaussian over the polar lattice.
* :meth:`_generated_batch_to_array_dicts` rebuilds the 3x3 lattice matrix from
  the generated polar representation before computing lengths/angles.

Everything else (EMA, evaluators, training/validation/test loops, formula
conditioning) is inherited unchanged.
"""

from __future__ import annotations

import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch

from src.models.components.lattice_polar import (
    lattice_polar_build_torch,
    lattice_polar_decompose_torch,
)
from src.models.meanflow_raw_module import MeanFlowRawLitModule


class MeanFlowCSPLitModule(MeanFlowRawLitModule):
    """MeanFlow CSP module using fractional coords + 6-D polar lattice."""

    @property
    def _lattice_rep_dim(self) -> int:
        return int(getattr(self.denoiser, "lattice_rep_dim", 6))

    @property
    def _coord_dim(self) -> int:
        return int(getattr(self.denoiser, "coord_dim", 3))

    # ------------------------------------------------------------------
    # encoding
    # ------------------------------------------------------------------

    def encode_batch(self, batch: Data):
        """Pack fractional coords + broadcast polar lattice into dense ``x``."""
        max_nodes = int(self.denoiser.input_size)

        frac = batch.frac_coords % 1.0
        frac_dense, mask = to_dense_batch(frac, batch.batch, max_num_nodes=max_nodes)
        atom_types, _ = to_dense_batch(batch.atom_types, batch.batch, max_num_nodes=max_nodes)

        cell = batch.cell.view(-1, 3, 3).to(frac_dense.dtype)        # (B, 3, 3)
        k = lattice_polar_decompose_torch(cell).to(frac_dense.dtype)  # (B, 6)
        k_tokens = k.unsqueeze(1).expand(-1, frac_dense.shape[1], -1)  # (B, N, 6)

        x = torch.cat([frac_dense, k_tokens], dim=-1)                # (B, N, 9)
        return x, atom_types, mask

    # ------------------------------------------------------------------
    # sampling prior
    # ------------------------------------------------------------------

    def _sample_initial_noise(self, cur_batch, num_nodes, in_channels, mask, device):
        """CrystalFlow prior: U[0,1) fractional coords + identity-centred polar k."""
        coord_dim = self._coord_dim
        lattice_rep_dim = self._lattice_rep_dim
        sigma = float(getattr(self.meanflow, "lattice_polar_sigma", 0.1))

        frac = torch.rand(cur_batch, num_nodes, coord_dim, device=device)
        k0 = torch.randn(cur_batch, lattice_rep_dim, device=device) * sigma
        k0[:, -1] += 1.0  # centre on the identity lattice (log-space origin)
        k_tokens = k0.unsqueeze(1).expand(-1, num_nodes, -1)
        return torch.cat([frac, k_tokens], dim=-1)

    # ------------------------------------------------------------------
    # decoding
    # ------------------------------------------------------------------

    def _generated_batch_to_array_dicts(
        self,
        x_gen: torch.Tensor,
        atom_types: torch.Tensor,
        mask: torch.Tensor,
        sample_offset: int = 0,
    ) -> list[dict]:
        """Rebuild structures: polar k -> 3x3 lattice, frac coords -> cart."""
        coord_dim = self._coord_dim
        lattice_rep_dim = self._lattice_rep_dim
        eps = 1e-8
        batch_size = x_gen.shape[0]
        samples = []

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

            frac_coords_i = x_gen[i, active_mask, :coord_dim] % 1.0
            k_tokens_i = x_gen[i, active_mask, coord_dim:coord_dim + lattice_rep_dim]
            k_i = k_tokens_i.mean(dim=0)                              # (6,)
            cell_i = lattice_polar_build_torch(k_i.unsqueeze(0).float())[0]  # (3, 3)

            pos_i = frac_coords_i.float() @ cell_i

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
