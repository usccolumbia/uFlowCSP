"""Dataset for Alexandria structures in raw atomic format for mean flow training.

Reads a CSV with ``material_id`` + ``cif`` columns (same format as MP20's all.csv),
processes the CIFs using the shared preprocessing pipeline, and stores the result
in a **separate** cache directory (``<csv_dir>/alex_raw/``) that never overlaps with
the existing ``mp20_raw/`` cache.
"""

import os
import tempfile
import warnings
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset

from src.data.components.atom_ordering_transform import apply_canonical_ordering
from src.data.components.preprocessing_utils import preprocess

warnings.simplefilter("ignore", UserWarning)


class AlexRawData(InMemoryDataset):
    """Alexandria crystal structures in raw atomic format for direct mean flow training.

    Loads structures from a CSV file (``material_id`` + ``cif`` columns) and converts
    them to the same raw ``Data`` object format as ``MP20RawData``:
    ``atom_types``, ``pos``, ``frac_coords``, ``cell``, ``lattices``,
    ``lattices_scaled``, ``lengths``, ``lengths_scaled``, ``angles``,
    ``angles_radians``, ``lattice`` (per-atom (N, 9)), ``spacegroup``,
    ``num_atoms``, ``num_nodes``, ``token_idx``.

    Cache is written to ``<csv_dir>/alex_raw/processed/alex_raw.pt``.
    The existing ``mp20_raw/`` cache is **never** touched.

    Args:
        alex_csv: Absolute or relative path to the Alexandria CSV file.
        transform: Optional transform applied on every access.
        pre_transform: Optional pre-transform applied before saving.
        pre_filter: Optional pre-filter applied before saving.
        force_reload: If ``True``, ignore any existing cache and reprocess.
    """

    def __init__(
        self,
        alex_csv: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
        max_samples: int = 40_000,
        spacegroup_col: str = "spacegroup",
        seed: int = 42,
        compute_ordering: bool = False,
        ordering_mode: str = "symmetry",
        ordering_symprec: float = 0.1,
    ):
        self.alex_csv = os.path.abspath(alex_csv)
        self.max_samples = max_samples
        self.spacegroup_col = spacegroup_col
        self.seed = seed
        # Optional MCFlow-style canonical atom ordering (default OFF -> identical
        # behaviour and a separate cache file from the legacy pipeline).
        self.compute_ordering = bool(compute_ordering)
        self.ordering_mode = str(ordering_mode)
        self.ordering_symprec = float(ordering_symprec)
        # Cache lives in alex_raw/ next to the CSV — isolated from mp20_raw/
        root = os.path.join(os.path.dirname(self.alex_csv), "alex_raw")
        os.makedirs(root, exist_ok=True)

        super().__init__(root, transform, pre_transform, pre_filter, force_reload=force_reload)
        self.load(self.processed_paths[0])

        # Guard against stale caches that predate the spacegroup/cell fields
        if len(self) > 0:
            sample = self.get(0)
            required_fields = ("spacegroup", "cell", "frac_coords", "dataset_idx", "is_valid")
            if self.compute_ordering:
                required_fields = required_fields + ("orbit_id", "wyckoff_rank", "en_value")
            if any(not hasattr(sample, field) for field in required_fields):
                print("Detected stale alex_raw cache missing required fields; rebuilding...")
                self.process()
                self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return []

    @property
    def processed_file_names(self) -> List[str]:
        # Ordered variant uses a separate cache so the legacy cache is untouched.
        if getattr(self, "compute_ordering", False):
            return [f"alex_raw_ordered_{self.ordering_mode}.pt"]
        return ["alex_raw.pt"]

    def download(self):
        pass

    def process(self):
        """Process Alexandria CSV into raw Data objects identical in format to MP20RawData."""
        print(f"Processing Alexandria CSV: {self.alex_csv}")

        # Filter out any MP20 rows (material_id starting with 'mp-') that may be
        # present in the combined CSV.  MP20 data is loaded separately via
        # MP20RawData; including it here would create duplicates.
        # Use engine='python': CIF strings contain embedded quotes/newlines that
        # confuse the faster C parser, causing EOF tokenization errors.
        df_full = pd.read_csv(self.alex_csv, engine='python')
        n_before = len(df_full)
        df_alex = df_full[df_full["material_id"].astype(str).str.startswith("alex")].reset_index(drop=True)
        n_dropped = n_before - len(df_alex)
        if n_dropped:
            print(f"  Skipped {n_dropped} non-Alexandria rows (material_id not starting with 'alex') "
                  f"from CSV — only rows with 'alex' prefix are processed here.")
        print(f"  Alexandria-only rows to process: {len(df_alex)}")

        # Stratified sampling: pick up to max_samples rows distributed evenly
        # across all spacegroups present in the CSV, so no spacegroup is
        # over- or under-represented relative to its natural frequency.
        if self.max_samples is not None and len(df_alex) > self.max_samples:
            rng = np.random.default_rng(self.seed)
            if self.spacegroup_col in df_alex.columns:
                # Proportional stratified sample: each SG contributes
                # floor(max_samples * count_sg / total) rows, with remainders
                # filled by a random draw across all SGs.
                sg_counts = df_alex[self.spacegroup_col].value_counts()
                n_total = len(df_alex)
                alloc = (sg_counts * self.max_samples / n_total).astype(int)
                remainder = self.max_samples - alloc.sum()

                sampled_indices = []
                for sg, n_alloc in alloc.items():
                    sg_idx = df_alex.index[df_alex[self.spacegroup_col] == sg].tolist()
                    chosen = rng.choice(sg_idx, size=n_alloc, replace=False).tolist()
                    sampled_indices.extend(chosen)

                # Fill remainder: one extra sample from random SGs (weighted by leftover)
                if remainder > 0:
                    leftover_idx = list(set(df_alex.index.tolist()) - set(sampled_indices))
                    extra = rng.choice(leftover_idx, size=remainder, replace=False).tolist()
                    sampled_indices.extend(extra)

                df_alex = df_alex.loc[sampled_indices].reset_index(drop=True)
                print(f"  Stratified sample: {len(df_alex)} rows from "
                      f"{sg_counts.shape[0]} unique spacegroups "
                      f"(seed={self.seed}, max_samples={self.max_samples})")
            else:
                # No spacegroup column — fall back to plain random sample
                print(f"  Warning: spacegroup column '{self.spacegroup_col}' not found; "
                      f"using plain random sampling.")
                df_alex = df_alex.sample(n=self.max_samples, random_state=self.seed).reset_index(drop=True)
                print(f"  Random sample: {len(df_alex)} rows")

        # Write the filtered rows to a temporary CSV so preprocess() can read it
        with tempfile.NamedTemporaryFile(
            mode="w", suffix="_alex_filtered.csv", delete=False
        ) as tmp:
            tmp_path = tmp.name
            df_alex.to_csv(tmp_path, index=False)

        try:
            cached_data = preprocess(
                tmp_path,
                num_workers=32,
                niggli=True,
                primitive=False,
                graph_method="crystalnn",
                prop_list=[],          # No property labels required from Alexandria
                use_space_group=True,
                tol=0.1,
            )
        finally:
            os.remove(tmp_path)

        data_list = []
        for idx, data_dict in enumerate(cached_data):
            graph_arrays = data_dict["graph_arrays"]
            atom_types = graph_arrays["atom_types"]      # numpy (N,)
            frac_coords = graph_arrays["frac_coords"]    # numpy (N, 3)
            cell = graph_arrays["cell"]                  # numpy (3, 3) lattice matrix
            lattices = graph_arrays["lattices"]          # numpy (6,) a,b,c,alpha,beta,gamma
            lengths = graph_arrays["lengths"]            # numpy (3,)
            angles = graph_arrays["angles"]              # numpy (3,)
            num_atoms = int(graph_arrays["num_atoms"])

            # Normalised lengths (scale-invariant) and angles in radians
            _lengths = lengths / float(num_atoms) ** (1.0 / 3.0)
            _angles = np.radians(angles)

            cell_t = torch.Tensor(cell).unsqueeze(0)          # (1, 3, 3)
            num_atoms_t = torch.LongTensor([num_atoms])
            frac_coords_t = torch.Tensor(frac_coords)         # (N, 3)

            # Cartesian positions: frac @ cell  (same einsum as MP20)
            pos = torch.einsum(
                "bi,bij->bj",
                frac_coords_t,
                torch.repeat_interleave(cell_t, num_atoms_t, dim=0),
            )

            # Per-atom flattened lattice matrix (N, 9) — used by mean flow model
            lattice_flat = cell_t.flatten()                          # (9,)
            lattice_repeated = lattice_flat.unsqueeze(0).repeat(num_atoms, 1)  # (N, 9)

            raw_data = Data(
                id=data_dict["mp_id"],
                structure_id=data_dict["mp_id"],
                structure_idx=torch.tensor([idx], dtype=torch.long),
                dataset_idx=torch.tensor([0], dtype=torch.long),
                is_valid=torch.tensor([1], dtype=torch.long),
                atom_types=torch.LongTensor(atom_types),
                pos=pos,                                              # (N, 3)
                frac_coords=frac_coords_t,                            # (N, 3)
                cell=cell_t,                                          # (1, 3, 3)
                lattices=torch.Tensor(lattices).unsqueeze(0),         # (1, 6)
                lattices_scaled=torch.Tensor(                         # (1, 6)
                    np.concatenate([_lengths, _angles])
                ).unsqueeze(0),
                lengths=torch.Tensor(lengths).view(1, -1),            # (1, 3)
                lengths_scaled=torch.Tensor(_lengths).view(1, -1),    # (1, 3)
                angles=torch.Tensor(angles).view(1, -1),              # (1, 3)
                angles_radians=torch.Tensor(_angles).view(1, -1),     # (1, 3)
                lattice=lattice_repeated,                             # (N, 9)
                spacegroup=torch.LongTensor([data_dict["spacegroup"]]),
                num_atoms=torch.LongTensor([num_atoms]),
                num_nodes=torch.LongTensor([num_atoms]),
                token_idx=torch.arange(num_atoms),
            )

            if self.compute_ordering:
                raw_data = apply_canonical_ordering(
                    raw_data,
                    mode=self.ordering_mode,
                    symprec=self.ordering_symprec,
                )

            data_list.append(raw_data)

            if (idx + 1) % 1000 == 0:
                print(f"  Processed {idx + 1}/{len(cached_data)} Alexandria structures")

        print(f"Total Alexandria structures: {len(data_list)}")

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        self.save(data_list, self.processed_paths[0])
        print(f"Saved Alexandria cache to {self.processed_paths[0]}")
