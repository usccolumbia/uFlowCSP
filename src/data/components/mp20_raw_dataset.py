"""Dataset for raw atomic data (atom types, coordinates, lattice) for mean flow training."""

import warnings
from typing import Callable, List, Optional

import torch
from torch_geometric.data import Data, InMemoryDataset

from src.data.components.atom_ordering_transform import apply_canonical_ordering
from src.data.components.mp20_dataset import MP20

warnings.simplefilter("ignore", UserWarning)


class MP20RawData(InMemoryDataset):
    """MP20 dataset with raw atomic data for direct mean flow training.

    Loads MP20 structures and converts to raw data format without VAE encoding.
    All structures are marked as valid for CFG training.

    Args:
        mp20_root: Root directory of MP20 dataset (contains processed/mp20.pt)
        transform: Optional transform to be applied to data
        pre_transform: Optional pre-transform
        pre_filter: Optional pre-filter
        force_reload: Whether to reprocess the dataset
    """

    def __init__(
        self,
        mp20_root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
        compute_ordering: bool = False,
        ordering_mode: str = "symmetry",
        ordering_symprec: float = 0.1,
        dataset_name: str = "mp20",
    ):
        self.mp20_root = mp20_root
        # Dataset name drives the processed-cache dir/filename so multiple
        # benchmarks (mp20, perov, mpts, ...) never clash. Default "mp20" keeps
        # the existing cache path byte-for-byte.
        self.dataset_name = str(dataset_name)
        # Optional MCFlow-style canonical atom ordering (default OFF -> identical
        # behaviour and a separate cache file from the legacy pipeline).
        self.compute_ordering = bool(compute_ordering)
        self.ordering_mode = str(ordering_mode)
        self.ordering_symprec = float(ordering_symprec)

        # Create root directory for this raw dataset (per-dataset sibling of the
        # source root, e.g. .../mp20_raw, .../perov_raw, .../mpts_raw).
        import os
        root = os.path.join(os.path.dirname(mp20_root), f'{self.dataset_name}_raw')
        os.makedirs(root, exist_ok=True)

        super().__init__(root, transform, pre_transform, pre_filter, force_reload=force_reload)
        self.load(self.processed_paths[0])

        if len(self) > 0:
            sample = self.get(0)
            required_fields = ("spacegroup", "cell", "frac_coords", "dataset_idx", "is_valid")
            if self.compute_ordering:
                required_fields = required_fields + ("orbit_id", "wyckoff_rank", "en_value")
            if any(not hasattr(sample, field) for field in required_fields):
                print("Detected stale mp20_raw cache missing required conditioning fields; rebuilding...")
                self.process()
                self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return []

    @property
    def processed_file_names(self) -> List[str]:
        # Ordered variant uses a separate cache so the legacy cache is untouched.
        name = getattr(self, "dataset_name", "mp20")
        if getattr(self, "compute_ordering", False):
            return [f'{name}_raw_ordered_{self.ordering_mode}.pt']
        return [f'{name}_raw.pt']

    def download(self):
        pass

    def process(self):
        """Process MP20 data into raw format."""
        data_list = []

        # Load MP20 structures
        print("Loading MP20 structures into raw data format...")
        mp20_dataset = MP20(root=self.mp20_root)

        for idx in range(len(mp20_dataset)):
            data = mp20_dataset.get(idx)

            data_id = getattr(data, "id", None)
            structure_id = getattr(data, "structure_id", data_id)
            structure_idx = getattr(
                data,
                "structure_idx",
                torch.tensor([idx], dtype=torch.long),
            )
            if not torch.is_tensor(structure_idx):
                structure_idx = torch.tensor([structure_idx], dtype=torch.long)

            # Convert to raw format
            # data already has: atom_types, pos (cart coords), cell (1, 3, 3)
            # Flatten lattice for per-atom repetition
            lattice_flat = data.cell.flatten()  # (9,) from (1,3,3)

            # For mean flow, we need per-atom features
            # Repeat lattice for each atom
            num_atoms = data.num_nodes
            lattice_repeated = lattice_flat.unsqueeze(0).repeat(num_atoms, 1)  # (N, 9)

            # Create new data object with raw features
            raw_data = Data(
                id=data_id,
                structure_id=structure_id,
                structure_idx=structure_idx,
                dataset_idx=torch.tensor([0], dtype=torch.long),
                is_valid=torch.tensor([1], dtype=torch.long),
                atom_types=data.atom_types,  # (N,) long
                pos=data.pos,               # (N, 3) float
                frac_coords=data.frac_coords,
                cell=data.cell,
                lattices=data.lattices,
                lattices_scaled=data.lattices_scaled,
                lengths=data.lengths,
                lengths_scaled=data.lengths_scaled,
                angles=data.angles,
                angles_radians=data.angles_radians,
                lattice=lattice_repeated,   # (N, 9) float
                spacegroup=data.spacegroup,
                num_atoms=data.num_atoms,
                num_nodes=data.num_nodes,
                token_idx=data.token_idx,
            )

            if self.compute_ordering:
                raw_data = apply_canonical_ordering(
                    raw_data,
                    mode=self.ordering_mode,
                    symprec=self.ordering_symprec,
                )

            data_list.append(raw_data)

            if (idx + 1) % 1000 == 0:
                print(f"Processed {idx + 1}/{len(mp20_dataset)} MP20 structures")

        print(f"Total MP20 structures: {len(data_list)} (all valid)")

        # Apply pre-filter and pre-transform
        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        # Save processed data using new PyG API (matches self.load())
        self.save(data_list, self.processed_paths[0])
        print(f"\nSaved processed raw data to {self.processed_paths[0]}")


if __name__ == "__main__":
    # Test dataset loading
    dataset = MP20RawData(
        mp20_root="data/mp_20",
        force_reload=True,
    )
    print(f"\nDataset created successfully with {len(dataset)} structures")
    sample = dataset[0]
    print(f"Sample data: atom_types shape {sample.atom_types.shape}, pos shape {sample.pos.shape}, lattice shape {sample.lattice.shape}")