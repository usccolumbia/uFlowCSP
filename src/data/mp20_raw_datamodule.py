"""DataModule for raw atomic data (atom types, coordinates, lattice) for mean flow training."""

import csv
import json
from pathlib import Path
from typing import List, Optional, Set

from lightning import LightningDataModule
import torch
from torch.utils.data import ConcatDataset, Subset, random_split
from torch_geometric.loader import DataLoader as PyGDataLoader

from src.data.components.alex_raw_dataset import AlexRawData
from src.data.components.atom_ordering_transform import AtomOrderingTransform
from src.data.components.mp20_raw_dataset import MP20RawData


class MP20RawDataModule(LightningDataModule):
    """DataModule for raw atomic data mean flow training.

    Uses MP20 structures in raw format (atom types, coords, lattice) without VAE encoding.
    All structures marked as valid for CFG training.

    Args:
        mp20_root: Root directory of MP20 dataset
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory
        force_reload: Whether to reprocess the dataset
        train_split: Fraction of data for training (default 0.8, ignored if split_type != 'random')
        val_split: Fraction of data for validation (default 0.1, ignored if split_type != 'random')
        exclude_ids_csv: Optional CSV path containing material_id/mp_id to exclude
        exclude_ids: Optional explicit list of material IDs to exclude
        split_type: Which split to use. 'random' does a size-based random split.
            Any other value NAME is treated as a named split and loads
            data/splits/<name>_{train,val,test}_ids.json (e.g. 'difCSP', 'perov',
            'mpts'). Default 'difCSP'.
        dataset: Dataset name ('mp20', 'perov', 'mpts', ...). Drives the raw
            processed-cache path so benchmarks never clash. Default 'mp20'.
        data_root: Root dir of the dataset's raw data (contains raw/all.csv).
            Defaults to ``mp20_root`` when None (back-compat for MP-20).
    """

    def __init__(
        self,
        mp20_root: str,
        batch_size: int = 256,
        num_workers: int = 16,
        pin_memory: bool = False,
        force_reload: bool = False,
        train_split: float = 0.8,
        val_split: float = 0.1,
        exclude_ids_csv: Optional[str] = None,
        exclude_ids: Optional[List[str]] = None,
        split_type: str = "difCSP",
        dataset: str = "mp20",
        data_root: Optional[str] = None,
        use_alexandria: bool = False,
        alex_csv: Optional[str] = None,
        alex_max_samples: int = 40_000,
        val_size: Optional[int] = None,
        test_size: Optional[int] = None,
        atom_ordering: str = "none",
        perm_augment: bool = False,
        modulo_translation: bool = False,
        ordering_symprec: float = 0.1,
        ordering_seed: Optional[int] = None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[MP20RawData] = None
        self.data_val: Optional[MP20RawData] = None
        self.data_test: Optional[MP20RawData] = None

    @staticmethod
    def _normalize_id(value: object) -> Optional[str]:
        if value is None:
            return None
        if torch.is_tensor(value):
            if value.numel() != 1:
                return None
            value = value.item()
        normalized = str(value).strip()
        return normalized or None

    def _load_exclude_ids(self) -> Set[str]:
        exclude_set: Set[str] = set()

        for item in self.hparams.exclude_ids or []:
            normalized = self._normalize_id(item)
            if normalized:
                exclude_set.add(normalized)

        csv_path = self.hparams.exclude_ids_csv
        if csv_path:
            csv_file = Path(csv_path)
            if not csv_file.exists():
                raise FileNotFoundError(f"exclude_ids_csv not found: {csv_path}")

            with csv_file.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fieldnames = reader.fieldnames or []
                if "material_id" in fieldnames:
                    id_key = "material_id"
                elif "mp_id" in fieldnames:
                    id_key = "mp_id"
                else:
                    raise ValueError(
                        f"CSV must contain material_id or mp_id column: {csv_path}"
                    )

                for row in reader:
                    normalized = self._normalize_id(row.get(id_key))
                    if normalized:
                        exclude_set.add(normalized)

        return exclude_set

    def setup(self, stage: Optional[str] = None):
        """Load data and create train/val/test splits."""

        if not self.data_train and not self.data_val and not self.data_test:
            # ── 0. Resolve optional MCFlow-style atom ordering ──────────────
            # Defaults keep the legacy pipeline byte-for-byte: atom_ordering
            # 'none' -> no ordering, legacy cache, no augmentation transform.
            ordering = str(self.hparams.atom_ordering or "none").lower()
            compute_ordering = ordering in ("simple", "symmetry")
            ordering_mode = ordering if compute_ordering else "symmetry"
            ordering_transform = None
            if compute_ordering and (
                self.hparams.perm_augment or self.hparams.modulo_translation
            ):
                ordering_transform = AtomOrderingTransform(
                    perm_augment=self.hparams.perm_augment,
                    modulo_translation=self.hparams.modulo_translation,
                    seed=self.hparams.ordering_seed,
                )
            if compute_ordering:
                print(
                    f"Atom ordering enabled: mode={ordering_mode}, "
                    f"perm_augment={self.hparams.perm_augment}, "
                    f"modulo_translation={self.hparams.modulo_translation}"
                )

            # ── 1. Load raw dataset (mp20/perov/mpts/...) ───────────────────
            data_root = self.hparams.data_root or self.hparams.mp20_root
            print(f"Loading raw '{self.hparams.dataset}' dataset from {data_root} "
                  f"for mean flow training...")
            mp20_dataset = MP20RawData(
                mp20_root=data_root,
                dataset_name=self.hparams.dataset,
                force_reload=self.hparams.force_reload,
                transform=ordering_transform,
                compute_ordering=compute_ordering,
                ordering_mode=ordering_mode,
                ordering_symprec=self.hparams.ordering_symprec,
            )

            # Exclude requested material IDs before splitting.
            exclude_ids = self._load_exclude_ids()
            if exclude_ids:
                keep_indices = []
                dropped = 0
                for idx in range(len(mp20_dataset)):
                    sample = mp20_dataset.get(idx)
                    sample_id = getattr(sample, "structure_id", getattr(sample, "id", None))
                    sample_id = self._normalize_id(sample_id)
                    if sample_id in exclude_ids:
                        dropped += 1
                    else:
                        keep_indices.append(idx)
                mp20_dataset = Subset(mp20_dataset, keep_indices)
                print(
                    f"Excluded {dropped} MP20 samples using {len(exclude_ids)} requested IDs."
                )
                print(f"MP20 samples after exclusion: {len(mp20_dataset)}")

            # ── 2. Optionally load Alexandria dataset ───────────────────────
            if self.hparams.use_alexandria and self.hparams.alex_csv:
                print(f"Loading Alexandria dataset from {self.hparams.alex_csv}...")
                alex_dataset = AlexRawData(
                    alex_csv=self.hparams.alex_csv,
                    force_reload=self.hparams.force_reload,
                    max_samples=self.hparams.alex_max_samples,
                    transform=ordering_transform,
                    compute_ordering=compute_ordering,
                    ordering_mode=ordering_mode,
                    ordering_symprec=self.hparams.ordering_symprec,
                )
                if exclude_ids:
                    keep_indices = []
                    dropped = 0
                    for idx in range(len(alex_dataset)):
                        sample = alex_dataset.get(idx)
                        sample_id = getattr(sample, "structure_id", getattr(sample, "id", None))
                        sample_id = self._normalize_id(sample_id)
                        if sample_id in exclude_ids:
                            dropped += 1
                        else:
                            keep_indices.append(idx)
                    if dropped:
                        alex_dataset = Subset(alex_dataset, keep_indices)
                        print(f"Excluded {dropped} Alexandria samples. Remaining: {len(alex_dataset)}")
                print(f"Alexandria dataset: {len(alex_dataset)} structures")
                dataset = ConcatDataset([mp20_dataset, alex_dataset])
                print(
                    f"Combined dataset: {len(dataset)} structures "
                    f"(MP20: {len(mp20_dataset)}, Alexandria: {len(alex_dataset)})"
                )
            else:
                dataset = mp20_dataset

            # ── 3. Split ────────────────────────────────────────────────────
            if self.hparams.split_type != "random":
                # Named fixed split: load data/splits/<name>_{train,val,test}_ids.json.
                # The three id lists are disjoint by construction, so a sample can
                # only ever land in one subset (no train/test leakage). Works for
                # 'difCSP' (mp20) and any benchmark ('perov', 'mpts', ...).
                split_name = self.hparams.split_type
                split_dir = Path("data/splits")

                def _load_split_ids(kind: str) -> Set[str]:
                    path = split_dir / f"{split_name}_{kind}_ids.json"
                    if not path.exists():
                        raise FileNotFoundError(
                            f"Split file not found: {path}. Expected named-split id "
                            f"lists for split_type='{split_name}'. Run "
                            f"prepare_benchmark_data.py for this dataset first."
                        )
                    with open(path, "r") as f:
                        return set(str(x).strip() for x in json.load(f))

                train_ids = _load_split_ids("train")
                val_ids = _load_split_ids("val")
                test_ids = _load_split_ids("test")

                train_indices = []
                val_indices = []
                test_indices = []
                for idx in range(len(dataset)):
                    sample = dataset[idx]
                    sample_id = getattr(sample, "structure_id", getattr(sample, "id", None))
                    sample_id = self._normalize_id(sample_id)
                    if sample_id in train_ids:
                        train_indices.append(idx)
                    elif sample_id in val_ids:
                        val_indices.append(idx)
                    elif sample_id in test_ids:
                        test_indices.append(idx)

                self.data_train = Subset(dataset, train_indices)
                self.data_val = Subset(dataset, val_indices)
                self.data_test = Subset(dataset, test_indices)

                print(f"{split_name} split:")
                print(f"  Train: {len(train_indices)}")
                print(f"  Val:   {len(val_indices)}")
                print(f"  Test:  {len(test_indices)}")
            else:
                # Random split — supports absolute val_size/test_size for large datasets
                total_size = len(dataset)
                if self.hparams.val_size is not None and self.hparams.test_size is not None:
                    val_size = self.hparams.val_size
                    test_size = self.hparams.test_size
                    train_size = total_size - val_size - test_size
                    if train_size <= 0:
                        raise ValueError(
                            f"val_size ({val_size}) + test_size ({test_size}) = "
                            f"{val_size + test_size} must be less than total ({total_size})."
                        )
                else:
                    train_size = int(self.hparams.train_split * total_size)
                    val_size = int(self.hparams.val_split * total_size)
                    test_size = total_size - train_size - val_size

                print(f"\nDataset split:")
                print(f"  Total:  {total_size}")
                print(f"  Train:  {train_size}")
                print(f"  Val:    {val_size}")
                print(f"  Test:   {test_size}")

                self.data_train, self.data_val, self.data_test = random_split(
                    dataset, [train_size, val_size, test_size]
                )

    def train_dataloader(self):
        return PyGDataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return PyGDataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return PyGDataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )