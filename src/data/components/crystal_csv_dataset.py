"""Generic CIF/CSV crystal dataset for the additional public datasets."""

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Lattice, Structure
from torch.utils.data import Dataset
from torch_geometric.data import Data

from src.data.components.cif import cif_to_arrays


class CrystalCsvDataset(Dataset):
    """Parse a crystal CSV into the common graph format used by the model."""

    def __init__(self, root: str | Path, split: str, num_elements: int = 100):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.num_elements = num_elements
        self.records = self._load_records()

    def _load_records(self) -> list[dict[str, Any]]:
        if self.split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {self.split}")
        csv_path = self.root / f"{self.split}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")

        signature = {
            "size": csv_path.stat().st_size,
            "mtime_ns": csv_path.stat().st_mtime_ns,
        }
        cache_path = self.root / "processed" / f"{self.split}.pt"
        if cache_path.is_file():
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            if (
                isinstance(cached, dict)
                and cached.get("version") == 1
                and cached.get("signature") == signature
            ):
                return cached["records"]

        table = pd.read_csv(csv_path)
        records = []
        skipped = 0
        print(f"[数据] 解析 {csv_path}，共 {len(table)} 条记录", flush=True)
        for index, row in table.iterrows():
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=".*Issues encountered while parsing CIF.*",
                    )
                    warnings.filterwarnings(
                        "ignore",
                        message=".*No Pauling electronegativity.*",
                    )
                    arrays = cif_to_arrays(str(row["cif"]))
            except Exception as error:
                skipped += 1
                if skipped <= 5:
                    print(f"[数据] 跳过无效 CIF 行 {index}: {error}", flush=True)
                continue
            records.append(
                {
                    "id": str(row["material_id"]),
                    "atom_types": np.asarray(arrays["atom_types"], dtype=np.int64),
                    "frac_coords": np.asarray(arrays["frac_coords"], dtype=np.float64),
                    "lengths": np.asarray(arrays["lengths"], dtype=np.float64),
                    "angles": np.asarray(arrays["angles"], dtype=np.float64),
                    "num_atoms": int(arrays["num_atoms"]),
                    "spacegroup": self._spacegroup(row),
                    "energy_per_atom": self._energy(row),
                }
            )
            if len(records) % 1000 == 0:
                print(
                    f"[数据] {csv_path.parent.name}/{self.split}: "
                    f"已解析 {len(records)}/{len(table)}",
                    flush=True,
                )
        if not records:
            raise RuntimeError(f"No valid crystal structures found in {csv_path}")

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "version": 1,
                "signature": signature,
                "records": records,
            },
            cache_path,
        )
        print(
            f"[数据] {csv_path.parent.name}/{self.split}: "
            f"有效 {len(records)}，跳过 {skipped}，缓存 {cache_path}",
            flush=True,
        )
        return records

    @staticmethod
    def _spacegroup(row: pd.Series) -> int:
        for key in ("spacegroup.number", "spacegroup.number.conv"):
            value = row.get(key)
            if value is not None and pd.notna(value):
                return int(value)
        return 0

    @staticmethod
    def _energy(row: pd.Series) -> float:
        for key in (
            "energy_per_atom",
            "formation_energy_per_atom",
            "energy_above_hull",
            "heat_all",
        ):
            value = row.get(key)
            if value is not None and pd.notna(value):
                return float(value)
        return 0.0

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Data:
        record = self.records[index]
        atom_types = torch.as_tensor(record["atom_types"], dtype=torch.long)
        frac_coords = torch.as_tensor(record["frac_coords"], dtype=torch.float32)
        lengths = torch.as_tensor(record["lengths"], dtype=torch.float32)
        angles = torch.as_tensor(record["angles"], dtype=torch.float32)
        num_atoms = int(record["num_atoms"])

        lattice = Lattice.from_parameters(*lengths.tolist(), *angles.tolist())
        cell = torch.tensor(lattice.matrix, dtype=torch.float32).unsqueeze(0)
        lengths_scaled = lengths / float(num_atoms) ** (1.0 / 3.0)
        angles_radians = torch.deg2rad(angles)
        lattices = torch.cat([lengths, angles], dim=0).unsqueeze(0)
        lattices_scaled = torch.cat([lengths_scaled, angles_radians], dim=0).unsqueeze(0)
        pos = torch.einsum(
            "bi,bij->bj",
            frac_coords,
            cell.repeat(num_atoms, 1, 1),
        )
        composition = torch.bincount(atom_types, minlength=self.num_elements).float()

        return Data(
            id=record["id"],
            atom_types=atom_types,
            pos=pos,
            frac_coords=frac_coords,
            cell=cell,
            lattices=lattices,
            lattices_scaled=lattices_scaled,
            lengths=lengths.view(1, -1),
            lengths_scaled=lengths_scaled.view(1, -1),
            angles=angles.view(1, -1),
            angles_radians=angles_radians.view(1, -1),
            num_atoms=torch.tensor([num_atoms], dtype=torch.long),
            num_nodes=torch.tensor([num_atoms], dtype=torch.long),
            token_idx=torch.arange(num_atoms, dtype=torch.long),
            dataset_idx=torch.tensor([0], dtype=torch.long),
            spacegroup=torch.tensor([int(record["spacegroup"])], dtype=torch.long),
            composition=composition.unsqueeze(0),
            energy_per_atom=torch.tensor(
                [float(record["energy_per_atom"])], dtype=torch.float32
            ),
        )

    def get_structure(self, index: int) -> Structure:
        record = self.records[index]
        lattice = Lattice.from_parameters(
            *record["lengths"].tolist(), *record["angles"].tolist()
        )
        return Structure(
            lattice=lattice,
            species=record["atom_types"].tolist(),
            coords=record["frac_coords"].tolist(),
            coords_are_cartesian=False,
        )
