"""Carbon-24 crystal dataset used for composition-conditioned CSP."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Lattice, Structure
from torch.utils.data import Dataset
from torch_geometric.data import Data

from src.data.components.cif import cif_to_arrays


class Carbon24Dataset(Dataset):
    """Carbon-only crystal structures from ``data/carbon_24``.

    Train/val use the provided ``*_ori.pt`` graph caches. Test structures are
    parsed from ``test.csv`` and cached under ``processed/test.pt``.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        num_elements: int = 100,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.num_elements = num_elements
        self.records = self._load_records()

    def _load_records(self) -> list[dict[str, Any]]:
        if self.split in {"train", "val"}:
            cache = self.root / f"{self.split}_ori.pt"
            table = pd.read_csv(self.root / f"{self.split}.csv")
            lookup = table.set_index("material_id").to_dict("index")
            raw = torch.load(cache, map_location="cpu", weights_only=False)
            records = []
            for item in raw:
                material_id = item["mp_id"]
                row = lookup[material_id]
                frac_coords, atom_types, lengths, angles, _, _, num_atoms = item["graph_arrays"]
                records.append(
                    {
                        "id": material_id,
                        "atom_types": np.asarray(atom_types, dtype=np.int64),
                        "frac_coords": np.asarray(frac_coords, dtype=np.float64),
                        "lengths": np.asarray(lengths, dtype=np.float64),
                        "angles": np.asarray(angles, dtype=np.float64),
                        "num_atoms": int(num_atoms),
                        "spacegroup": int(row["spacegroup.number"]),
                        "energy_per_atom": float(row["energy_per_atom"]),
                        "cif": row["cif"],
                    }
                )
            return records

        if self.split != "test":
            raise ValueError(f"Unknown split: {self.split}")
        cache = self.root / "processed" / "test.pt"
        if cache.exists():
            return torch.load(cache, map_location="cpu", weights_only=False)

        table = pd.read_csv(self.root / "test.csv")
        records = []
        for _, row in table.iterrows():
            arrays = cif_to_arrays(row["cif"])
            records.append(
                {
                    "id": row["material_id"],
                    "atom_types": np.asarray(arrays["atom_types"], dtype=np.int64),
                    "frac_coords": np.asarray(arrays["frac_coords"], dtype=np.float64),
                    "lengths": np.asarray(arrays["lengths"], dtype=np.float64),
                    "angles": np.asarray(arrays["angles"], dtype=np.float64),
                    "num_atoms": int(arrays["num_atoms"]),
                    "spacegroup": int(row["spacegroup.number"]),
                    "energy_per_atom": float(row["energy_per_atom"]),
                    "cif": row["cif"],
                }
            )
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(records, cache)
        return records

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

        data = Data(
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
            energy_per_atom=torch.tensor([float(record["energy_per_atom"])], dtype=torch.float32),
        )
        return data

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
