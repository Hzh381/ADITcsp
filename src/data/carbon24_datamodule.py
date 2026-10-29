"""Backward-compatible Carbon-24 data module."""

from src.data.crystal_datamodule import CrystalDataModule


class Carbon24DataModule(CrystalDataModule):
    def __init__(self, root: str, batch_size: dict, num_workers: dict) -> None:
        super().__init__(
            root=root,
            dataset_name="carbon_24",
            batch_size=batch_size,
            num_workers=num_workers,
        )
