"""Dataset selection and Lightning data loading for all crystal datasets."""

from pathlib import Path
from typing import Optional

from lightning import LightningDataModule
from torch_geometric.loader import DataLoader

from src.data.components.carbon24_dataset import Carbon24Dataset
from src.data.components.crystal_csv_dataset import CrystalCsvDataset


SUPPORTED_DATASETS = ("carbon_24", "mp_20", "mpts_52", "perov_5")


def build_crystal_dataset(
    root: str | Path,
    dataset_name: str,
    split: str,
    num_elements: int = 100,
):
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    root = Path(root)
    if dataset_name == "carbon_24":
        return Carbon24Dataset(root, split, num_elements=num_elements)
    return CrystalCsvDataset(root, split, num_elements=num_elements)


class CrystalDataModule(LightningDataModule):
    def __init__(
        self,
        root: str,
        dataset_name: str,
        batch_size: dict,
        num_workers: dict,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

    def setup(self, stage: Optional[str] = None) -> None:
        root = Path(self.hparams.root)
        self.train_dataset = build_crystal_dataset(
            root, self.hparams.dataset_name, "train"
        )
        self.val_dataset = build_crystal_dataset(
            root, self.hparams.dataset_name, "val"
        )
        self.test_dataset = build_crystal_dataset(
            root, self.hparams.dataset_name, "test"
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size["train"],
            shuffle=True,
            num_workers=self.hparams.num_workers["train"],
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size["val"],
            shuffle=False,
            num_workers=self.hparams.num_workers["val"],
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size["test"],
            shuffle=False,
            num_workers=self.hparams.num_workers["test"],
        )
