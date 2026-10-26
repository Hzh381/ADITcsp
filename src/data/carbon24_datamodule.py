"""LightningDataModule for the Carbon-24 CSP dataset."""

from typing import Optional

from lightning import LightningDataModule
from omegaconf import DictConfig
from torch_geometric.loader import DataLoader

from src.data.components.carbon24_dataset import Carbon24Dataset


class Carbon24DataModule(LightningDataModule):
    def __init__(
        self,
        root: str,
        batch_size: DictConfig,
        num_workers: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = Carbon24Dataset(self.hparams.root, "train")
        self.val_dataset = Carbon24Dataset(self.hparams.root, "val")
        self.test_dataset = Carbon24Dataset(self.hparams.root, "test")

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
