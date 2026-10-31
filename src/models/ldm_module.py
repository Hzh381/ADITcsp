"""Composition-conditioned latent flow-matching model for Carbon-24."""

import random
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch
from torchmetrics import MeanMetric

from src.data.components.composition_utils import composition_from_batch
from src.models.vae_module import VariationalAutoencoderLitModule
from src.utils import pylogger

log = pylogger.RankedLogger(__name__)


class LatentDiffusionLitModule(LightningModule):
    def __init__(
        self,
        autoencoder_ckpt: str,
        denoiser: torch.nn.Module,
        interpolant,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        scheduler_frequency: int,
        compile: bool = False,
        mask_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.autoencoder = VariationalAutoencoderLitModule.load_from_checkpoint(
            autoencoder_ckpt, map_location="cpu"
        )
        self.autoencoder.requires_grad_(False)
        self.autoencoder.eval()
        if not getattr(self.autoencoder, "composition_conditioning", False):
            raise ValueError("The VAE checkpoint must be composition-conditioned.")
        self.denoiser = denoiser
        self.interpolant = interpolant
        self.masking_enabled = bool(getattr(denoiser, "supports_masking", False))
        self.has_diversity = bool(getattr(denoiser, "has_diversity", False))
        names = ["loss", "x_loss"]
        if self.masking_enabled:
            names += ["plain_loss", "m_loss", "m_x_loss"]
        if self.has_diversity:
            names += [
                "div_loss",
                "loss_orth",
                "loss_mi",
                "loss_disp",
                "div_weight",
            ]
        self.train_metrics = torch.nn.ModuleDict({k: MeanMetric() for k in names})
        self.val_metrics = torch.nn.ModuleDict(
            {
                "mp20": torch.nn.ModuleDict(
                    {k: MeanMetric() for k in ["loss", "x_loss"]}
                )
            }
        )
        self.test_metrics = torch.nn.ModuleDict(
            {
                "mp20": torch.nn.ModuleDict(
                    {k: MeanMetric() for k in ["loss", "x_loss"]}
                )
            }
        )

    def _prepare_batch(self, batch: Data):
        """Encode one crystal batch and prepare the latent flow-matching inputs."""
        with torch.no_grad():
            encoded = self.autoencoder.encode(batch)
            x_1 = encoded["posterior"].sample()
            x_1, mask = to_dense_batch(x_1, encoded["batch"])
        self.interpolant.device = x_1.device
        noisy = self.interpolant.corrupt_batch(
            {"x_1": x_1, "token_mask": mask, "diffuse_mask": mask}
        )
        dataset_idx = batch.dataset_idx + 1
        composition = composition_from_batch(batch, self.denoiser.composition_dim)
        spacegroup = torch.zeros_like(batch.spacegroup)
        return noisy, {
            "dataset_idx": dataset_idx,
            "spacegroup": spacegroup,
            "mask": mask,
            "composition": composition,
        }

    def _denoise(self, noisy, conditioning, enable_mask: bool = False):
        """Run one denoiser pass, including optional learned self-conditioning."""
        x_sc = None
        if self.interpolant.self_condition and random.random() < self.interpolant.self_condition_prob:
            with torch.no_grad():
                x_sc = self.denoiser(
                    x=noisy["x_t"],
                    t=noisy["t"],
                    dataset_idx=conditioning["dataset_idx"],
                    spacegroup=conditioning["spacegroup"],
                    mask=conditioning["mask"],
                    x_sc=None,
                    composition=conditioning["composition"],
                )
        model_kwargs = {
            "x": noisy["x_t"],
            "t": noisy["t"],
            "dataset_idx": conditioning["dataset_idx"],
            "spacegroup": conditioning["spacegroup"],
            "mask": conditioning["mask"],
            "x_sc": x_sc,
            "composition": conditioning["composition"],
        }
        if enable_mask:
            if not self.masking_enabled:
                raise ValueError("This denoiser does not support masked training.")
            model_kwargs["enable_mask"] = True
        return self.denoiser(**model_kwargs)

    def forward(self, batch: Data):
        noisy, conditioning = self._prepare_batch(batch)
        return self._denoise(noisy, conditioning), noisy

    def criterion(self, noisy, pred):
        target = noisy["x_1"]
        scale = 1 - torch.min(noisy["t"].unsqueeze(-1), torch.tensor(0.9, device=target.device))
        error = (target - pred) / scale
        valid = noisy["token_mask"] * noisy["diffuse_mask"]
        denominator = valid.sum(-1) * pred.size(-1)
        x_loss = (error.square() * valid[..., None]).sum(dim=(-1, -2)) / denominator
        return {"loss": x_loss.mean(), "x_loss": x_loss}

    def _log_losses(self, losses, stage):
        metrics = getattr(self, f"{stage}_metrics")["mp20"]
        for name, value in losses.items():
            metrics[name](value)
            self.log(
                f"{stage}/{name}",
                metrics[name],
                on_epoch=True,
                sync_dist=True,
                prog_bar=False,
            )

    def training_step(self, batch, batch_idx):
        noisy, conditioning = self._prepare_batch(batch)
        plain = self.criterion(noisy, self._denoise(noisy, conditioning))
        logs = {
            "loss": plain["loss"],
            "x_loss": plain["x_loss"],
        }
        total = plain["loss"]

        if self.masking_enabled:
            logs["plain_loss"] = plain["loss"]
            masked = self.criterion(
                noisy,
                self._denoise(noisy, conditioning, enable_mask=True),
            )
            total = total + self.hparams.mask_loss_weight * masked["loss"]
            logs["m_loss"] = masked["loss"]
            logs["m_x_loss"] = masked["x_loss"]

            diversity_term = getattr(self.denoiser, "diversity_term", None)
            if diversity_term is not None:
                term = diversity_term(self.global_step, plain["loss"].detach())
                if term is not None:
                    total = total + term
                    logs["div_loss"] = term.detach()
                    components = self.denoiser.diversity_components()
                    for name in ("loss_orth", "loss_mi", "loss_disp"):
                        if name in components:
                            logs[name] = torch.as_tensor(
                                components[name],
                                device=total.device,
                                dtype=total.dtype,
                            )
                    logs["div_weight"] = torch.as_tensor(
                        self.denoiser.diversity_weight(),
                        device=total.device,
                        dtype=total.dtype,
                    )

        logs["loss"] = total
        for name, value in logs.items():
            self.train_metrics[name](value)
            self.log(
                f"train/{name}",
                self.train_metrics[name],
                on_step=True,
                on_epoch=False,
                prog_bar=False,
            )
        return total

    def on_train_epoch_start(self):
        for metric in self.train_metrics.values():
            metric.reset()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        pred, noisy = self.forward(batch)
        self._log_losses(self.criterion(noisy, pred), stage="val")

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        pred, noisy = self.forward(batch)
        self._log_losses(self.criterion(noisy, pred), stage="test")

    def on_validation_epoch_start(self):
        for metric in self.val_metrics["mp20"].values():
            metric.reset()

    def on_test_epoch_start(self):
        for metric in self.test_metrics["mp20"].values():
            metric.reset()

    def sample_and_decode(self, composition: torch.Tensor, cfg_scale: float = 2.0):
        self.interpolant.device = self.device
        composition = composition.to(self.device, dtype=torch.float32)
        batch_size = composition.size(0)
        lengths = composition.sum(-1).long()
        if torch.any(lengths <= 0):
            raise ValueError("Composition must contain at least one atom.")
        dataset_idx = torch.ones(batch_size, dtype=torch.long, device=self.device)
        spacegroup = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        mask = torch.zeros(batch_size, int(lengths.max()), dtype=torch.bool, device=self.device)
        for i, length in enumerate(lengths):
            mask[i, :length] = True
        samples = self.interpolant.sample_with_classifier_free_guidance(
            batch_size=batch_size,
            num_tokens=int(lengths.max()),
            emb_dim=self.denoiser.d_x,
            model=self.denoiser,
            dataset_idx=dataset_idx,
            spacegroup=spacegroup,
            cfg_scale=cfg_scale,
            token_mask=mask,
            composition=composition,
        )
        x = samples["clean_traj"][-1][mask]
        batch = {
            "x": x,
            "num_atoms": lengths,
            "batch": torch.repeat_interleave(torch.arange(batch_size, device=self.device), lengths),
            "token_idx": (torch.cumsum(mask, dim=-1, dtype=torch.int64) - 1)[mask],
            "composition": composition,
        }
        return self.autoencoder.decode(batch), batch, samples

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.autoencoder = torch.compile(self.autoencoder)
            self.denoiser = torch.compile(self.denoiser)

    def configure_optimizers(self):
        optimizer_factory = self.hparams.optimizer
        if getattr(optimizer_factory, "func", None) is torch.optim.AdamW:
            optimizer = optimizer_factory(params=self.denoiser.parameters())
        else:
            optimizer = optimizer_factory(params=self.denoiser.named_parameters())
        if self.hparams.scheduler is None:
            return {"optimizer": optimizer}
        scheduler = self.hparams.scheduler(optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": self.hparams.scheduler_frequency,
            },
        }
