"""All-atom VAE used for Carbon-24 reconstruction and composition conditioning."""

import copy
from typing import Dict, Literal, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from lightning import LightningModule
from omegaconf import DictConfig
from torch_geometric.data import Data
from torch_scatter import scatter
from torchmetrics import MeanMetric

from src.data.components.composition_utils import composition_from_batch
from src.eval.crystal_reconstruction import CrystalReconstructionEvaluator
from src.models.components.kabsch_utils import random_rotation_matrix
from src.utils import pylogger

log = pylogger.RankedLogger(__name__)


class DiagonalGaussianDistribution:
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=-1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if deterministic:
            self.var = self.std = torch.zeros_like(self.mean)

    def sample(self):
        return self.mean + self.std * torch.randn_like(self.mean)

    def kl(self):
        if self.deterministic:
            return torch.zeros(self.mean.shape[0], device=self.mean.device)
        return 0.5 * torch.sum(
            torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=1
        )

    def mode(self):
        return self.mean


class VariationalAutoencoderLitModule(LightningModule):
    """Reconstruction VAE with an optional composition-conditioned decoder."""

    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        latent_dim: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        scheduler_frequency: int,
        loss_weights: Dict,
        augmentations: DictConfig,
        visualization: DictConfig,
        compile: bool,
        composition_conditioning: bool = False,
        num_elements: int = 100,
        composition_dropout: float = 0.0,
        composition_loss_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.encoder = encoder
        self.decoder = decoder
        self.composition_conditioning = composition_conditioning
        self.num_elements = num_elements
        self.composition_dropout = composition_dropout
        self.composition_loss_weight = composition_loss_weight

        if composition_conditioning:
            self.composition_embedder = torch.nn.Sequential(
                torch.nn.Linear(num_elements, self.encoder.d_model),
                torch.nn.SiLU(),
                torch.nn.Linear(self.encoder.d_model, self.encoder.d_model),
            )

        self.quant_conv = torch.nn.Linear(self.encoder.d_model, 2 * latent_dim, bias=False)
        self.post_quant_conv = torch.nn.Linear(latent_dim, self.decoder.d_model, bias=False)

        self.loss_weights = loss_weights
        self.loss_weights_atom_types = torch.nn.Parameter(
            torch.tensor(list(self.loss_weights["loss_atom_types"].values())),
            requires_grad=False,
        )
        self.loss_weights_lengths = torch.nn.Parameter(
            torch.tensor(list(self.loss_weights["loss_lengths"].values())),
            requires_grad=False,
        )
        self.loss_weights_angles = torch.nn.Parameter(
            torch.tensor(list(self.loss_weights["loss_angles"].values())),
            requires_grad=False,
        )
        self.loss_weights_frac_coords = torch.nn.Parameter(
            torch.tensor(list(self.loss_weights["loss_frac_coords"].values())),
            requires_grad=False,
        )
        self.loss_weights_pos = torch.nn.Parameter(
            torch.tensor(list(self.loss_weights["loss_pos"].values())),
            requires_grad=False,
        )
        self.loss_weights_kl = torch.nn.Parameter(
            torch.tensor(list(self.loss_weights["loss_kl"].values())),
            requires_grad=False,
        )

        self.val_reconstruction_evaluators = {"mp20": CrystalReconstructionEvaluator()}
        self.test_reconstruction_evaluators = {"mp20": CrystalReconstructionEvaluator()}

        names = [
            "loss",
            "loss_atom_types",
            "loss_lengths",
            "loss_angles",
            "loss_frac_coords",
            "loss_pos",
            "loss_kl",
            "unscaled/loss_atom_types",
            "unscaled/loss_lengths",
            "unscaled/loss_angles",
            "unscaled/loss_frac_coords",
            "unscaled/loss_pos",
            "unscaled/loss_kl",
            "loss_composition",
        ]
        self.train_metrics = torch.nn.ModuleDict({k: MeanMetric() for k in names})
        self.val_metrics = torch.nn.ModuleDict(
            {
                "mp20": torch.nn.ModuleDict(
                    {**{k: MeanMetric() for k in names}, "match_rate": MeanMetric(), "rms_dist": MeanMetric()}
                )
            }
        )
        self.test_metrics = copy.deepcopy(self.val_metrics)

    def encode(self, batch):
        encoded = self.encoder(batch)
        encoded["posterior"] = DiagonalGaussianDistribution(self.quant_conv(encoded["x"]))
        return encoded

    def decode(self, encoded):
        if self.composition_conditioning:
            if encoded.get("composition") is None:
                raise ValueError("Composition-conditioned VAE requires composition.")
            encoded["composition_embed"] = self.composition_embedder(encoded["composition"].float())
        encoded["x"] = self.post_quant_conv(encoded["x"])
        return self.decoder(encoded)

    def forward(self, batch: Data, sample_posterior: bool = True):
        encoded = self.encode(batch)
        if self.composition_conditioning:
            composition = composition_from_batch(batch, self.num_elements)
            if self.training and self.composition_dropout > 0.0:
                drop = torch.rand(composition.size(0), 1, device=composition.device) < self.composition_dropout
                composition = composition.masked_fill(drop, 0.0)
            encoded["composition"] = composition
        encoded["x"] = encoded["posterior"].sample() if sample_posterior else encoded["posterior"].mode()
        return self.decode(encoded), encoded

    def reconstruction_criterion(self, batch, out):
        loss_atom_types = F.cross_entropy(out["atom_types"], batch.atom_types, reduction="none")
        loss_lengths = F.mse_loss(out["lengths"], batch.lengths_scaled, reduction="none").mean(dim=1)
        loss_angles = F.mse_loss(out["angles"], batch.angles_radians, reduction="none").mean(dim=1)
        loss_frac_coords = F.mse_loss(out["frac_coords"], batch.frac_coords, reduction="none").mean(dim=1)
        pos_true = batch.pos / 10.0
        pred_center = scatter(out["pos"], batch.batch, dim=0, reduce="mean")[batch.batch]
        true_center = scatter(pos_true, batch.batch, dim=0, reduce="mean")[batch.batch]
        loss_pos = F.mse_loss(out["pos"] - pred_center, pos_true - true_center, reduction="none").mean(dim=1)
        return {
            "loss_atom_types": loss_atom_types,
            "loss_lengths": loss_lengths,
            "loss_angles": loss_angles,
            "loss_frac_coords": loss_frac_coords,
            "loss_pos": loss_pos,
        }

    def criterion(self, batch, encoded, out):
        recon = self.reconstruction_criterion(batch, out)
        kl = encoded["posterior"].kl()
        w_atom = self.loss_weights_atom_types[batch.dataset_idx[batch.batch]]
        w_len = self.loss_weights_lengths[batch.dataset_idx]
        w_ang = self.loss_weights_angles[batch.dataset_idx]
        w_frac = self.loss_weights_frac_coords[batch.dataset_idx[batch.batch]]
        w_pos = self.loss_weights_pos[batch.dataset_idx[batch.batch]]
        w_kl = self.loss_weights_kl[batch.dataset_idx[batch.batch]]
        loss = (
            (w_atom * recon["loss_atom_types"]).mean()
            + (w_len * recon["loss_lengths"]).mean()
            + (w_ang * recon["loss_angles"]).mean()
            + (w_frac * recon["loss_frac_coords"]).mean()
            + (w_pos * recon["loss_pos"]).mean()
            + (w_kl * kl).mean()
        )
        losses = {
            "loss": loss,
            "loss_atom_types": w_atom * recon["loss_atom_types"],
            "loss_lengths": w_len * recon["loss_lengths"],
            "loss_angles": w_ang * recon["loss_angles"],
            "loss_frac_coords": w_frac * recon["loss_frac_coords"],
            "loss_pos": w_pos * recon["loss_pos"],
            "loss_kl": w_kl * kl,
            "unscaled/loss_atom_types": recon["loss_atom_types"],
            "unscaled/loss_lengths": recon["loss_lengths"],
            "unscaled/loss_angles": recon["loss_angles"],
            "unscaled/loss_frac_coords": recon["loss_frac_coords"],
            "unscaled/loss_pos": recon["loss_pos"],
            "unscaled/loss_kl": kl,
        }
        if self.composition_conditioning and self.composition_loss_weight > 0.0:
            probs = F.softmax(out["atom_types"], dim=-1)
            pred_counts = scatter(probs, batch.batch, dim=0, reduce="sum")
            target_counts = composition_from_batch(batch, self.num_elements)
            composition_loss = F.mse_loss(pred_counts, target_counts)
            losses["loss"] = loss + self.composition_loss_weight * composition_loss
            losses["loss_composition"] = composition_loss
        return losses

    def training_step(self, batch, batch_idx):
        with torch.no_grad():
            periodic = batch.dataset_idx == 0
            node_periodic = periodic[batch.batch]
            if self.hparams.augmentations.frac_coords and node_periodic.any():
                translation = torch.normal(batch.lengths.mean(0).abs(), batch.lengths.std(0).nan_to_num(1e-8)) / 2
                batch.pos = batch.pos + translation
                inv = torch.linalg.inv(batch.cell[batch.batch][node_periodic])
                batch.frac_coords[node_periodic] = torch.einsum(
                    "bi,bij->bj", batch.pos[node_periodic], inv
                ) % 1.0
            if self.hparams.augmentations.pos:
                rotation = random_rotation_matrix(validate=True, device=self.device)
                batch.pos = batch.pos @ rotation.T
                batch.cell = batch.cell @ rotation.T
            if self.hparams.augmentations.noise > 0.0:
                total = batch.num_atoms.sum().item()
                count = int(total * self.hparams.augmentations.noise)
                perturbed = torch.tensor(
                    np.random.choice(total, count, replace=False), device=self.device
                )
                atom_types = batch.atom_types.clone()
                pos = batch.pos.clone()
                frac = batch.frac_coords.clone()
                batch.atom_types[perturbed] = 0
                perturbed = torch.tensor(
                    np.random.choice(total, count, replace=False), device=self.device
                )
                batch.pos[perturbed] += 0.1 * torch.randn_like(batch.pos[perturbed])
                if node_periodic.any():
                    inv = torch.linalg.inv(batch.cell[batch.batch][node_periodic])
                    batch.frac_coords[node_periodic] = torch.einsum(
                        "bi,bij->bj", batch.pos[node_periodic], inv
                    ) % 1.0
        out, encoded = self.forward(batch)
        if self.hparams.augmentations.noise > 0.0:
            batch.atom_types, batch.pos, batch.frac_coords = atom_types, pos, frac
        losses = self.criterion(batch, encoded, out)
        for name, value in losses.items():
            self.train_metrics[name](value)
            self.log(
                f"train/{name}",
                self.train_metrics[name],
                on_step=True,
                on_epoch=False,
                prog_bar=False,
            )
        return losses["loss"]

    def on_train_epoch_start(self):
        for metric in self.train_metrics.values():
            metric.reset()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        self.evaluation_step(batch, batch_idx, stage="val")

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        self.evaluation_step(batch, batch_idx, stage="test")

    def on_validation_epoch_start(self):
        self.on_evaluation_epoch_start("val")

    def on_test_epoch_start(self):
        self.on_evaluation_epoch_start("test")

    def on_evaluation_epoch_start(self, stage: Literal["val", "test"]):
        metrics = self.val_metrics["mp20"] if stage == "val" else self.test_metrics["mp20"]
        for metric in metrics.values():
            metric.reset()
        evaluator = self.val_reconstruction_evaluators["mp20"] if stage == "val" else self.test_reconstruction_evaluators["mp20"]
        evaluator.clear()

    def evaluation_step(self, batch, batch_idx, stage):
        metrics = self.val_metrics["mp20"] if stage == "val" else self.test_metrics["mp20"]
        evaluator = self.val_reconstruction_evaluators["mp20"] if stage == "val" else self.test_reconstruction_evaluators["mp20"]
        evaluator.device = metrics["loss"].device
        out, encoded = self.forward(batch)
        losses = self.criterion(batch, encoded, out)
        for name, value in losses.items():
            metrics[name](value)
            self.log(
                f"{stage}/{name}",
                metrics[name],
                on_epoch=True,
                sync_dist=True,
                prog_bar=False,
            )
        start = 0
        for idx, data in enumerate(batch.to_data_list()):
            end = start + int(data.num_atoms)
            sample_idx = batch_idx * batch.batch_size + idx
            evaluator.append_pred_array(
                {
                    "atom_types": out["atom_types"][start:end].argmax(-1).detach().cpu().numpy(),
                    "pos": out["pos"][start:end].detach().cpu().numpy() * 10.0,
                    "frac_coords": out["frac_coords"][start:end].detach().cpu().numpy(),
                    "lengths": out["lengths"][idx].detach().cpu().numpy() * data.num_atoms.item() ** (1 / 3),
                    "angles": torch.rad2deg(out["angles"][idx]).detach().cpu().numpy(),
                    "sample_idx": sample_idx,
                }
            )
            start = end
            evaluator.append_gt_array(
                {
                    "atom_types": data.atom_types.detach().cpu().numpy(),
                    "pos": data.pos.detach().cpu().numpy(),
                    "frac_coords": data.frac_coords.detach().cpu().numpy(),
                    "lengths": data.lengths.squeeze().detach().cpu().numpy(),
                    "angles": data.angles.squeeze().detach().cpu().numpy(),
                    "sample_idx": sample_idx,
                }
            )

    def on_validation_epoch_end(self):
        self.on_evaluation_epoch_end("val")

    def on_test_epoch_end(self):
        self.on_evaluation_epoch_end("test")

    def on_evaluation_epoch_end(self, stage):
        evaluator = self.val_reconstruction_evaluators["mp20"] if stage == "val" else self.test_reconstruction_evaluators["mp20"]
        if not evaluator.pred_arrays_list:
            return
        metrics = self.val_metrics["mp20"] if stage == "val" else self.test_metrics["mp20"]
        for name, value in evaluator.get_metrics().items():
            metrics[name](value)
            self.log(
                f"{stage}/{name}",
                metrics[name],
                on_epoch=True,
                sync_dist=True,
                prog_bar=False,
            )

    def configure_optimizers(self):
        optimizer_factory = self.hparams.optimizer
        if getattr(optimizer_factory, "func", None) is torch.optim.AdamW:
            optimizer = optimizer_factory(params=self.parameters())
        else:
            optimizer = optimizer_factory(params=self.named_parameters())
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
