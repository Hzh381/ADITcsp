"""Composition-conditioned latent flow-matching model for Carbon-24."""

import math
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
from src.models.crystal_mmfm_prior import CrystalMMFMPrior
from src.models.decoder_struct_loss import compute_struct_loss
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
        ot_coupler: torch.nn.Module | None = None,
        posterior_mmd: torch.nn.Module | None = None,
        mmd_weight: float = 0.0,
        mmd_every: int = 0,
        mmd_steps: int = 100,
        mmd_batch_size: int = 0,
        decoder_aware: bool = False,
        cell_loss_weight: float = 0.1,
        geo_loss_weight: float = 0.1,
        coord_loss_weight: float = 0.1,
        coord_max_shifts: int = 1,
        coord_shift_candidates: int = 8,
        mmfm_prior_path: str | None = None,
        mmfm_mode_coupling: bool = False,
        mmfm_balanced: bool = False,
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
        self.ot_coupler = ot_coupler
        self.posterior_mmd = posterior_mmd
        self.mmd_weight = float(mmd_weight)
        self.mmd_every = int(mmd_every)
        self.mmd_steps = int(mmd_steps)
        self.mmd_batch_size = int(mmd_batch_size)
        self.decoder_aware = bool(decoder_aware)
        self.cell_loss_weight = float(cell_loss_weight)
        self.geo_loss_weight = float(geo_loss_weight)
        self.coord_loss_weight = float(coord_loss_weight)
        self.coord_max_shifts = int(coord_max_shifts)
        self.coord_shift_candidates = int(coord_shift_candidates)
        self.mmfm_prior_path = mmfm_prior_path
        self.mmfm_mode_coupling = bool(mmfm_mode_coupling)
        self.mmfm_balanced = bool(mmfm_balanced)
        self.mmfm_prior = (
            CrystalMMFMPrior.load(mmfm_prior_path) if mmfm_prior_path else None
        )
        if self.decoder_aware and ot_coupler is not None:
            raise ValueError(
                "Decoder-aware FM and minibatch OT coupling are alternative "
                "pairing schemes; run them in separate experiments."
            )
        if self.mmfm_prior is not None and ot_coupler is not None:
            raise ValueError(
                "Crystal-MM-FM and minibatch OT coupling are alternative "
                "pairing schemes; run them in separate experiments."
            )
        if self.decoder_aware and (
            self.cell_loss_weight < 0.0
            or self.geo_loss_weight < 0.0
            or self.coord_loss_weight < 0.0
        ):
            raise ValueError("Decoder-aware loss weights must be non-negative.")
        if self.coord_max_shifts < 0 or self.coord_shift_candidates < 0:
            raise ValueError(
                "coord_max_shifts and coord_shift_candidates must be non-negative."
            )
        self.masking_enabled = bool(getattr(denoiser, "supports_masking", False))
        self.has_diversity = bool(getattr(denoiser, "has_diversity", False))
        names = ["loss", "x_loss"]
        if self.masking_enabled:
            names += ["plain_loss", "m_loss", "m_x_loss"]
        if self.has_diversity:
            names += [
                "div_loss",
                "div_raw_loss",
                "loss_orth",
                "loss_mi",
                "loss_disp",
                "div_weight",
            ]
        if self.ot_coupler is not None:
            names.append("ot_cost")
        if self.posterior_mmd is not None:
            names.extend(["mmd_loss", "mmd_raw_loss", "mmd_bandwidth"])
        if self.decoder_aware:
            names.extend(["cell_loss", "geo_loss", "coord_loss"])
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

    def train(self, mode: bool = True):
        """Keep the frozen autoencoder in eval mode when the parent trains.

        ``requires_grad_(False)`` only stops gradient updates; it does not
        disable dropout. Lightning puts the whole module in training mode,
        which would silently re-enable the autoencoder's dropout and change
        the latent targets between training and inference.
        """
        super().train(mode)
        self.autoencoder.eval()
        return self

    def _prepare_batch(self, batch: Data):
        """Encode one crystal batch and prepare the latent flow-matching inputs."""
        with torch.no_grad():
            encoded = self.autoencoder.encode(batch)
            x_1 = encoded["posterior"].sample()
            x_1, mask = to_dense_batch(x_1, encoded["batch"])
        composition = composition_from_batch(batch, self.denoiser.composition_dim)
        group_ids = None
        if self.ot_coupler is not None:
            _, group_ids = torch.unique(
                composition.detach(),
                dim=0,
                return_inverse=True,
            )
        self.interpolant.device = x_1.device
        x_0 = None
        if self.mmfm_prior is not None:
            batch_ids = getattr(batch, "id", None)
            if isinstance(batch_ids, str):
                batch_ids = [batch_ids]
            elif batch_ids is not None and not isinstance(batch_ids, list):
                batch_ids = [str(value) for value in batch_ids]
            x_0 = self.mmfm_prior.sample_train_source(
                composition,
                mask,
                batch_ids=batch_ids,
                mode_coupling=self.mmfm_mode_coupling,
            )
        noisy = self.interpolant.corrupt_batch(
            {
                "x_1": x_1,
                "token_mask": mask,
                "diffuse_mask": mask,
            },
            ot_coupler=self.ot_coupler if self.training else None,
            group_ids=group_ids,
            x_0=x_0,
        )
        if self.decoder_aware:
            target_frac, _ = to_dense_batch(batch.frac_coords, encoded["batch"])
            noisy["target_frac"] = target_frac
            noisy["target_lengths"] = batch.lengths
            noisy["target_angles"] = batch.angles_radians
            noisy["target_atom_types"] = batch.atom_types
        dataset_idx = batch.dataset_idx + 1
        spacegroup = torch.zeros_like(batch.spacegroup)
        return noisy, {
            "dataset_idx": dataset_idx,
            "spacegroup": spacegroup,
            "mask": mask,
            "composition": composition,
            "num_atoms": batch.num_atoms.view(-1),
        }

    def _terminal_mmd_loss(self, noisy, conditioning):
        batch_size = conditioning["mask"].size(0)
        if self.mmd_batch_size > 0:
            batch_size = min(batch_size, self.mmd_batch_size)
        if batch_size < 2:
            return None, None

        mask = conditioning["mask"][:batch_size]
        composition = conditioning["composition"][:batch_size]
        dataset_idx = conditioning["dataset_idx"][:batch_size]
        spacegroup = conditioning["spacegroup"][:batch_size]
        posterior = noisy["x_1"][:batch_size]
        x_0 = None
        if self.mmfm_prior is not None:
            x_0 = self.mmfm_prior.sample_train_source(
                composition,
                mask,
                mode_coupling=False,
            )

        was_training = self.denoiser.training
        self.denoiser.eval()
        try:
            generated = self.interpolant.sample_for_alignment(
                batch_size=batch_size,
                num_tokens=mask.size(1),
                emb_dim=self.denoiser.d_x,
                model=self.denoiser,
                dataset_idx=dataset_idx,
                spacegroup=spacegroup,
                num_timesteps=self.mmd_steps,
                token_mask=mask,
                composition=composition,
                x_0=x_0,
            )
        finally:
            self.denoiser.train(was_training)
        mmd_loss = self.posterior_mmd(generated, posterior, mask)
        bandwidth = self.posterior_mmd.last_bandwidth
        return mmd_loss, bandwidth

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

    def _decoder_structure_loss(self, pred, conditioning, noisy):
        """Decode the predicted endpoint and return cell/periodic losses."""
        mask = conditioning["mask"]
        num_atoms = conditioning["num_atoms"]
        batch_size = pred.size(0)
        flat_batch = torch.repeat_interleave(
            torch.arange(batch_size, device=pred.device),
            num_atoms,
        )
        flat = {
            "x": pred[mask],
            "num_atoms": num_atoms,
            "batch": flat_batch,
            "token_idx": (torch.cumsum(mask, dim=-1, dtype=torch.int64) - 1)[mask],
            "composition": conditioning["composition"],
        }
        decoded = self.autoencoder.decode(flat)
        pred_frac, _ = to_dense_batch(
            decoded["frac_coords"],
            flat_batch,
            max_num_nodes=mask.size(1),
        )
        scale = num_atoms.to(pred.dtype).pow(1.0 / 3.0)
        pred_lengths = decoded["lengths"] * scale[:, None]
        pred_frac = torch.nan_to_num(pred_frac, nan=0.0)
        pred_lengths = torch.nan_to_num(
            pred_lengths,
            nan=1.0,
            posinf=1e3,
            neginf=1e-3,
        )
        pred_angles = torch.nan_to_num(
            decoded["angles"],
            nan=math.pi / 2.0,
            posinf=math.pi - 0.1,
            neginf=0.1,
        )
        pred_types = None
        target_types = None
        if self.coord_loss_weight > 0.0:
            pred_types_flat = decoded["atom_types"].argmax(dim=-1)
            pred_types, _ = to_dense_batch(
                pred_types_flat.float(),
                flat_batch,
                max_num_nodes=mask.size(1),
            )
            target_flat = noisy.get("target_atom_types")
            if target_flat is not None:
                target_types, _ = to_dense_batch(
                    target_flat.float(),
                    flat_batch,
                    max_num_nodes=mask.size(1),
                )
                target_types = target_types.long()
            pred_types = pred_types.long()
            if target_types is None:
                pred_types = None
        result = compute_struct_loss(
            pred_frac,
            pred_lengths,
            pred_angles,
            noisy["target_frac"],
            noisy["target_lengths"],
            noisy["target_angles"],
            mask,
            pred_types=pred_types,
            target_types=target_types,
            include_coord=self.coord_loss_weight > 0.0,
            coord_max_shifts=self.coord_max_shifts,
            coord_shift_candidates=self.coord_shift_candidates,
        )
        return {
            key: value if torch.isfinite(value) else pred.new_zeros(())
            for key, value in result.items()
        }

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
        pred = self._denoise(noisy, conditioning)
        plain = self.criterion(noisy, pred)
        logs = {
            "loss": plain["loss"],
            "x_loss": plain["x_loss"],
        }
        total = plain["loss"]
        if self.decoder_aware and (
            self.cell_loss_weight > 0.0
            or self.geo_loss_weight > 0.0
            or self.coord_loss_weight > 0.0
        ):
            structure = self._decoder_structure_loss(pred, conditioning, noisy)
            cell_term = self.cell_loss_weight * structure["cell_loss"]
            geo_term = self.geo_loss_weight * structure["geo_loss"]
            coord_term = self.coord_loss_weight * structure["coord_loss"]
            total = total + cell_term + geo_term + coord_term
            logs["cell_loss"] = structure["cell_loss"].detach()
            logs["geo_loss"] = structure["geo_loss"].detach()
            logs["coord_loss"] = structure["coord_loss"].detach()
        if self.ot_coupler is not None:
            logs["ot_cost"] = torch.as_tensor(
                0.0
                if self.ot_coupler.last_cost is None
                else self.ot_coupler.last_cost,
                device=total.device,
                dtype=total.dtype,
            )

        if self.masking_enabled:
            logs["plain_loss"] = plain["loss"]
            masked = self.criterion(
                noisy,
                self._denoise(noisy, conditioning, enable_mask=True),
            )
            total = total + self.hparams.mask_loss_weight * masked["loss"]
            logs["m_loss"] = masked["loss"]
            logs["m_x_loss"] = masked["x_loss"]

        if self.has_diversity:
            diversity_term = getattr(self.denoiser, "diversity_term", None)
            if diversity_term is not None:
                term = diversity_term(self.global_step, plain["loss"].detach())
                if term is not None:
                    total = total + term
                    logs["div_loss"] = term.detach()
                    components = self.denoiser.diversity_components()
                    if "div_loss" in components:
                        logs["div_raw_loss"] = torch.as_tensor(
                            components["div_loss"],
                            device=total.device,
                            dtype=total.dtype,
                        )
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

        if self.posterior_mmd is not None:
            should_align = (
                self.mmd_every > 0
                and self.mmd_weight > 0.0
                and self.global_step % self.mmd_every == 0
            )
            mmd_raw_loss = None
            mmd_bandwidth = None
            if should_align:
                mmd_raw_loss, mmd_bandwidth = self._terminal_mmd_loss(
                    noisy,
                    conditioning,
                )
            if mmd_raw_loss is None:
                mmd_raw_loss = total.new_zeros(())
            mmd_loss = self.mmd_weight * mmd_raw_loss
            total = total + mmd_loss
            logs["mmd_loss"] = mmd_loss.detach()
            logs["mmd_raw_loss"] = mmd_raw_loss.detach()
            if mmd_bandwidth is None and self.posterior_mmd.last_bandwidth is not None:
                mmd_bandwidth = self.posterior_mmd.last_bandwidth
            logs["mmd_bandwidth"] = torch.as_tensor(
                0.0 if mmd_bandwidth is None else mmd_bandwidth,
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
        x_0 = None
        if self.mmfm_prior is not None:
            x_0 = self.mmfm_prior.sample_inference_source(
                composition,
                mask,
                balanced=self.mmfm_balanced,
            )
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
            x_0=x_0,
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
