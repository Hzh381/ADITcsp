"""DiverseDiT denoiser for composition-conditioned crystal latent diffusion.

This module keeps the standard :class:`~src.models.denoisers.dit.DiT`
backbone and adds the two mechanisms proposed in DiverseDiT:

1. mirrored long residual connections between the two halves of the block
   stack;
2. an inter-block representation diversity loss.

The original ``DiT`` and ``CoDeMDT`` implementations are left unchanged.
"""

from __future__ import annotations

import itertools
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.denoisers.dit import DiT, get_pos_embedding


class FP32LayerNorm(nn.LayerNorm):
    """Layer normalization evaluated in FP32 for mixed-precision stability."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = F.layer_norm(
            inputs.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.to(inputs.dtype)


class LongResidualFusion(nn.Module):
    """Fuse a later block output with the mirrored earlier-half representation."""

    def __init__(self, d_model: int):
        super().__init__()
        self.norm = FP32LayerNorm(2 * d_model, eps=1e-6)
        self.proj = nn.Linear(2 * d_model, d_model)

    def forward(self, current: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(torch.cat([current, skip], dim=-1)))


class BlockDiversityLoss(nn.Module):
    """DiverseDiT block-diversity objective on padded token sequences.

    The three paper terms are:

    * orthogonality between block mean features;
    * a cosine-similarity proxy for mutual-information minimization;
    * dispersion of channel-wise activation usage.

    Only valid crystal tokens participate in all statistics.
    """

    def __init__(
        self,
        block_count: int = 10,
        max_pairs: int = 10,
        lambda_orth: float = 0.33,
        lambda_mi: float = 0.33,
        lambda_disp: float = 0.33,
        mi_absolute: bool = True,
        loss_clamp: float = 5.0,
    ):
        super().__init__()
        self.block_count = int(block_count)
        self.max_pairs = int(max_pairs)
        self.lambda_orth = float(lambda_orth)
        self.lambda_mi = float(lambda_mi)
        self.lambda_disp = float(lambda_disp)
        self.mi_absolute = bool(mi_absolute)
        self.loss_clamp = float(loss_clamp)

    def _select_features(
        self,
        features: Sequence[torch.Tensor],
    ) -> list[tuple[int, torch.Tensor]]:
        count = len(features)
        if self.block_count <= 0 or self.block_count >= count:
            indices = torch.arange(count, device=features[0].device)
        else:
            indices = torch.randperm(count, device=features[0].device)[
                : self.block_count
            ].sort().values
        return [(int(index), features[int(index)]) for index in indices]

    def forward(
        self,
        features: Sequence[torch.Tensor],
        token_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor | int]:
        if len(features) < 2:
            zero = features[0].new_zeros(())
            return {
                "div_loss": zero,
                "loss_orth": zero,
                "loss_mi": zero,
                "loss_disp": zero,
                "num_pairs": 0,
                "selected_blocks": len(features),
            }

        selected = self._select_features(features)
        pairs = list(itertools.combinations(selected, 2))
        if self.max_pairs > 0:
            pairs = pairs[: self.max_pairs]

        mask = token_mask.bool()
        counts = mask.sum().clamp_min(1).to(selected[0][1].dtype)
        means = {
            index: (feature * mask[..., None]).sum(dim=(0, 1)) / counts
            for index, feature in selected
        }

        orth_terms = []
        mi_terms = []
        for (index_i, feat_i), (index_j, feat_j) in pairs:
            orth_terms.append(
                F.cosine_similarity(
                    means[index_i],
                    means[index_j],
                    dim=0,
                    eps=1e-8,
                )
            )

            normalized_i = F.normalize(feat_i, p=2, dim=-1, eps=1e-8)
            normalized_j = F.normalize(feat_j, p=2, dim=-1, eps=1e-8)
            token_cosine = (normalized_i * normalized_j).sum(dim=-1)
            pair_mi = (token_cosine * mask).sum() / counts
            mi_terms.append(pair_mi.abs() if self.mi_absolute else pair_mi)

        if orth_terms:
            loss_orth = torch.stack(orth_terms).mean()
            loss_mi = torch.stack(mi_terms).mean()
        else:
            loss_orth = features[0].new_zeros(())
            loss_mi = features[0].new_zeros(())

        activations = []
        for _, feature in selected:
            valid = feature[mask]
            if valid.numel() == 0:
                continue
            normalized = F.normalize(valid, p=2, dim=0, eps=1e-8)
            activations.append(normalized.abs().mean(dim=0))
        if activations:
            activation = torch.stack(activations, dim=0).mean(dim=0)
            activation = activation / activation.abs().amax().clamp_min(1e-8)
            variance = activation.var(unbiased=False)
            loss_disp = -torch.clamp(variance / 0.25, min=0.0, max=1.0)
        else:
            loss_disp = features[0].new_zeros(())

        total = (
            self.lambda_orth * loss_orth
            + self.lambda_mi * loss_mi
            + self.lambda_disp * loss_disp
        )
        total = torch.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0)
        if self.loss_clamp > 0:
            total = total.clamp(max=self.loss_clamp)
        return {
            "div_loss": total,
            "loss_orth": loss_orth,
            "loss_mi": loss_mi,
            "loss_disp": loss_disp,
            "num_pairs": len(pairs),
            "selected_blocks": len(selected),
        }


class DiverseDiT(DiT):
    """Standard DiT with DiverseDiT residual and representation-diversity terms."""

    supports_masking = False
    has_diversity = True

    def __init__(
        self,
        *args,
        long_residual: bool = True,
        diversity_loss: bool = True,
        diversity_blocks: int = 10,
        diversity_max_pairs: int = 10,
        lambda_orth: float = 0.33,
        lambda_mi: float = 0.33,
        lambda_disp: float = 0.33,
        diversity_loss_scale: float = 1.0,
        diversity_mi_absolute: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_layers = len(self.blocks)
        self.long_residual = bool(long_residual)
        if self.long_residual and self.num_layers % 2 != 0:
            raise ValueError(
                "DiverseDiT long residual connections require an even number of blocks."
            )

        half = self.num_layers // 2
        self.long_residual_fusions = (
            nn.ModuleList([LongResidualFusion(self.d_model) for _ in range(half)])
            if self.long_residual
            else nn.ModuleList()
        )
        for fusion in self.long_residual_fusions:
            nn.init.xavier_uniform_(fusion.proj.weight)
            nn.init.zeros_(fusion.proj.bias)

        self.diversity_loss_enabled = bool(diversity_loss)
        self.diversity_loss_fn = BlockDiversityLoss(
            block_count=diversity_blocks,
            max_pairs=diversity_max_pairs,
            lambda_orth=lambda_orth,
            lambda_mi=lambda_mi,
            lambda_disp=lambda_disp,
            mi_absolute=diversity_mi_absolute,
        )
        self.diversity_loss_scale = float(diversity_loss_scale)
        self._last_div = None
        self._last_div_weight = 0.0

    def _condition(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        dataset_idx: torch.Tensor,
        spacegroup: torch.Tensor,
        composition: torch.Tensor | None,
    ) -> torch.Tensor:
        t_embed = self.t_embedder(t.squeeze(1))
        dataset_embed = self.dataset_embedder(dataset_idx, self.training)
        spacegroup_embed = self.spacegroup_embedder(spacegroup, self.training)
        condition = t_embed + dataset_embed + spacegroup_embed
        if self.composition_embedder is not None:
            if composition is None:
                composition = torch.zeros(
                    x.size(0),
                    self.composition_dim,
                    device=x.device,
                    dtype=x.dtype,
                )
            composition = composition.to(device=x.device, dtype=x.dtype)
            if self.training and self.composition_dropout_prob > 0.0:
                drop = (
                    torch.rand(composition.size(0), 1, device=composition.device)
                    < self.composition_dropout_prob
                )
                composition = composition.masked_fill(drop, 0.0)
            condition = condition + self.composition_embedder(composition)
        return condition

    def forward(
        self,
        x,
        t,
        dataset_idx,
        spacegroup,
        mask,
        x_sc=None,
        composition=None,
    ):
        self._last_div = None
        self._last_div_weight = 0.0

        token_index = torch.cumsum(mask, dim=-1, dtype=torch.int64) - 1
        pos_emb = get_pos_embedding(token_index, self.d_model)
        if x_sc is None:
            x_sc = torch.zeros_like(x)
        x = self.x_embedder(torch.cat([x, x_sc], dim=-1)) + pos_emb
        condition = self._condition(
            x,
            t,
            dataset_idx,
            spacegroup,
            composition,
        )

        half = self.num_layers // 2
        block_features: list[torch.Tensor] = []
        collect_diversity = self.training and self.diversity_loss_enabled
        for index, block in enumerate(self.blocks):
            x = block(x, condition, ~mask)
            if self.long_residual and index >= half:
                source_index = self.num_layers - 1 - index
                x = self.long_residual_fusions[index - half](
                    x,
                    block_features[source_index],
                )
            if self.long_residual or collect_diversity:
                block_features.append(x)

        output = self.final_layer(x, condition)
        output = output * mask[..., None]

        if collect_diversity:
            self._last_div = self.diversity_loss_fn(block_features, mask)
        return output

    def diversity_loss(self):
        return self._last_div

    def diversity_components(self):
        if self._last_div is None:
            return {}
        return {
            key: float(value.detach())
            for key, value in self._last_div.items()
            if isinstance(value, torch.Tensor)
        }

    def diversity_weight(self, divergence_loss: torch.Tensor | None = None):
        loss = self._last_div["div_loss"] if divergence_loss is None else divergence_loss
        value = float(loss.detach())
        if value > 0.5:
            return 1.0
        if value > 0.1:
            return (value - 0.1) / 0.5
        return 0.0

    def diversity_term(self, step, diffusion_loss=None):
        if self._last_div is None:
            return None
        divergence_loss = self._last_div["div_loss"]
        weight = self.diversity_weight(divergence_loss)
        self._last_div_weight = weight
        return self.diversity_loss_scale * weight * divergence_loss
