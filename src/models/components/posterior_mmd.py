"""Terminal distribution alignment between generated and posterior latents."""

from __future__ import annotations

import torch
import torch.nn as nn


class PosteriorMMD(nn.Module):
    """Biased multi-scale RBF MMD between two latent batches.

    For padded ``[B, N, D]`` latents, the module flattens all valid tokens
    when every sample has the same token mask; otherwise it uses a masked
    mean over atom tokens so batches of different sizes remain comparable.
    """

    def __init__(
        self,
        kernel_scales: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
        epsilon: float = 1e-8,
    ):
        super().__init__()
        if not kernel_scales or any(scale <= 0 for scale in kernel_scales):
            raise ValueError("kernel_scales must contain positive values.")
        self.kernel_scales = tuple(float(scale) for scale in kernel_scales)
        self.epsilon = float(epsilon)
        self.last_bandwidth = None

    def _features(self, values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if values.dim() == 2:
            if mask is not None:
                raise ValueError("mask is only valid for [B, N, D] tensors.")
            return values
        if values.dim() != 3:
            raise ValueError("latents must be shaped [B, D] or [B, N, D].")
        if mask is None:
            mask = torch.ones(
                values.shape[:2],
                dtype=torch.bool,
                device=values.device,
            )
        if mask.shape != values.shape[:2]:
            raise ValueError(
                f"mask must have shape {tuple(values.shape[:2])}, got {tuple(mask.shape)}"
            )
        mask = mask.bool()
        if mask.any() and bool((mask == mask[:1]).all()):
            return values[mask].reshape(values.size(0), -1)
        denominator = mask.sum(dim=-1, keepdim=True).clamp_min(1)
        return (values * mask[..., None]).sum(dim=1) / denominator.to(values.dtype)

    def pairwise_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or y.dim() != 2:
            raise ValueError("pairwise_distance expects [B, D] tensors.")
        if x.size(1) != y.size(1):
            raise ValueError("latent feature dimensions must match.")
        squared = torch.cdist(x, y).square()
        return squared / max(x.size(1), 1)

    def _kernel(self, distance: torch.Tensor, bandwidth: torch.Tensor) -> torch.Tensor:
        scales = distance.new_tensor(self.kernel_scales)
        scale_shape = (len(self.kernel_scales),) + (1,) * distance.dim()
        widths = bandwidth * scales.reshape(scale_shape)
        return torch.exp(-distance.unsqueeze(0) / (2.0 * widths.square())).mean(dim=0)

    def forward(
        self,
        generated: torch.Tensor,
        posterior: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if generated.size(0) != posterior.size(0):
            raise ValueError("generated and posterior batches must have equal size.")
        if generated.size(0) < 2:
            self.last_bandwidth = None
            return generated.new_zeros(())

        x = self._features(generated, mask)
        y = self._features(posterior, mask)
        combined = torch.cat([x.detach(), y.detach()], dim=0)
        pairwise = self.pairwise_distance(combined, combined)
        positive = pairwise[pairwise > self.epsilon]
        bandwidth = (
            positive.median()
            if positive.numel() > 0
            else combined.new_tensor(1.0)
        ).clamp_min(self.epsilon)
        self.last_bandwidth = float(bandwidth)

        xx = self._kernel(self.pairwise_distance(x, x), bandwidth)
        yy = self._kernel(self.pairwise_distance(y, y), bandwidth)
        xy = self._kernel(self.pairwise_distance(x, y), bandwidth)
        return (xx.mean() + yy.mean() - 2.0 * xy.mean()).clamp_min(0.0)
