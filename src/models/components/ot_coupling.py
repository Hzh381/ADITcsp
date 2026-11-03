"""Minibatch optimal-transport coupling for flow-matching endpoints."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class MinibatchOTCoupler(nn.Module):
    """Pair Gaussian noise with posterior latents using Hungarian matching.

    The coupling changes only which ``z_0`` is paired with which ``z_1``
    during training. It does not alter either marginal distribution and has
    no trainable parameters.
    """

    def __init__(self, cost_type: str = "euclidean", epsilon: float = 1e-8):
        super().__init__()
        if cost_type != "euclidean":
            raise ValueError(f"Unsupported OT cost type: {cost_type!r}")
        self.cost_type = cost_type
        self.epsilon = float(epsilon)
        self.last_cost = None

    def pairwise_cost(
        self,
        z0: torch.Tensor,
        z1: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a normalized pairwise squared-distance matrix.

        Args:
            z0: Gaussian latents shaped ``[B, ...]``.
            z1: posterior latents with the same trailing shape.
            mask: optional valid-token mask ``[B, N]`` when inputs are
                ``[B, N, D]``.
        """
        if z0.shape != z1.shape:
            raise ValueError(f"z0 and z1 shapes differ: {z0.shape} vs {z1.shape}")
        if z0.shape[0] == 0:
            return z0.new_zeros((0, 0))

        if z0.dim() == 3:
            if mask is None:
                mask = torch.ones(
                    z0.shape[:2],
                    dtype=torch.bool,
                    device=z0.device,
                )
            if mask.shape != z0.shape[:2]:
                raise ValueError(
                    f"mask must have shape {tuple(z0.shape[:2])}, got {tuple(mask.shape)}"
                )
            mask = mask.bool()
            squared = (z0[:, None] - z1[None, :]).square()
            valid = mask[:, None, :] & mask[None, :, :]
            counts = valid.sum(dim=-1) * z0.size(-1)
            cost = (squared * valid[..., None]).sum(dim=(-1, -2))
            return cost / counts.clamp_min(1).to(cost.dtype)

        if mask is not None:
            raise ValueError("mask is only supported for [B, N, D] latents")
        dimension = z0[0].numel()
        cost = (z0[:, None] - z1[None, :]).square().sum(dim=-1)
        return cost / max(dimension, 1)

    @staticmethod
    def _solve(cost: torch.Tensor) -> torch.Tensor:
        """Return the column permutation for each row of ``cost``."""
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError as error:
            raise ImportError(
                "MinibatchOTCoupler requires scipy.optimize.linear_sum_assignment."
            ) from error
        rows, columns = linear_sum_assignment(cost.detach().cpu().numpy())
        if not np.array_equal(rows, np.arange(cost.size(0))):
            raise RuntimeError("Hungarian solver returned unexpected row order.")
        return torch.as_tensor(columns, dtype=torch.long, device=cost.device)

    @torch.no_grad()
    def forward(
        self,
        z0: torch.Tensor,
        z1: torch.Tensor,
        mask: torch.Tensor | None = None,
        group_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return ``z1`` reordered to minimize the minibatch transport cost."""
        if group_ids is None:
            group_ids = torch.zeros(z0.size(0), dtype=torch.long, device=z0.device)
        if group_ids.shape != (z0.size(0),):
            raise ValueError("group_ids must have one entry per latent.")

        matched = z1.clone()
        matched_cost_values = []
        for group_id in torch.unique(group_ids, sorted=True):
            indices = torch.nonzero(group_ids == group_id, as_tuple=False).flatten()
            if indices.numel() < 2:
                continue
            group_mask = None if mask is None else mask[indices]
            cost = self.pairwise_cost(z0[indices], z1[indices], group_mask)
            permutation = self._solve(cost)
            matched[indices] = z1[indices[permutation]]
            matched_cost_values.append(
                cost[torch.arange(indices.numel(), device=cost.device), permutation]
            )

        self.last_cost = (
            None
            if not matched_cost_values
            else float(torch.cat(matched_cost_values).mean())
        )
        return matched
