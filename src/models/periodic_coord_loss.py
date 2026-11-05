"""Periodic atom-coordinate supervision for decoder-aware flow matching.

The matching stage (translation search plus Hungarian assignment) is
discrete and runs under ``torch.no_grad()``.  The selected assignment is
then fixed while the periodic squared distance is differentiated with
respect to the predicted fractional coordinates.
"""

from __future__ import annotations

import numpy as np
import torch


def _shift_grid(
    max_shifts: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    axis = torch.arange(
        -max_shifts,
        max_shifts + 1,
        device=device,
        dtype=dtype,
    )
    return torch.stack(
        torch.meshgrid(axis, axis, axis, indexing="ij"),
        dim=-1,
    ).reshape(-1, 3)


def _wrapped_difference(pred_frac: torch.Tensor, target_frac: torch.Tensor) -> torch.Tensor:
    difference = pred_frac[:, None, :] - target_frac[None, :, :]
    return difference - torch.round(difference)


def periodic_cost_matrix(
    pred_frac: torch.Tensor,
    target_frac: torch.Tensor,
    lattice: torch.Tensor,
    pred_types: torch.Tensor | None = None,
    target_types: torch.Tensor | None = None,
    max_shifts: int = 1,
) -> torch.Tensor:
    """Return the pairwise periodic squared-distance cost matrix.

    Args:
        pred_frac: ``[P, 3]`` predicted fractional coordinates.
        target_frac: ``[T, 3]`` target fractional coordinates.
        lattice: ``[3, 3]`` target lattice matrix (rows are lattice vectors).
        pred_types/target_types: optional integer element labels. A pair is
            forbidden when the labels differ.
    """
    if pred_frac.shape != (pred_frac.size(0), 3):
        raise ValueError("pred_frac must be [P, 3].")
    if target_frac.shape != (target_frac.size(0), 3):
        raise ValueError("target_frac must be [T, 3].")
    if lattice.shape != (3, 3):
        raise ValueError("lattice must be [3, 3].")
    if pred_frac.size(0) != target_frac.size(0):
        raise ValueError("The coordinate loss requires equal atom counts.")

    delta = _wrapped_difference(pred_frac, target_frac)
    best = None
    for shift in _shift_grid(
        max_shifts,
        device=pred_frac.device,
        dtype=pred_frac.dtype,
    ):
        cart = torch.einsum("pti,ij->ptj", delta + shift, lattice)
        distance_squared = cart.square().sum(dim=-1)
        best = (
            distance_squared
            if best is None
            else torch.minimum(best, distance_squared)
        )

    if pred_types is not None or target_types is not None:
        if pred_types is None or target_types is None:
            raise ValueError("pred_types and target_types must be provided together.")
        compatible = pred_types[:, None] == target_types[None, :]
        penalty = best.detach().max().clamp_min(1.0) * 100.0
        best = best.masked_fill(
            ~compatible,
            penalty,
        )
    return best


def _candidate_shifts(
    pred_frac: torch.Tensor,
    target_frac: torch.Tensor,
    max_candidates: int,
) -> list[torch.Tensor]:
    """Generate deterministic translation candidates without a full N^2 sweep."""
    # ``shift`` is added to the prediction, so the candidate is target - pred.
    difference = _wrapped_difference(target_frac, pred_frac)
    flat = difference.reshape(-1, 3)
    candidates = [
        pred_frac.new_zeros(3),
        flat.mean(dim=0),
        flat.median(dim=0).values,
    ]
    quantized = torch.round(flat / 1e-2)
    unique_shifts, counts = torch.unique(quantized, dim=0, return_counts=True)
    top_count = min(3, counts.numel())
    for index in torch.topk(counts, k=top_count).indices:
        candidates.append(unique_shifts[index].to(flat.dtype) * 1e-2)
    num_atoms = min(pred_frac.size(0), target_frac.size(0))
    for index in range(num_atoms):
        candidates.append(difference[index % target_frac.size(0), index])
        candidates.append(difference[index, index % pred_frac.size(0)])

    if flat.size(0) <= max_candidates:
        candidates.extend(flat.unbind(dim=0))

    unique = []
    for candidate in candidates:
        if not any(
            torch.allclose(candidate, existing, atol=1e-6, rtol=0.0)
            for existing in unique
        ):
            unique.append(candidate)
        if len(unique) >= max_candidates:
            break
    return unique


def periodic_hungarian_match(
    pred_frac: torch.Tensor,
    target_frac: torch.Tensor,
    lattice: torch.Tensor,
    pred_types: torch.Tensor | None = None,
    target_types: torch.Tensor | None = None,
    max_shifts: int = 1,
    max_shift_candidates: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, float] | None:
    """Search a translation and return a fixed element-constrained assignment."""
    from scipy.optimize import linear_sum_assignment

    if max_shift_candidates < 1:
        raise ValueError("max_shift_candidates must be positive.")

    def evaluate(shift):
        cost = periodic_cost_matrix(
            pred_frac.detach() + shift,
            target_frac.detach(),
            lattice.detach(),
            pred_types,
            target_types,
            max_shifts=max_shifts,
        )
        cost_np = cost.detach().cpu().numpy()
        if not np.isfinite(cost_np).all():
            return None
        rows, columns = linear_sum_assignment(cost_np)
        if not np.array_equal(rows, np.arange(pred_frac.size(0))):
            return None
        if pred_types is not None:
            compatible = pred_types[:, None] == target_types[None, :]
            row_index = torch.as_tensor(
                rows, dtype=torch.long, device=compatible.device
            )
            column_index = torch.as_tensor(
                columns, dtype=torch.long, device=compatible.device
            )
            if not bool(compatible[row_index, column_index].all()):
                return None
        value = float(cost_np[rows, columns].sum())
        return (
            shift.detach(),
            torch.as_tensor(columns, dtype=torch.long, device=pred_frac.device),
            value,
        )

    best = None
    for shift in _candidate_shifts(
        pred_frac.detach(),
        target_frac.detach(),
        max_shift_candidates,
    ):
        candidate = evaluate(shift)
        if candidate is not None and (best is None or candidate[2] < best[2]):
            best = candidate

    # Refine the translation from the currently matched pairs. This recovers
    # translations that are not present among the capped anchor candidates.
    for _ in range(1):
        if best is None:
            break
        matched_difference = (
            target_frac.detach()[best[1]] - pred_frac.detach()
        )
        matched_difference = matched_difference - torch.round(matched_difference)
        refined_shift = matched_difference.mean(dim=0)
        candidate = evaluate(refined_shift)
        if candidate is None or candidate[2] >= best[2] - 1e-9:
            break
        best = candidate
    return best


def _matched_periodic_error(
    pred_frac: torch.Tensor,
    target_frac: torch.Tensor,
    lattice: torch.Tensor,
    shift: torch.Tensor,
    assignment: torch.Tensor,
    max_shifts: int = 1,
) -> torch.Tensor:
    delta = pred_frac + shift - target_frac[assignment]
    best = None
    for periodic_shift in _shift_grid(
        max_shifts,
        device=pred_frac.device,
        dtype=pred_frac.dtype,
    ):
        cart = torch.einsum("ni,ij->nj", delta + periodic_shift, lattice)
        distance_squared = cart.square().sum(dim=-1)
        best = (
            distance_squared
            if best is None
            else torch.minimum(best, distance_squared)
        )
    return best.mean()


def periodic_coord_loss(
    pred_frac: torch.Tensor,
    target_frac: torch.Tensor,
    lattice: torch.Tensor,
    pred_types: torch.Tensor | None = None,
    target_types: torch.Tensor | None = None,
    atom_mask: torch.Tensor | None = None,
    max_shifts: int = 1,
    max_shift_candidates: int = 4,
) -> torch.Tensor:
    """Return the mean matched periodic squared-coordinate error.

    Args:
        pred_frac/target_frac: dense ``[B, N, 3]`` fractional coordinates.
        lattice: dense ``[B, 3, 3]`` target lattice matrices.  The target
            lattice is used as the distance metric so ``L_cell`` and
            ``L_coord`` do not double-penalize predicted cell scale.
        pred_types/target_types: optional dense ``[B, N]`` element labels.
        atom_mask: dense ``[B, N]`` valid-atom mask.
    """
    if pred_frac.shape != target_frac.shape:
        raise ValueError("pred_frac and target_frac must have the same shape.")
    if lattice.shape != (pred_frac.size(0), 3, 3):
        raise ValueError("lattice must have shape [B, 3, 3].")
    if pred_types is not None and target_types is not None:
        if pred_types.shape != target_types.shape:
            raise ValueError("pred_types and target_types must have the same shape.")
    elif pred_types is not None or target_types is not None:
        raise ValueError("pred_types and target_types must be provided together.")

    if atom_mask is None:
        atom_mask = torch.ones(
            pred_frac.shape[:2],
            dtype=torch.bool,
            device=pred_frac.device,
        )
    if atom_mask.shape != pred_frac.shape[:2]:
        raise ValueError("atom_mask must have shape [B, N].")

    losses = []
    for batch_index in range(pred_frac.size(0)):
        indices = torch.nonzero(atom_mask[batch_index], as_tuple=False).flatten()
        if indices.numel() == 0:
            continue
        pred = pred_frac[batch_index, indices]
        target = target_frac[batch_index, indices]
        pred_type = (
            pred_types[batch_index, indices] if pred_types is not None else None
        )
        target_type = (
            target_types[batch_index, indices] if target_types is not None else None
        )
        match = periodic_hungarian_match(
            pred,
            target,
            lattice[batch_index],
            pred_type,
            target_type,
            max_shifts=max_shifts,
            max_shift_candidates=max_shift_candidates,
        )
        if match is None:
            continue
        shift, assignment, _ = match
        losses.append(
            _matched_periodic_error(
                pred,
                target,
                lattice[batch_index],
                shift,
                assignment,
                max_shifts=max_shifts,
            )
        )

    if not losses:
        return pred_frac.new_zeros(())
    return torch.stack(losses).mean()
