"""Differentiable cell and periodic structural losses for decoder-aware FM.

The helpers operate on batched, padded tensors:

* ``lengths``  ``[B, 3]`` in Angstrom.
* ``angles``   ``[B, 3]`` in radians.
* ``frac``     ``[B, N, 3]`` fractional coordinates.
* ``token_mask`` ``[B, N]`` with ``True`` for real atoms.

All functions keep the computational graph so gradients can flow from the
frozen VAE decoder back into the latent denoiser.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from src.models.periodic_coord_loss import periodic_coord_loss


def build_cell_matrices(lengths: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Build lattice matrices (vectors as rows) from lengths and angles.

    The orientation is one valid convention for the six lattice parameters.
    StructureMatcher compares invariant geometry, so the absolute orientation
    does not matter.
    """
    eps = 1e-3
    lengths = lengths.clamp_min(eps)
    angles = angles.clamp(min=0.1, max=math.pi - 0.1)
    a, b, c = lengths.unbind(dim=-1)
    alpha, beta, gamma = angles.unbind(dim=-1)
    cos_alpha = torch.cos(alpha)
    cos_beta = torch.cos(beta)
    cos_gamma = torch.cos(gamma)
    sin_gamma = torch.sin(gamma).clamp_min(0.1)

    a_vec = torch.stack([a, torch.zeros_like(a), torch.zeros_like(a)], dim=-1)
    b_vec = torch.stack(
        [b * cos_gamma, b * sin_gamma, torch.zeros_like(b)], dim=-1
    )
    c_x = c * cos_beta
    c_y = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    c_z = torch.sqrt((c * c - c_x * c_x - c_y * c_y).clamp_min(eps * eps))
    c_vec = torch.stack([c_x, c_y, c_z], dim=-1)
    return torch.stack([a_vec, b_vec, c_vec], dim=-2)


def periodic_pair_distances(
    frac_coords: torch.Tensor,
    cell: torch.Tensor,
    max_shifts: int = 1,
) -> torch.Tensor:
    """Return ``[B, N, N]`` periodic minimum-image distances.

    The wrapped difference is tested against neighbouring integer shifts so
    triclinic cells do not rely on a component-wise [-0.5, 0.5] shortcut.
    """
    delta = frac_coords[:, :, None, :] - frac_coords[:, None, :, :]
    delta = delta - torch.round(delta)
    axis = torch.arange(
        -max_shifts,
        max_shifts + 1,
        device=frac_coords.device,
        dtype=frac_coords.dtype,
    )
    shifts = torch.stack(
        torch.meshgrid(axis, axis, axis, indexing="ij"),
        dim=-1,
    ).reshape(-1, 3)
    best = None
    for shift in shifts:
        cart = torch.einsum("bnki,bij->bnkj", delta + shift, cell)
        distance_squared = cart.square().sum(dim=-1)
        best = (
            distance_squared
            if best is None
            else torch.minimum(best, distance_squared)
        )
    # A pair of predicted atoms can coincide exactly; raw sqrt(0) has an
    # infinite derivative and poisons the whole DiT update with NaN.
    return torch.sqrt(best.clamp_min(0.0) + 1e-6)


def periodic_pair_mask(token_mask: torch.Tensor) -> torch.Tensor:
    """``[B, N, N]`` mask selecting unique valid pairs ``i < j``."""
    n = token_mask.size(1)
    index = torch.arange(n, device=token_mask.device)
    upper = index[:, None] < index[None, :]
    valid = token_mask[:, :, None] & token_mask[:, None, :]
    return valid & upper.unsqueeze(0)


def compute_cell_loss(
    pred_lengths: torch.Tensor,
    pred_angles: torch.Tensor,
    target_lengths: torch.Tensor,
    target_angles: torch.Tensor,
    graph_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Length log-ratio and lattice-angle cosine error.

    Predicted lengths come from an unconstrained linear head, so a small
    positivity penalty keeps them inside the domain where the log is defined.
    """
    eps = 1e-3
    pred_pos = pred_lengths.clamp_min(eps)
    log_error = (
        torch.log(pred_pos) - torch.log(target_lengths.clamp_min(eps))
    ).abs()
    cos_error = (torch.cos(pred_angles) - torch.cos(target_angles)).abs()
    negative_penalty = F.relu(-pred_lengths)
    angle_penalty = F.relu(-pred_angles) + F.relu(pred_angles - math.pi)
    per_graph = (
        log_error + cos_error + negative_penalty + angle_penalty
    ).sum(dim=-1)

    if graph_mask is None:
        return per_graph.mean()
    graph_mask = graph_mask.to(device=per_graph.device, dtype=per_graph.dtype)
    return (per_graph * graph_mask).sum() / graph_mask.sum().clamp_min(1.0)


def compute_periodic_loss(
    pred_frac_coords: torch.Tensor,
    pred_lengths: torch.Tensor,
    pred_angles: torch.Tensor,
    target_frac_coords: torch.Tensor,
    target_lengths: torch.Tensor,
    target_angles: torch.Tensor,
    token_mask: torch.Tensor,
    rho: torch.Tensor | None = None,
    sigma: float = 0.5,
) -> torch.Tensor:
    """Smooth radial-distribution error using periodic minimum-image distances."""
    if rho is None:
        rho = torch.linspace(
            float(sigma),
            6.0,
            32,
            device=pred_frac_coords.device,
            dtype=pred_frac_coords.dtype,
        )
    pair_mask = periodic_pair_mask(token_mask)
    counts = token_mask.sum(dim=-1).clamp_min(1).to(pred_frac_coords.dtype)

    pred_cell = build_cell_matrices(pred_lengths, pred_angles)
    target_cell = build_cell_matrices(target_lengths, target_angles)
    pred_distances = periodic_pair_distances(pred_frac_coords, pred_cell)
    target_distances = periodic_pair_distances(target_frac_coords, target_cell)

    pred_hist = []
    target_hist = []
    pair_mask_f = pair_mask.to(pred_frac_coords.dtype)
    sigma_tensor = pred_frac_coords.new_tensor(float(sigma))
    for centre in rho:
        pred_kernel = torch.exp(
            -0.5 * ((pred_distances - centre) / sigma_tensor).square()
        )
        target_kernel = torch.exp(
            -0.5 * ((target_distances - centre) / sigma_tensor).square()
        )
        pred_hist.append((pred_kernel * pair_mask_f).sum(dim=(1, 2)) / counts)
        target_hist.append((target_kernel * pair_mask_f).sum(dim=(1, 2)) / counts)

    pred_hist = torch.stack(pred_hist, dim=-1)
    target_hist = torch.stack(target_hist, dim=-1)
    return (pred_hist - target_hist).abs().sum(dim=-1).mean()


def compute_struct_loss(
    pred_frac_coords: torch.Tensor,
    pred_lengths: torch.Tensor,
    pred_angles: torch.Tensor,
    target_frac_coords: torch.Tensor,
    target_lengths: torch.Tensor,
    target_angles: torch.Tensor,
    token_mask: torch.Tensor,
    pred_types: torch.Tensor | None = None,
    target_types: torch.Tensor | None = None,
    include_coord: bool = False,
    coord_max_shifts: int = 1,
    coord_shift_candidates: int = 4,
    coord_batch_size: int = 0,
    rho: torch.Tensor | None = None,
    sigma: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Return cell, periodic and optional atom-coordinate terms."""
    cell_loss = compute_cell_loss(
        pred_lengths,
        pred_angles,
        target_lengths,
        target_angles,
    )
    geo_loss = compute_periodic_loss(
        pred_frac_coords,
        pred_lengths,
        pred_angles,
        target_frac_coords,
        target_lengths,
        target_angles,
        token_mask,
        rho=rho,
        sigma=sigma,
    )
    if include_coord:
        target_cell = build_cell_matrices(target_lengths, target_angles)
        if coord_batch_size > 0 and pred_frac_coords.size(0) > coord_batch_size:
            selected = torch.randperm(
                pred_frac_coords.size(0),
                device=pred_frac_coords.device,
            )[:coord_batch_size]
        else:
            selected = slice(None)
        selected_pred_types = (
            pred_types[selected] if pred_types is not None else None
        )
        selected_target_types = (
            target_types[selected] if target_types is not None else None
        )
        coord_loss = periodic_coord_loss(
            pred_frac_coords[selected],
            target_frac_coords[selected],
            target_cell[selected],
            pred_types=selected_pred_types,
            target_types=selected_target_types,
            atom_mask=token_mask[selected],
            max_shifts=coord_max_shifts,
            max_shift_candidates=coord_shift_candidates,
        )
    else:
        coord_loss = pred_frac_coords.new_zeros(())
    return {
        "cell_loss": cell_loss,
        "geo_loss": geo_loss,
        "coord_loss": coord_loss,
    }
