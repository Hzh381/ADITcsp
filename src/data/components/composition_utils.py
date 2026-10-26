"""Helpers for composition vectors used by the CSP-conditioned models."""

import torch
import torch.nn.functional as F


def composition_from_batch(batch, num_elements: int = 100) -> torch.Tensor:
    """Return per-graph element counts ``[B, num_elements]``.

    The input is a PyG ``Batch`` with ``atom_types`` and ``batch`` fields.
    """
    atom_types = batch.atom_types.clamp(min=0, max=num_elements - 1)
    if hasattr(batch, "batch"):
        batch_index = batch.batch
        num_graphs = int(batch_index.max().item()) + 1
    else:
        batch_index = torch.zeros_like(atom_types)
        num_graphs = 1
    composition = torch.zeros(
        num_graphs, num_elements, device=atom_types.device, dtype=torch.float32
    )
    composition.index_add_(
        0,
        batch_index,
        F.one_hot(atom_types.long(), num_classes=num_elements).float(),
    )
    return composition


def composition_from_atom_types(
    atom_types: torch.Tensor,
    batch_index: torch.Tensor | None = None,
    num_graphs: int = 1,
    num_elements: int = 100,
) -> torch.Tensor:
    """Composition vector from a flat atom-type tensor."""
    atom_types = atom_types.clamp(min=0, max=num_elements - 1).long()
    if batch_index is None:
        batch_index = torch.zeros_like(atom_types)
    composition = torch.zeros(
        num_graphs, num_elements, device=atom_types.device, dtype=torch.float32
    )
    composition.index_add_(
        0,
        batch_index,
        F.one_hot(atom_types, num_classes=num_elements).float(),
    )
    return composition
