"""Helpers for converting flattened PyG crystal batches to dense geometry."""

import torch
from torch_geometric.utils import to_dense_batch


def to_dense_periodic_inputs(frac_coords, cell, batch_index):
    """Return dense fractional coordinates and one cell matrix per structure.

    PyG may collate ``cell`` as either a graph-level ``[B, 3, 3]`` tensor or a
    per-node ``[N, 3, 3]`` / ``[N, 1, 3, 3]`` tensor depending on how the
    source ``Data`` object was constructed. HiLo accepts both layouts, so this
    helper normalizes them without changing the token mask or atom order.
    """
    frac_dense, token_mask = to_dense_batch(frac_coords, batch_index)
    num_nodes = batch_index.numel()
    num_graphs = int(batch_index.max().item()) + 1 if num_nodes else 0

    if cell.dim() == 2:
        if cell.shape != (3, 3):
            raise ValueError("cell must have shape (3, 3)")
        dense_cell = cell.unsqueeze(0).expand(num_graphs, 3, 3)
    elif cell.dim() == 3:
        if cell.size(0) == num_graphs:
            dense_cell = cell
        elif cell.size(0) == num_nodes:
            dense_cell, _ = to_dense_batch(cell, batch_index)
        else:
            raise ValueError(
                "cell has neither one matrix per graph nor one per token"
            )
    elif cell.dim() == 4:
        if cell.size(0) != num_nodes:
            if cell.size(0) == num_graphs and cell.size(1) == 1:
                dense_cell = cell[:, 0]
            else:
                raise ValueError("per-token cell must have shape (N, 3, 3)")
        else:
            dense_cell, _ = to_dense_batch(cell, batch_index)
    elif cell.dim() == 5 and cell.size(0) == num_graphs and cell.size(2) == 1:
        dense_cell = cell.squeeze(2)
    else:
        raise ValueError(
            "cell must be (3, 3), (B, 3, 3), (N, 3, 3), "
            "(N, 1, 3, 3), or a dense per-token equivalent"
        )

    if dense_cell.dim() == 5 and dense_cell.size(2) == 1:
        dense_cell = dense_cell.squeeze(2)
    return frac_dense, dense_cell, token_mask
