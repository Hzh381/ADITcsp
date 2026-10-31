"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import math
from typing import Dict

import torch
from torch import nn
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter

from src.models.components.hiloattention import (
    HiLoTransformerEncoder,
    HiLoTransformerEncoderLayer,
)
from src.models.components.crystal_batching import to_dense_periodic_inputs


def get_index_embedding(indices, emb_dim, max_len=2048):
    """Creates sine / cosine positional embeddings from a prespecified indices.

    Args:
        indices: offsets of size [..., num_tokens] of type integer
        emb_dim: dimension of the embeddings to create
        max_len: maximum length

    Returns:
        positional embedding of shape [..., num_tokens, emb_dim]
    """
    K = torch.arange(emb_dim // 2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / emb_dim))
    ).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / emb_dim))
    ).to(indices.device)
    pos_embedding = torch.cat([pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding


class TransformerEncoder(nn.Module):
    """Transformer encoder as part of standard Transformer-based VAEs.

    Args:
        max_num_elements: Maximum number of elements in the dataset
        d_model: Dimension of the model
        nhead: Number of attention heads
        dim_feedforward: Dimension of the feedforward network
        activation: Activation function to use
        dropout: Dropout rate
        norm_first: Whether to use pre-normalization in Transformer blocks
        bias: Whether to use bias
        num_layers: Number of layers
    """

    def __init__(
        self,
        max_num_elements=100,
        d_model: int = 1024,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        activation: str = "gelu",
        dropout: float = 0.0,
        norm_first: bool = True,
        bias: bool = True,
        num_layers: int = 6,
        attention_type: str = "standard",
        low_freq_heads: int = 2,
        hilo_cutoff: float = 5.0,
        hilo_max_shifts: int = 1,
        hilo_num_freqs: int = 4,
        hilo_edge_hidden: int = 64,
        hilo_global_hidden: int = 64,
        hilo_pool_type: str = "grid",
        hilo_pool_grid_size: int = 3,
        hilo_pool_chunk_size: int = 4,
    ):
        super().__init__()

        self.max_num_elements = max_num_elements
        self.d_model = d_model
        self.num_layers = num_layers
        self.attention_type = attention_type

        self.atom_type_embedder = nn.Embedding(max_num_elements, d_model)
        self.pos_embedder = nn.Sequential(
            nn.Linear(3, d_model, bias=False),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.frac_coords_embedder = nn.Sequential(
            nn.Linear(3, d_model, bias=False),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        activation_name = activation
        activation_module = {
            "gelu": nn.GELU(approximate="tanh"),
            "relu": nn.ReLU(),
        }[activation]
        if attention_type == "standard":
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    activation=activation_module,
                    dropout=dropout,
                    batch_first=True,
                    norm_first=norm_first,
                    bias=bias,
                ),
                norm=nn.LayerNorm(d_model),
                num_layers=num_layers,
            )
        elif attention_type == "hi_lo":
            encoder_layer = HiLoTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                activation=activation_name,
                dropout=dropout,
                norm_first=norm_first,
                bias=bias,
                low_freq_heads=low_freq_heads,
                cutoff=hilo_cutoff,
                max_shifts=hilo_max_shifts,
                num_freqs=hilo_num_freqs,
                edge_hidden=hilo_edge_hidden,
                global_hidden=hilo_global_hidden,
                pool_type=hilo_pool_type,
                pool_grid_size=hilo_pool_grid_size,
                pool_chunk_size=hilo_pool_chunk_size,
            )
            self.transformer = HiLoTransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
                norm=nn.LayerNorm(d_model),
            )
        else:
            raise ValueError(
                "attention_type must be 'standard' or 'hi_lo', "
                f"got {attention_type!r}"
            )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch: Data object with the following attributes:
                atom_types (torch.Tensor): Atomic numbers of atoms in the batch
                pos (torch.Tensor): Cartesian coordinates of atoms in the batch
                frac_coords (torch.Tensor): Fractional coordinates of atoms in the batch
                cell (torch.Tensor): Lattice vectors of the unit cell
                lattices (torch.Tensor): Lattice parameters of the unit cell (lengths and angles)
                lengths (torch.Tensor): Lengths of the lattice vectors
                angles (torch.Tensor): Angles between the lattice vectors
                num_atoms (torch.Tensor): Number of atoms in the batch
                batch (torch.Tensor): Batch index for each atom
        """
        x = self.atom_type_embedder(batch.atom_types)  # (n, d)
        x += self.pos_embedder(batch.pos)
        x += self.frac_coords_embedder(batch.frac_coords)

        # Positional embedding
        x += get_index_embedding(batch.token_idx, self.d_model)

        # Convert from PyG batch to dense batch with padding
        x, token_mask = to_dense_batch(x, batch.batch)

        # Transformer forward pass
        if getattr(self, "attention_type", "standard") == "hi_lo":
            frac_coords, cell, token_mask = to_dense_periodic_inputs(
                batch.frac_coords, batch.cell, batch.batch
            )
            x = self.transformer(x, frac_coords, cell, token_mask)
        else:
            x = self.transformer.forward(x, src_key_padding_mask=(~token_mask))
        x = x[token_mask]

        return {
            "x": x,
            "num_atoms": batch.num_atoms,
            "batch": batch.batch,
            "token_idx": batch.token_idx,
            "frac_coords": batch.frac_coords,
            "cell": batch.cell,
        }
