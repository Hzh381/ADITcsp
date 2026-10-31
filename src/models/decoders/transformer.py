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


class TransformerDecoder(nn.Module):
    """Transformer decoder as part of pure Transformer-based VAEs.

    See src/models/encoders/transformer.py for documentation.
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

        self.atom_types_head = nn.Linear(d_model, max_num_elements, bias=True)
        self.pos_head = nn.Linear(d_model, 3, bias=False)
        self.frac_coords_head = nn.Linear(d_model, 3, bias=False)
        self.lattice_head = nn.Linear(d_model, 6, bias=False)

    def forward(self, encoded_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            encoded_batch: Dict with the following attributes:
                x (torch.Tensor): Encoded batch of atomic environments
                num_atoms (torch.Tensor): Number of atoms in each molecular environment
                batch (torch.Tensor): Batch index for each atom
                token_idx (torch.Tensor): Token index for each atom
        """
        x = encoded_batch["x"]

        composition_embed = encoded_batch.get("composition_embed")
        if composition_embed is not None:
            x = x + composition_embed[encoded_batch["batch"]]

        # Positional embedding
        x += get_index_embedding(encoded_batch["token_idx"], self.d_model)

        # Convert from PyG batch to dense batch with padding
        x, token_mask = to_dense_batch(x, encoded_batch["batch"])

        # Transformer forward pass
        if getattr(self, "attention_type", "standard") == "hi_lo":
            frac_coords, cell, token_mask = to_dense_periodic_inputs(
                encoded_batch["frac_coords"],
                encoded_batch["cell"],
                encoded_batch["batch"],
            )
            x = self.transformer(x, frac_coords, cell, token_mask)
        else:
            x = self.transformer.forward(x, src_key_padding_mask=(~token_mask))
        x = x[token_mask]

        # Global pooling: (n, d) -> (bsz, d)
        x_global = scatter(x, encoded_batch["batch"], dim=0, reduce="mean")

        # Atomic type prediction head
        atom_types_out = self.atom_types_head(x)

        # Lattice lengths and angles prediction head
        lattices_out = self.lattice_head(x_global)

        # Fractional coordinates prediction head
        frac_coords_out = self.frac_coords_head(x)

        # Cartesian coordinates prediction head
        pos_out = self.pos_head(x)

        return {
            "atom_types": atom_types_out,
            "lattices": lattices_out,
            "lengths": lattices_out[:, :3],
            "angles": lattices_out[:, 3:],
            "frac_coords": frac_coords_out,
            "pos": pos_out,
        }
