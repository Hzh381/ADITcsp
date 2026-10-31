"""Dual-frequency (Hi-Fi / Lo-Fi) attention for periodic crystal tokens.

This reference copy implements the attention block from the diagram. The
canonical module used by ADITcsp lives at
``src/models/components/hiloattention.py``. It is intentionally standalone
(torch only, no ``torch_geometric`` / ``torch_scatter`` dependency) and works
on dense padded tensors, which is the layout passed by the integrated VAE.

Equation map (``L = [a, b, c]`` has lattice vectors as rows, i.e.
``cart = frac @ L`` as in ``src/data/components/preprocessing_utils.py``):

High-frequency branch (local geometry, ``(1 - alpha) N_h`` heads)
    df_ij       = f_j - f_i
    n*_ij       = argmin_{n in Z^3} || L (df_ij + n) ||_2
    df*_ij      = df_ij + n*_ij
    d_ij        = || L df*_ij ||_2
    N(i)        = {j != i : d_ij < r_c}
    dbar_ij     = d_ij / r_c
    e_ij        = gamma_df(df*_ij) (+) gamma_d(dbar_ij)
    B_hij       = MLP_edge,h(e_ij)
    Attn_h      = softmax_j(q_hi^T k_hj / sqrt(d_h) + B_hij + M_ij) v_hj

Low-frequency branch (global lattice context, ``alpha N_h`` heads)
    psi(L, N)   = [log(a / N^(1/3)), log(b / N^(1/3)), log(c / N^(1/3)),
                   cos alpha, cos beta, cos gamma]
    w_h         = MLP_global,h(psi)          in R^{d_h}
    g_h         = sigmoid(MLP_g,h(psi))      in (0, 1)
    q_hI        = W^q_h x_I                  (fine tokens)
    k_hJ        = W^k_h x_tilde_J            (average-pooled tokens)
    B_hIJ       = (g_h / sqrt(d_b)) (q_hI (*) w_h)^T k_hJ
    Attn_h      = softmax_J(Q_h K_h^T / sqrt(d_h) + B_hJ) V_h

The head budget follows the figure: ``nhead`` heads are split into
``n_hi = nhead - low_freq_heads`` high-frequency heads and ``low_freq_heads``
low-frequency heads. Both branches emit ``d_h``-sized head outputs, which are
concatenated and projected once; this keeps the residual path
shape-preserving. (The diagram draws one projection per branch, but that
reading does not type-check against the ``Concat`` + residual ``(+)``
operation; the head-split reading matches the ``(1 - alpha) N_h`` /
``alpha N_h`` notation in the figure and is what is implemented here.)

Design choices that the figure leaves open:

* ``x_tilde`` is obtained by average pooling. ``pool_type="grid"`` bins atoms
  into a ``g x g x g`` lattice-aligned grid and averages the features of each
  occupied bin, i.e. the 3-D analogue of the 2-D average pooling in the
  diagram; ``"chunk"`` averages consecutive token blocks; ``"none"`` uses the
  fine tokens directly (plain global attention).
* Lo-Fi queries come from the fine tokens while keys/values come from the
  pooled tokens, exactly as written in ``q_hI = W^q_h x_I`` and
  ``k_hJ = W^k_h x_tilde_J``. This keeps the output length equal to the atom
  count, which the residual connection requires.
* Atoms without a neighbour inside ``r_c`` attend to themselves so that the
  softmax is always well defined; padded tokens always receive exactly zero.
* The output layers of the edge MLPs and of ``MLP_global`` are
  zero-initialised, so a freshly built module starts from standard scaled
  dot-product attention (within the Hi-Fi neighbourhood mask) and grows the
  Hi-Fi / Lo-Fi biases during training. ``g_h`` therefore starts at
  ``sigmoid(0) = 0.5`` and only becomes data-dependent once ``w_h`` has moved.

Usage inside the integrated VAE encoder::

    stack = HiLoTransformerEncoder(
        HiLoTransformerEncoderLayer(
            d_model=256, nhead=8, low_freq_heads=2, cutoff=5.0
        ),
        num_layers=6,
    )
    # x: (B, N, d) dense, zero padded; frac: (B, N, 3); cell: (B, 3, 3)
    # mask: (B, N) bool (True for real atoms)
    x = stack(x, frac, cell, mask)

``cell`` may also be given per atom as ``(B, N, 3, 3)``, e.g. when a dataset
repeats the lattice for every node and the caller keeps that layout through
``to_dense_batch``.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

__all__ = [
    "HiLoAttention",
    "HiLoGeometry",
    "HiLoTransformerEncoder",
    "HiLoTransformerEncoderLayer",
    "build_hi_lo_geometry",
    "edge_feature_dim",
    "integer_shift_grid",
    "lattice_descriptor",
    "minimum_image_displacements",
]

PoolType = Literal["grid", "chunk", "none"]
_POOL_TYPES = ("grid", "chunk", "none")


# --------------------------------------------------------------------------- #
# Geometry: minimum image convention, edge features, lattice descriptor
# --------------------------------------------------------------------------- #
def edge_feature_dim(num_freqs: int) -> int:
    """Width of ``e_ij = gamma_df(df*) (+) gamma_d(dbar)``.

    ``gamma_df`` uses ``num_freqs`` Fourier bands per fractional component
    (6 * num_freqs values) and ``gamma_d`` uses ``num_freqs`` bands on the
    normalised distance (2 * num_freqs values).
    """
    if num_freqs < 1:
        raise ValueError("num_freqs must be >= 1")
    return 8 * num_freqs


def integer_shift_grid(
    max_shifts: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.long,
) -> torch.Tensor:
    """All integer translations in ``[-max_shifts, max_shifts]^3``.

    Returns:
        ``(2 * max_shifts + 1) ** 3, 3`` tensor of lattice translations.
    """
    if max_shifts < 0:
        raise ValueError("max_shifts must be non-negative")
    axis = torch.arange(-max_shifts, max_shifts + 1, device=device, dtype=dtype)
    grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1)
    return grid.reshape(-1, 3)


def minimum_image_displacements(
    frac_coords: torch.Tensor,
    cell: torch.Tensor,
    max_shifts: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Periodic minimum-image displacements and distances.

    Implements
    ``n*_ij = argmin_{n in Z^3} || L (df_ij + n) ||_2`` by enumerating integer
    shifts and using the metric tensor ``G = L L^T`` so the Cartesian distance
    is ``sqrt(df G df^T)`` without materialising Cartesian coordinates.

    Fractional coordinates are wrapped into ``[0, 1)`` first, which makes the
    search invariant to integer lattice translations; 27 shifts are then
    sufficient for any wrapped difference. Crystallographic cells are far from
    degenerate, so a fixed ±1 search is safe; ``max_shifts`` can be raised for
    pathological cells.

    Args:
        frac_coords: ``(B, N, 3)`` fractional coordinates.
        cell: ``(B, 3, 3)`` lattice matrices with ``a, b, c`` as rows.
        max_shifts: integer search radius.

    Returns:
        ``df_star`` of shape ``(B, N, N, 3)`` with ``df_star[b, i, j] = f_j -
        f_i + n*_ij``, and ``distance`` of shape ``(B, N, N)`` with the
        corresponding Cartesian distances.
    """
    if frac_coords.dim() != 3 or frac_coords.size(-1) != 3:
        raise ValueError("frac_coords must have shape (B, N, 3)")
    if (
        cell.dim() != 3
        or cell.shape[1:] != (3, 3)
        or cell.size(0) != frac_coords.size(0)
    ):
        raise ValueError("cell must have shape (B, 3, 3) matching frac_coords")

    frac = frac_coords % 1.0
    # delta[b, i, j] = f_j - f_i
    delta = frac.unsqueeze(1) - frac.unsqueeze(2)
    metric = cell @ cell.transpose(-1, -2)
    shifts = integer_shift_grid(
        max_shifts, device=frac.device, dtype=frac.dtype
    )

    best_dist2 = None
    best_disp = None
    for shift in shifts:
        candidate = delta + shift  # (B, N, N, 3)
        dist2 = torch.einsum("bnki,bij,bnkj->bnk", candidate, metric, candidate)
        if best_dist2 is None:
            best_dist2, best_disp = dist2, candidate
            continue
        closer = dist2 < best_dist2
        best_dist2 = torch.where(closer, dist2, best_dist2)
        best_disp = torch.where(closer.unsqueeze(-1), candidate, best_disp)

    return best_disp, best_dist2.clamp_min(0.0).sqrt()


def _fractional_fourier(disp_frac: torch.Tensor, num_freqs: int) -> torch.Tensor:
    """``gamma_df``: sin/cos bands of the fractional displacement, ``6F`` wide."""
    bands = torch.arange(
        1, num_freqs + 1, device=disp_frac.device, dtype=disp_frac.dtype
    )
    angles = 2.0 * math.pi * disp_frac.unsqueeze(-1) * bands
    return torch.cat((angles.sin(), angles.cos()), dim=-1).flatten(-2)


def _distance_fourier(
    normalized_distance: torch.Tensor, num_freqs: int
) -> torch.Tensor:
    """``gamma_d``: sin/cos bands of ``d_ij / r_c``, ``2F`` wide."""
    bands = torch.arange(
        1, num_freqs + 1, device=normalized_distance.device,
        dtype=normalized_distance.dtype,
    )
    angles = math.pi * normalized_distance.unsqueeze(-1) * bands
    return torch.cat((angles.sin(), angles.cos()), dim=-1)


def _normalize_cell(cell: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Return ``cell`` as ``(B, 3, 3)`` from (3, 3), (1, 3, 3) or (B, N, 3, 3)."""
    if not torch.is_tensor(cell):
        raise TypeError("cell must be a torch.Tensor")
    if cell.dim() == 4:
        if cell.size(0) != batch_size or cell.shape[2:] != (3, 3):
            raise ValueError("per-atom cell must have shape (B, N, 3, 3)")
        return cell[:, 0]
    if cell.dim() == 3:
        if cell.shape[1:] != (3, 3):
            raise ValueError("cell must have shape (B, 3, 3)")
        if cell.size(0) == 1 and batch_size != 1:
            return cell.expand(batch_size, 3, 3)
        if cell.size(0) != batch_size:
            raise ValueError(
                "expected one (3, 3) lattice matrix per structure; per-atom "
                "cells must be reduced to one matrix per structure first"
            )
        return cell
    if cell.dim() == 2:
        if cell.shape != (3, 3):
            raise ValueError("cell must have shape (3, 3)")
        return cell.unsqueeze(0).expand(batch_size, 3, 3)
    raise ValueError("cell must have shape (3, 3), (B, 3, 3) or (B, N, 3, 3)")


def lattice_descriptor(
    cell: torch.Tensor, num_atoms: torch.Tensor | None = None
) -> torch.Tensor:
    """``psi(L, N)`` from the figure, shape ``(B, 6)``.

    ``a, b, c`` are the row norms of ``L``; ``alpha = angle(b, c)``,
    ``beta = angle(a, c)``, ``gamma = angle(a, b)``; ``N`` is the atom count,
    used to normalise the lengths per atom exactly like ``lengths_scaled`` in
    the ADITcsp data pipeline.
    """
    if cell.dim() != 3 or cell.shape[1:] != (3, 3):
        raise ValueError("cell must have shape (B, 3, 3)")
    lengths = torch.linalg.vector_norm(cell, dim=-1).clamp_min(1e-6)
    if num_atoms is None:
        scale = cell.new_ones(cell.size(0))
    else:
        scale = (
            num_atoms.to(device=cell.device, dtype=cell.dtype)
            .clamp_min(1.0)
            .pow(1.0 / 3.0)
        )

    log_lengths = torch.log(lengths / scale.unsqueeze(-1))

    def cosine(
        u: torch.Tensor, v: torch.Tensor, nu: torch.Tensor, nv: torch.Tensor
    ) -> torch.Tensor:
        return ((u * v).sum(dim=-1) / (nu * nv)).clamp(-1.0, 1.0)

    cos_alpha = cosine(cell[:, 1], cell[:, 2], lengths[:, 1], lengths[:, 2])
    cos_beta = cosine(cell[:, 0], cell[:, 2], lengths[:, 0], lengths[:, 2])
    cos_gamma = cosine(cell[:, 0], cell[:, 1], lengths[:, 0], lengths[:, 1])
    return torch.cat(
        (log_lengths, cos_alpha.unsqueeze(-1), cos_beta.unsqueeze(-1),
         cos_gamma.unsqueeze(-1)),
        dim=-1,
    )


@dataclass(frozen=True)
class HiLoGeometry:
    """Periodic neighbourhood shared by all Hi-Fi heads of a stack.

    Attributes:
        neighbor_mask: ``(B, N, N)`` bool; ``True`` where atom ``j`` is inside
            the cutoff of atom ``i`` (self entries are ``True`` only for
            isolated atoms, which attend to themselves).
        pair_index: ``(E, 3)`` long tensor of ``(batch, i, j)`` neighbour
            triples.
        edge_features: ``(E, edge_dim)`` tensor holding ``e_ij`` for each
            triple, i.e. ``gamma_df(df*_ij) (+) gamma_d(dbar_ij)``.
        cutoff: ``r_c`` used to build the graph.
    """

    neighbor_mask: torch.Tensor
    pair_index: torch.Tensor
    edge_features: torch.Tensor
    cutoff: float


def build_hi_lo_geometry(
    frac_coords: torch.Tensor,
    cell: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    cutoff: float = 5.0,
    max_shifts: int = 1,
    num_freqs: int = 4,
) -> HiLoGeometry:
    """Build the neighbour graph and ``e_ij`` features once per forward pass.

    Args:
        frac_coords: ``(B, N, 3)`` fractional coordinates.
        cell: ``(B, 3, 3)`` (or per-atom ``(B, N, 3, 3)``) lattice matrices.
        mask: ``(B, N)`` bool, ``True`` for real atoms. ``None`` means all real.
        cutoff: ``r_c`` in Angstrom.
        max_shifts: integer image search radius.
        num_freqs: Fourier bands for ``gamma_df`` / ``gamma_d``.

    Returns:
        A :class:`HiLoGeometry`. Building it once and handing it to every
        layer avoids recomputing the 27-image search per layer.
    """
    if cutoff <= 0.0:
        raise ValueError("cutoff must be positive")
    if frac_coords.dim() != 3 or frac_coords.size(-1) != 3:
        raise ValueError("frac_coords must have shape (B, N, 3)")

    batch_size, num_tokens, _ = frac_coords.shape
    cell = _normalize_cell(cell, batch_size)
    if mask is None:
        mask = torch.ones(
            batch_size, num_tokens, dtype=torch.bool, device=frac_coords.device
        )
    else:
        mask = mask.to(device=frac_coords.device, dtype=torch.bool)
        if mask.shape != (batch_size, num_tokens):
            raise ValueError("mask must have shape (B, N)")

    displacement, distance = minimum_image_displacements(
        frac_coords, cell, max_shifts
    )

    same_structure = mask[:, :, None] & mask[:, None, :]
    eye = torch.eye(num_tokens, dtype=torch.bool, device=frac_coords.device)
    eye = eye.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    neighbor = same_structure & (distance < cutoff) & ~eye

    # An atom with no partner inside the cutoff would otherwise have an
    # all-masked softmax row; let it attend to itself instead.
    isolated = mask & ~neighbor.any(dim=-1)
    if isolated.any():
        neighbor = neighbor | (isolated[:, :, None] & eye)

    pair_index = neighbor.nonzero(as_tuple=False)  # (E, 3)
    batch_idx, row_idx, col_idx = pair_index.unbind(dim=-1)
    disp_pairs = displacement[batch_idx, row_idx, col_idx]
    dist_pairs = distance[batch_idx, row_idx, col_idx]
    edge_features = torch.cat(
        (
            _fractional_fourier(disp_pairs, num_freqs),
            _distance_fourier(dist_pairs / cutoff, num_freqs),
        ),
        dim=-1,
    )

    return HiLoGeometry(
        neighbor_mask=neighbor,
        pair_index=pair_index,
        edge_features=edge_features,
        cutoff=float(cutoff),
    )


# --------------------------------------------------------------------------- #
# Hi-Fi / Lo-Fi attention
# --------------------------------------------------------------------------- #
class HiLoAttention(nn.Module):
    """Multi-head attention with Hi-Fi and Lo-Fi heads (see module docstring).

    Args:
        d_model: token width.
        nhead: total head budget ``N_h``; must divide the attention width.
        low_freq_heads: ``alpha N_h`` heads routed to the global branch.
            ``0`` gives local attention only, ``nhead`` global attention only.
        head_dim: per-head width ``d_h``; defaults to ``d_model // nhead``.
        dropout: attention weight dropout.
        bias: use bias in the input / output projections.
        cutoff: ``r_c`` for the Hi-Fi neighbour graph (Angstrom).
        max_shifts: integer image search radius for the minimum image rule.
        num_freqs: Fourier bands for ``gamma_df`` / ``gamma_d``.
        edge_hidden: hidden width of ``MLP_edge,h``.
        global_hidden: hidden width of ``MLP_global,h`` / ``MLP_g,h``.
        pool_type: how ``x_tilde`` is built (``"grid"``, ``"chunk"``,
            ``"none"``).
        pool_grid_size: ``g`` for ``pool_type="grid"``; produces ``g^3``
            coarse tokens per structure.
        pool_chunk_size: block size for ``pool_type="chunk"``.
        bias_dim: ``d_b`` in the Lo-Fi bias scale; defaults to ``d_h``.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        *,
        low_freq_heads: int = 2,
        head_dim: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
        cutoff: float = 5.0,
        max_shifts: int = 1,
        num_freqs: int = 4,
        edge_hidden: int = 64,
        global_hidden: int = 64,
        pool_type: PoolType = "grid",
        pool_grid_size: int = 3,
        pool_chunk_size: int = 4,
        bias_dim: int | None = None,
    ) -> None:
        super().__init__()
        if d_model < 1:
            raise ValueError("d_model must be positive")
        if nhead < 1:
            raise ValueError("nhead must be positive")
        if low_freq_heads < 0 or low_freq_heads > nhead:
            raise ValueError("low_freq_heads must be in [0, nhead]")
        if head_dim is None:
            if d_model % nhead != 0:
                raise ValueError("d_model must be divisible by nhead")
            head_dim = d_model // nhead
        if head_dim < 1:
            raise ValueError("head_dim must be positive")
        if cutoff <= 0.0:
            raise ValueError("cutoff must be positive")
        if max_shifts < 0:
            raise ValueError("max_shifts must be non-negative")
        if pool_type not in _POOL_TYPES:
            raise ValueError(f"pool_type must be one of {_POOL_TYPES}")
        if pool_grid_size < 1:
            raise ValueError("pool_grid_size must be positive")
        if pool_chunk_size < 1:
            raise ValueError("pool_chunk_size must be positive")
        if bias_dim is None:
            bias_dim = head_dim
        if bias_dim < 1:
            raise ValueError("bias_dim must be positive")

        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.head_dim = int(head_dim)
        self.n_lo = int(low_freq_heads)
        self.n_hi = self.nhead - self.n_lo
        self.scale = self.head_dim ** -0.5
        self.cutoff = float(cutoff)
        self.max_shifts = int(max_shifts)
        self.num_freqs = int(num_freqs)
        self.edge_dim = edge_feature_dim(self.num_freqs)
        self.pool_type = pool_type
        self.pool_grid_size = int(pool_grid_size)
        self.pool_chunk_size = int(pool_chunk_size)
        self.bias_dim = int(bias_dim)
        self.dropout = float(dropout)
        self.attn_dropout = nn.Dropout(self.dropout)

        if self.n_hi > 0:
            self.hi_q = nn.Linear(d_model, self.n_hi * self.head_dim, bias=bias)
            self.hi_k = nn.Linear(d_model, self.n_hi * self.head_dim, bias=bias)
            self.hi_v = nn.Linear(d_model, self.n_hi * self.head_dim, bias=bias)
            self.edge_mlps = nn.ModuleList(
                nn.Sequential(
                    nn.Linear(self.edge_dim, edge_hidden),
                    nn.SiLU(),
                    nn.Linear(edge_hidden, 1),
                )
                for _ in range(self.n_hi)
            )
            for mlp in self.edge_mlps:
                nn.init.zeros_(mlp[-1].weight)
                nn.init.zeros_(mlp[-1].bias)
        else:
            self.edge_mlps = nn.ModuleList()

        if self.n_lo > 0:
            self.lo_q = nn.Linear(d_model, self.n_lo * self.head_dim, bias=bias)
            self.lo_k = nn.Linear(d_model, self.n_lo * self.head_dim, bias=bias)
            self.lo_v = nn.Linear(d_model, self.n_lo * self.head_dim, bias=bias)
            self.lo_w = nn.Sequential(
                nn.Linear(6, global_hidden),
                nn.SiLU(),
                nn.Linear(global_hidden, self.n_lo * self.head_dim),
            )
            self.lo_g = nn.Sequential(
                nn.Linear(6, global_hidden),
                nn.SiLU(),
                nn.Linear(global_hidden, self.n_lo),
            )
            nn.init.zeros_(self.lo_w[-1].weight)
            nn.init.zeros_(self.lo_w[-1].bias)
            nn.init.zeros_(self.lo_g[-1].weight)
            nn.init.zeros_(self.lo_g[-1].bias)

        self.out_proj = nn.Linear(self.nhead * self.head_dim, d_model, bias=bias)

    # ------------------------------------------------------------------ utils
    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, nhead={self.nhead}, "
            f"n_hi={self.n_hi}, n_lo={self.n_lo}, head_dim={self.head_dim}, "
            f"cutoff={self.cutoff}, pool_type={self.pool_type!r}"
        )

    def _split_heads(self, tensor: torch.Tensor, heads: int) -> torch.Tensor:
        batch_size, num_tokens, _ = tensor.shape
        return tensor.view(batch_size, num_tokens, heads, self.head_dim).transpose(1, 2)

    def _merge_heads(
        self, tensor: torch.Tensor, batch_size: int, num_tokens: int
    ) -> torch.Tensor:
        return tensor.transpose(1, 2).reshape(
            batch_size, num_tokens, tensor.size(1) * self.head_dim
        )

    def _pool_coarse(
        self, x: torch.Tensor, frac_coords: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Average-pool fine tokens into ``x_tilde`` (low-frequency tokens)."""
        if self.pool_type == "none":
            return x, mask

        batch_size, num_tokens, dim = x.shape
        if self.pool_type == "grid":
            grid = self.pool_grid_size
            wrapped = frac_coords % 1.0
            bins = torch.floor(wrapped * grid).long().clamp(0, grid - 1)
            bin_index = (bins[..., 0] * grid + bins[..., 1]) * grid + bins[..., 2]
            num_coarse = grid**3
        else:  # "chunk"
            chunk = self.pool_chunk_size
            num_coarse = (num_tokens + chunk - 1) // chunk
            positions = torch.arange(num_tokens, device=x.device)
            bin_index = (positions // chunk).unsqueeze(0).expand(batch_size, -1)

        weights = mask.to(x.dtype)
        # index_add_ wants a 1-D index, so flatten (batch, coarse) into one axis
        offsets = torch.arange(batch_size, device=x.device).unsqueeze(1) * num_coarse
        flat_index = (bin_index + offsets).reshape(-1)
        pooled = x.new_zeros(batch_size * num_coarse, dim)
        counts = x.new_zeros(batch_size * num_coarse)
        pooled.index_add_(
            0, flat_index, (x * weights.unsqueeze(-1)).reshape(-1, dim)
        )
        counts.index_add_(0, flat_index, weights.reshape(-1))
        pooled = pooled.view(batch_size, num_coarse, dim)
        counts = counts.view(batch_size, num_coarse)
        coarse_mask = counts > 0
        pooled = pooled / counts.clamp_min(1.0).unsqueeze(-1)
        return pooled, coarse_mask

    def _hi_fi_bias(self, geometry: HiLoGeometry) -> torch.Tensor:
        """``B^Hi-Fi_hij = MLP_edge,h(e_ij)`` scattered into a dense bias."""
        batch_size, num_tokens, _ = geometry.neighbor_mask.shape
        bias = geometry.edge_features.new_zeros(
            batch_size, self.n_hi, num_tokens, num_tokens
        )
        head_bias = torch.cat(
            [mlp(geometry.edge_features) for mlp in self.edge_mlps], dim=-1
        )
        batch_idx, row_idx, col_idx = geometry.pair_index.unbind(dim=-1)
        bias[batch_idx, :, row_idx, col_idx] = head_bias.to(bias.dtype)
        return bias

    def _lo_fi_bias(
        self, psi: torch.Tensor, query: torch.Tensor, key: torch.Tensor
    ) -> torch.Tensor:
        """``B^Lo-Fi_hIJ = (g_h / sqrt(d_b)) (q_hI (*) w_h)^T k_hJ``."""
        weight = self.lo_w(psi).view(-1, self.n_lo, self.head_dim)
        gate = torch.sigmoid(self.lo_g(psi))
        bias = torch.einsum("bhid,bhd,bhjd->bhij", query, weight, key)
        scale = gate / math.sqrt(self.bias_dim)
        return bias * scale[:, :, None, None]

    def _attend(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        bias: torch.Tensor | None,
        token_mask: torch.Tensor,
        pair_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Masked scaled dot-product attention with an additive bias."""
        logits = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        if bias is not None:
            logits = logits + bias.to(logits.dtype)
        if pair_mask is not None:
            allowed = pair_mask[:, None]
        else:
            allowed = token_mask[:, None, None, :]
        logits = logits.masked_fill(~allowed, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        weights = self.attn_dropout(weights)
        return torch.matmul(weights, value)

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        x: torch.Tensor,
        frac_coords: torch.Tensor,
        cell: torch.Tensor,
        mask: torch.Tensor | None = None,
        geometry: HiLoGeometry | None = None,
    ) -> torch.Tensor:
        """Apply the attention block to dense crystal tokens.

        Args:
            x: ``(B, N, d_model)`` token features (padding slots may hold
                anything; their outputs are zeroed).
            frac_coords: ``(B, N, 3)`` fractional coordinates.
            cell: ``(B, 3, 3)`` lattice matrices (rows ``a, b, c``); per-atom
                ``(B, N, 3, 3)`` is also accepted.
            mask: ``(B, N)`` bool, ``True`` for real atoms.
            geometry: pre-computed :class:`HiLoGeometry`; built on the fly when
                omitted and the Hi-Fi branch is active.

        Returns:
            ``(B, N, d_model)`` with zeros on padded tokens.
        """
        if x.dim() != 3:
            raise ValueError("x must have shape (B, N, d_model)")
        batch_size, num_tokens, dim = x.shape
        if dim != self.d_model:
            raise ValueError(f"expected d_model={self.d_model}, got {dim}")
        if frac_coords.shape != (batch_size, num_tokens, 3):
            raise ValueError("frac_coords must have shape (B, N, 3)")

        frac_coords = frac_coords.to(device=x.device, dtype=x.dtype)
        cell = _normalize_cell(cell.to(device=x.device, dtype=x.dtype), batch_size)
        if mask is None:
            mask = torch.ones(
                batch_size, num_tokens, dtype=torch.bool, device=x.device
            )
        else:
            mask = mask.to(device=x.device, dtype=torch.bool)
            if mask.shape != (batch_size, num_tokens):
                raise ValueError("mask must have shape (B, N)")

        heads = []

        if self.n_hi > 0:
            if geometry is None:
                geometry = build_hi_lo_geometry(
                    frac_coords,
                    cell,
                    mask,
                    cutoff=self.cutoff,
                    max_shifts=self.max_shifts,
                    num_freqs=self.num_freqs,
                )
            else:
                if geometry.neighbor_mask.shape != (batch_size, num_tokens, num_tokens):
                    raise ValueError("geometry does not match the token batch")
                if geometry.edge_features.size(-1) != self.edge_dim:
                    raise ValueError(
                        "geometry edge_features width does not match num_freqs"
                    )

            query = self._split_heads(self.hi_q(x), self.n_hi)
            key = self._split_heads(self.hi_k(x), self.n_hi)
            value = self._split_heads(self.hi_v(x), self.n_hi)
            output = self._attend(
                query,
                key,
                value,
                self._hi_fi_bias(geometry),
                mask,
                pair_mask=geometry.neighbor_mask,
            )
            heads.append(self._merge_heads(output, batch_size, num_tokens))

        if self.n_lo > 0:
            coarse, coarse_mask = self._pool_coarse(x, frac_coords, mask)
            query = self._split_heads(self.lo_q(x), self.n_lo)
            key = self._split_heads(self.lo_k(coarse), self.n_lo)
            value = self._split_heads(self.lo_v(coarse), self.n_lo)
            psi = lattice_descriptor(cell, mask.sum(dim=-1))
            output = self._attend(
                query,
                key,
                value,
                self._lo_fi_bias(psi, query, key),
                coarse_mask,
            )
            heads.append(self._merge_heads(output, batch_size, num_tokens))

        output = self.out_proj(torch.cat(heads, dim=-1))
        return output * mask.unsqueeze(-1).to(output.dtype)


# --------------------------------------------------------------------------- #
# Pre-norm transformer block and stack
# --------------------------------------------------------------------------- #
def _get_activation(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU(approximate="tanh")
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"unsupported activation {name!r}")


class HiLoTransformerEncoderLayer(nn.Module):
    """Pre-norm transformer encoder layer whose self-attention is
    :class:`HiLoAttention`.

    Drop-in analogue of ``nn.TransformerEncoderLayer`` for dense crystal
    batches, with the extra ``frac_coords`` / ``cell`` / ``geometry``
    arguments needed by the periodic attention bias.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int | None = None,
        dropout: float = 0.0,
        activation: str = "gelu",
        norm_first: bool = True,
        bias: bool = True,
        low_freq_heads: int = 2,
        head_dim: int | None = None,
        cutoff: float = 5.0,
        max_shifts: int = 1,
        num_freqs: int = 4,
        edge_hidden: int = 64,
        global_hidden: int = 64,
        pool_type: PoolType = "grid",
        pool_grid_size: int = 3,
        pool_chunk_size: int = 4,
        bias_dim: int | None = None,
    ) -> None:
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model
        self.d_model = int(d_model)
        self.self_attn = HiLoAttention(
            d_model,
            nhead,
            low_freq_heads=low_freq_heads,
            head_dim=head_dim,
            dropout=dropout,
            bias=bias,
            cutoff=cutoff,
            max_shifts=max_shifts,
            num_freqs=num_freqs,
            edge_hidden=edge_hidden,
            global_hidden=global_hidden,
            pool_type=pool_type,
            pool_grid_size=pool_grid_size,
            pool_chunk_size=pool_chunk_size,
            bias_dim=bias_dim,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)
        self.norm_first = bool(norm_first)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation(activation)

    def _sa_block(
        self,
        x: torch.Tensor,
        frac_coords: torch.Tensor,
        cell: torch.Tensor,
        mask: torch.Tensor | None,
        geometry: HiLoGeometry | None,
    ) -> torch.Tensor:
        x = self.self_attn(x, frac_coords, cell, mask, geometry)
        return self.dropout1(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def forward(
        self,
        x: torch.Tensor,
        frac_coords: torch.Tensor,
        cell: torch.Tensor,
        mask: torch.Tensor | None = None,
        geometry: HiLoGeometry | None = None,
    ) -> torch.Tensor:
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), frac_coords, cell, mask, geometry
            )
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                x + self._sa_block(x, frac_coords, cell, mask, geometry)
            )
            x = self.norm2(x + self._ff_block(x))
        if mask is not None:
            # Keep padding slots exactly zero so they can never leak into the
            # next block (attention masks keys, but the FFN is position-wise).
            x = x * mask.unsqueeze(-1).to(x.dtype)
        return x


class HiLoTransformerEncoder(nn.Module):
    """Stack of :class:`HiLoTransformerEncoderLayer` sharing one geometry.

    The minimum-image neighbour search and edge features depend only on
    ``frac_coords`` / ``cell``, so they are computed once per forward pass and
    reused by every layer. All layers must therefore use the same ``cutoff``,
    ``max_shifts`` and ``num_freqs`` (validated at construction).
    """

    def __init__(
        self,
        encoder_layer: HiLoTransformerEncoderLayer,
        num_layers: int,
        norm: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if not isinstance(encoder_layer, HiLoTransformerEncoderLayer):
            raise TypeError(
                "encoder_layer must be a HiLoTransformerEncoderLayer"
            )
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.num_layers = int(num_layers)
        self.norm = norm
        self.d_model = encoder_layer.d_model

        configs = {
            (layer.self_attn.cutoff, layer.self_attn.max_shifts,
             layer.self_attn.num_freqs)
            for layer in self.layers
        }
        if len(configs) > 1:
            raise ValueError(
                "all layers must share cutoff / max_shifts / num_freqs so a "
                "single geometry can be reused"
            )
        self._cutoff, self._max_shifts, self._num_freqs = configs.pop()
        self._needs_geometry = any(
            layer.self_attn.n_hi > 0 for layer in self.layers
        )

    def forward(
        self,
        x: torch.Tensor,
        frac_coords: torch.Tensor,
        cell: torch.Tensor,
        mask: torch.Tensor | None = None,
        geometry: HiLoGeometry | None = None,
    ) -> torch.Tensor:
        if geometry is None and self._needs_geometry:
            geometry = build_hi_lo_geometry(
                frac_coords,
                cell,
                mask,
                cutoff=self._cutoff,
                max_shifts=self._max_shifts,
                num_freqs=self._num_freqs,
            )
        for layer in self.layers:
            x = layer(x, frac_coords, cell, mask, geometry)
        if self.norm is not None:
            x = self.norm(x)
        return x
