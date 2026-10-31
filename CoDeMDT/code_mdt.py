"""CoDeMDT: Context-Diverse Masked Diffusion Transformer.

Reference implementation for the composition-conditioned crystal latent
flow-matching pipeline of ADITcsp. The canonical integrated module lives at
``ADITcsp/src/models/denoisers/code_mdt.py`` and is selected by
``src/train_csp.py --denoiser code_mdt``.

Design (see ``INTEGRATION.md`` for the wiring):

* Backbone is MDTv2 unchanged -- asymmetric masked encoder with U-Net long
  shortcuts, one side interpolater (SI) block, and a decoder with a dense input
  shortcut from the noisy latent.
* The only new mechanism is an **encoder-side context diversity loss** computed
  on unmasked token representations. Rationale: the SI predicts masked tokens
  from encoder context; if every encoder block emits the same representation the
  context is redundant and masked reconstruction degenerates to copying one
  feature. Diversity is therefore a *precondition* of masked reconstruction, not
  an extra regularizer, and it is deliberately not applied to decoder
  representations, which serve the denoising prediction.

Two substitutions are required because the tokens here are crystal atoms rather
than image patches, and both are configurable:

1. ``rel_pos_bias="1d"`` (default) keeps MDTv2's relative position bias role but
   uses a 1-D relative-offset table instead of the 2-D grid table, which is
   undefined for an unordered atom sequence. ``"none"`` disables it.
2. Masking works on variable-length padded sequences, so all gathers are padded
   and attention uses ``key_padding_mask``.

Integration contract used by ``LatentDiffusionLitModule``:

    forward(x, t, dataset_idx, spacegroup, mask, x_sc=None, composition=None,
            enable_mask=False, return_aux=False) -> pred [, aux]
    forward_with_cfg(x, t, dataset_idx, spacegroup, mask, cfg_scale, x_sc=None,
                     composition=None) -> pred
    supports_masking = True
    diversity_loss() -> dict | None
    diversity_term(step, diffusion_loss) -> tensor | None
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "CoDeMDT",
    "EncoderDiversityLoss",
    "DiversityWeighter",
    "RelativePositionBias1D",
]

_EPS = 1e-8


#################################################################################
#                         Base transformer components                           #
# (self-contained copies of DiT/MDT building blocks, underscore-prefixed so     #
#  this file never collides with ADITcsp's ``dit.py`` when both are imported)   #
#################################################################################


def _modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class _Mlp(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class _TimestepEmbedder(nn.Module):
    def __init__(self, hidden_dim, frequency_dim=256):
        super().__init__()
        self.frequency_dim = frequency_dim
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.frequency_dim))


class _LabelEmbedder(nn.Module):
    """Class embedding with label dropout for classifier-free guidance."""

    def __init__(self, num_classes, hidden_dim, dropout_prob):
        super().__init__()
        use_cfg = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg, hidden_dim)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def forward(self, labels, train):
        if train and self.dropout_prob > 0:
            drop = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            labels = torch.where(drop, torch.zeros_like(labels), labels)
        return self.embedding_table(labels)


def _get_pos_embedding(indices, emb_dim, max_len=2048):
    """Sine/cosine embedding from prespecified token indices ``[..., N]``."""
    k = torch.arange(emb_dim // 2, device=indices.device)
    sin = torch.sin(indices[..., None] * math.pi / (max_len ** (2 * k[None] / emb_dim)))
    cos = torch.cos(indices[..., None] * math.pi / (max_len ** (2 * k[None] / emb_dim)))
    return torch.cat([sin, cos], dim=-1)


class RelativePositionBias1D(nn.Module):
    """Learnable 1-D relative-offset bias, the sequence analogue of MDT's ``B_r``.

    MDT adds a 2-D grid bias per head. Crystal atom tokens have no 2-D geometry,
    so this table indexes the signed offset ``pos_i - pos_j`` instead. It plays
    the same architectural role and, unlike the 2-D table, is well defined for a
    padded variable-length sequence.

    Args:
        num_heads: attention heads sharing this table's output channels.
        max_distance: offsets are clamped to ``[-(max_distance-1), max_distance-1]``.
    """

    def __init__(self, num_heads, max_distance=64):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.table = nn.Parameter(torch.zeros(2 * max_distance - 1, num_heads))
        nn.init.normal_(self.table, std=0.02)

    def forward(self, positions):
        """positions: ``[B, N]`` non-negative token indices -> bias ``[B, H, N, N]``."""
        offset = positions.unsqueeze(2) - positions.unsqueeze(1)
        offset = offset.clamp(-(self.max_distance - 1), self.max_distance - 1)
        offset = offset + (self.max_distance - 1)
        return self.table[offset].permute(0, 3, 1, 2)


class _Attention(nn.Module):
    """Multi-head self-attention with relative position bias and key padding mask."""

    def __init__(self, dim, num_heads, dropout=0.0, rel_bias=None):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.rel_bias = rel_bias
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x, positions, pad_mask):
        batch, num_tokens, dim = x.shape
        qkv = self.qkv(x).reshape(
            batch, num_tokens, 3, self.num_heads, dim // self.num_heads
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.rel_bias is not None:
            attn = attn + self.rel_bias(positions)
        if pad_mask is not None:
            attn = attn.masked_fill(pad_mask[:, None, None, :], float("-inf"))
        attn = self.attn_drop(attn.softmax(dim=-1))

        out = (attn @ v).transpose(1, 2).reshape(batch, num_tokens, dim)
        return self.proj_drop(self.proj(out))


class _Block(nn.Module):
    """adaLN-Zero transformer block with an optional U-Net style skip fusion."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0, rel_bias=None, skip=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = _Attention(dim, num_heads, dropout=dropout, rel_bias=rel_bias)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = _Mlp(dim, int(dim * mlp_ratio), dropout=dropout)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )
        self.skip_linear = nn.Linear(2 * dim, dim, bias=True) if skip else None

    def forward(self, x, c, positions, pad_mask, skip=None):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            _modulate(self.norm1(x), shift_msa, scale_msa), positions, pad_mask
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            _modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class _FinalLayer(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, out_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return self.linear(_modulate(self.norm_final(x), shift, scale))


#################################################################################
#                     Encoder-side context diversity loss                       #
#################################################################################


class EncoderDiversityLoss(nn.Module):
    """Three-term diversity penalty over encoder block representations.

    Given encoder block outputs ``h_l in R^{N_hat x D}`` restricted to unmasked
    tokens, the terms are

        L_orth = mean_{(i,j)} cos(mu_i, mu_j)                # block means differ
        L_MI   = mean_{(i,j)} mean_n  <h_hat_i,n, h_hat_j,n> # token features differ
        L_disp = -Var_d(a'_k)                                # dimensions disperse

    with ``mu_l`` the token mean, ``h_hat`` the L2-normalised token feature, and
    ``a'`` the dimension-wise mean activation normalised by its maximum
    magnitude. ``L_disp`` is computed once per sample from the mean over the
    selected blocks, since the spec gives no layer index for it.

    All three are positive-similarity terms except ``L_disp``, which is negated,
    so minimising the sum pushes the representations apart.

    Args:
        lambda_orth: weight of ``L_orth``.
        lambda_mi: weight of ``L_MI``.
        lambda_disp: weight of ``L_disp``.
    """

    def __init__(self, lambda_orth=1.0, lambda_mi=1.0, lambda_disp=0.1):
        super().__init__()
        self.lambda_orth = float(lambda_orth)
        self.lambda_mi = float(lambda_mi)
        self.lambda_disp = float(lambda_disp)

    def forward(self, features, token_mask):
        """features: list of ``[B, K, D]`` tensors; token_mask: bool ``[B, K]``."""
        features = [f for f in features if f is not None]
        if len(features) < 2:
            return None
        num_blocks = len(features)
        pairs = [(i, j) for i in range(num_blocks) for j in range(i + 1, num_blocks)]

        mask = token_mask.to(features[0].dtype).unsqueeze(-1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        means = [(h * mask).sum(dim=1) / counts for h in features]
        norms = [
            h / h.norm(dim=-1, keepdim=True).clamp(min=_EPS) for h in features
        ]

        token_weight = mask.sum().clamp(min=1.0)
        orth_terms, mi_terms = [], []
        for i, j in pairs:
            orth_terms.append(
                F.cosine_similarity(means[i], means[j], dim=-1, eps=_EPS).mean()
            )
            per_token = (norms[i] * norms[j]).sum(dim=-1)
            mi_terms.append(((per_token * mask.squeeze(-1)).sum()) / token_weight)

        loss_orth = torch.stack(orth_terms).mean()
        loss_mi = torch.stack(mi_terms).mean()

        activation = torch.stack(means, dim=0).mean(dim=0)
        activation = activation / activation.abs().amax(dim=-1, keepdim=True).clamp(
            min=_EPS
        )
        loss_disp = -activation.var(dim=-1, unbiased=False).mean()

        total = (
            self.lambda_orth * loss_orth
            + self.lambda_mi * loss_mi
            + self.lambda_disp * loss_disp
        )
        return {
            "div_loss": total,
            "loss_orth": loss_orth,
            "loss_mi": loss_mi,
            "loss_disp": loss_disp,
            "num_pairs": len(pairs),
            "num_tokens": int(token_mask.sum().item()),
        }


class DiversityWeighter:
    """Bounded, optionally adaptive weight for the diversity term.

    ``w`` is a plain Python float (detached), so it never enters the graph:

        raw    = target_ratio * L_diff / L_div           # adaptive (default)
        raw    = base_weight                             # fixed
        w      = clamp(raw, min_weight, max_weight) * warmup_ramp

    ``warmup_ramp`` is ``min(1, (step + 1) / warmup_steps)``. The clamps keep the
    term inside ``[min_weight, max_weight]`` (default ``[0.1, 0.5]``), as in the
    design note. Adaptive mode keeps ``w * L_div`` proportional to the diffusion
    loss; early in training the ratio saturates at ``max_weight`` and decays as
    the reconstruction loss shrinks.
    """

    def __init__(
        self,
        base_weight=0.2,
        min_weight=0.1,
        max_weight=0.5,
        warmup_steps=0,
        adaptive=True,
        target_ratio=0.02,
    ):
        if not 0.0 <= min_weight <= max_weight:
            raise ValueError("require 0 <= min_weight <= max_weight")
        self.base_weight = float(base_weight)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.warmup_steps = int(warmup_steps)
        self.adaptive = bool(adaptive)
        self.target_ratio = float(target_ratio)
        self.last_weight = float(base_weight)

    def weight(self, step, div_loss, diffusion_loss=None):
        raw = self.base_weight
        if self.adaptive and diffusion_loss is not None:
            raw = self.target_ratio * float(diffusion_loss) / max(
                float(div_loss), _EPS
            )
        weight = min(max(raw, self.min_weight), self.max_weight)
        if self.warmup_steps > 0:
            weight *= min(1.0, (int(step) + 1) / self.warmup_steps)
        self.last_weight = float(weight)
        return self.last_weight


#################################################################################
#                                   CoDeMDT                                    #
#################################################################################


class CoDeMDT(nn.Module):
    """Context-Diverse Masked Diffusion Transformer.

    Layout (``num_layers`` blocks total):

        en_in   : (num_layers - decode_layer) // 2 blocks, plain
        en_out  : (num_layers - decode_layer) // 2 blocks, U-Net skip from en_in
        side    : 1 block, the side interpolater (SI)
        de      : decode_layer blocks, dense skip from the noisy latent

    The masked pass runs ``en_*`` on the visible subset, scatters the result back
    with ``mask_token`` filling the removed slots, applies the masked shortcut
    ``x = SI(x) * M + x * (1 - M)``, and feeds the full sequence to ``de``. The
    plain pass never masks and is the only path used at inference.

    Args:
        d_x: latent dimension per atom token.
        d_model: transformer width.
        num_layers: block budget for encoder + decoder (the SI block is extra).
        nhead: attention heads per block.
        mlp_ratio: MLP expansion ratio.
        class_dropout_prob: label dropout for classifier-free guidance.
        num_datasets: dataset label count (0 is the null class).
        num_spacegroups: spacegroup label count (0 is the null class).
        composition_dim: element-count vector length; 0 disables composition.
        composition_dropout_prob: composition dropout for classifier-free guidance.
        dropout: attention/MLP dropout.
        mask_ratio: base fraction of visible tokens removed by the masked pass.
        mask_ratio_range: sampled ratio is ``U[mask_ratio, mask_ratio + range]``.
        decode_layer: decoder block count; ``num_layers - decode_layer`` must be
            even and at least 2.
        lambda_orth, lambda_mi, lambda_disp: diversity term weights.
        div_weight, div_weight_min, div_weight_max: bounded diversity weight.
        div_warmup_steps: linear ramp length for the diversity weight.
        div_adaptive: use the loss-ratio adaptive weight instead of a constant.
        div_target_ratio: adaptive target for ``w * L_div / L_diff``.
        div_blocks: indices of encoder blocks used by the diversity loss
            (0-based over ``encoder_blocks``); ``None`` uses all of them.
        rel_pos_bias: ``"1d"`` (default) or ``"none"``.
        rel_pos_max_distance: offset clamp for the 1-D relative bias table.
        div_on_plain: also compute the diversity loss in the plain pass. Off by
            default: the design ties diversity to the masked reconstruction path.
    """

    supports_masking = True
    has_diversity = True

    def __init__(
        self,
        d_x=8,
        d_model=384,
        num_layers=6,
        nhead=6,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_datasets=2,
        num_spacegroups=230,
        composition_dim=0,
        composition_dropout_prob=0.0,
        dropout=0.0,
        mask_ratio=0.3,
        mask_ratio_range=0.2,
        decode_layer=None,
        lambda_orth=1.0,
        lambda_mi=1.0,
        lambda_disp=0.1,
        div_weight=0.2,
        div_weight_min=0.1,
        div_weight_max=0.5,
        div_warmup_steps=0,
        div_adaptive=True,
        div_target_ratio=0.02,
        div_blocks=None,
        rel_pos_bias="1d",
        rel_pos_max_distance=64,
        div_on_plain=False,
    ):
        super().__init__()
        self.d_x = d_x
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.mask_ratio = float(mask_ratio)
        self.mask_ratio_range = float(mask_ratio_range)
        self.div_on_plain = bool(div_on_plain)
        if not 0.0 < self.mask_ratio < 1.0:
            raise ValueError("mask_ratio must be in (0, 1)")
        if self.mask_ratio_range < 0.0:
            raise ValueError("mask_ratio_range must be non-negative")
        if rel_pos_bias not in ("1d", "none"):
            raise ValueError("rel_pos_bias must be '1d' or 'none'")

        decode_layer = int(decode_layer) if decode_layer else max(1, num_layers // 3)
        encode_depth = num_layers - decode_layer
        if decode_layer < 1 or encode_depth < 2 or encode_depth % 2 != 0:
            raise ValueError(
                f"invalid split: num_layers={num_layers}, decode_layer={decode_layer}; "
                "num_layers - decode_layer must be even and >= 2"
            )
        self.decode_layer = decode_layer
        self.half_depth = encode_depth // 2
        self.rel_pos_bias_kind = rel_pos_bias

        def make_rel_bias():
            if rel_pos_bias != "1d":
                return None
            return RelativePositionBias1D(nhead, max_distance=rel_pos_max_distance)

        self.x_embedder = nn.Linear(2 * d_x, d_model, bias=True)
        self.t_embedder = _TimestepEmbedder(d_model)
        self.dataset_embedder = _LabelEmbedder(num_datasets, d_model, class_dropout_prob)
        self.spacegroup_embedder = _LabelEmbedder(
            num_spacegroups, d_model, class_dropout_prob
        )
        self.composition_dim = composition_dim
        self.composition_dropout_prob = composition_dropout_prob
        self.composition_embedder = None
        if composition_dim > 0:
            self.composition_embedder = nn.Sequential(
                nn.Linear(composition_dim, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
            )

        self.en_in_blocks = nn.ModuleList(
            [
                _Block(
                    d_model, nhead, mlp_ratio, dropout, rel_bias=make_rel_bias()
                )
                for _ in range(self.half_depth)
            ]
        )
        self.en_out_blocks = nn.ModuleList(
            [
                _Block(
                    d_model,
                    nhead,
                    mlp_ratio,
                    dropout,
                    rel_bias=make_rel_bias(),
                    skip=True,
                )
                for _ in range(self.half_depth)
            ]
        )
        self.side_blocks = nn.ModuleList(
            [_Block(d_model, nhead, mlp_ratio, dropout, rel_bias=make_rel_bias())]
        )
        self.de_blocks = nn.ModuleList(
            [
                _Block(
                    d_model,
                    nhead,
                    mlp_ratio,
                    dropout,
                    rel_bias=make_rel_bias(),
                    skip=True,
                )
                for _ in range(decode_layer)
            ]
        )
        self.final_layer = _FinalLayer(d_model, d_x)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))

        self.div_loss_fn = EncoderDiversityLoss(lambda_orth, lambda_mi, lambda_disp)
        self.div_weights = DiversityWeighter(
            base_weight=div_weight,
            min_weight=div_weight_min,
            max_weight=div_weight_max,
            warmup_steps=div_warmup_steps,
            adaptive=div_adaptive,
            target_ratio=div_target_ratio,
        )
        self.div_block_indices = (
            tuple(div_blocks) if div_blocks is not None else None
        )
        self._last_div = None

        self.initialize_weights()
        print(
            f"[CoDeMDT] mask_ratio={self.mask_ratio} "
            f"range={self.mask_ratio_range} decode_layer={decode_layer} "
            f"half_depth={self.half_depth} blocks={self.num_blocks} "
            f"rel_pos_bias={rel_pos_bias} div_blocks={len(self.encoder_blocks)} "
            f"w=[{div_weight_min:g},{div_weight_max:g}]",
            flush=True,
        )

    # ------------------------------------------------------------- properties
    @property
    def encoder_blocks(self):
        """Ordered encoder blocks: the two halves, exactly as MDTv2 indexes them."""
        return [*self.en_in_blocks, *self.en_out_blocks]

    @property
    def blocks(self):
        """All blocks in forward order (encoder, SI, decoder)."""
        return [*self.en_in_blocks, *self.en_out_blocks, *self.side_blocks, *self.de_blocks]

    @property
    def num_blocks(self):
        return len(self.blocks)

    @property
    def cka_blocks(self):
        """Blocks executed on every forward.

        The SI block only runs in the masked training pass, so hooking it would
        leave a permanently stale row in the CKA matrix (collected at validation,
        which uses the plain pass).
        """
        return [*self.en_in_blocks, *self.en_out_blocks, *self.de_blocks]

    def initialize_weights(self):
        def basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(basic_init)
        nn.init.normal_(self.dataset_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.spacegroup_embedder.embedding_table.weight, std=0.02)
        if self.composition_embedder is not None:
            nn.init.normal_(self.composition_embedder[0].weight, std=0.02)
            nn.init.normal_(self.composition_embedder[2].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    # -------------------------------------------------------------- internals
    def _build_condition(self, t, dataset_idx, spacegroup, composition, like):
        c = (
            self.t_embedder(t.squeeze(1))
            + self.dataset_embedder(dataset_idx, self.training)
            + self.spacegroup_embedder(spacegroup, self.training)
        )
        if self.composition_embedder is not None:
            if composition is None:
                composition = torch.zeros(
                    like.size(0), self.composition_dim, device=like.device, dtype=like.dtype
                )
            composition = composition.to(device=like.device, dtype=like.dtype)
            if self.training and self.composition_dropout_prob > 0.0:
                drop = (
                    torch.rand(composition.size(0), 1, device=composition.device)
                    < self.composition_dropout_prob
                )
                composition = composition.masked_fill(drop, 0.0)
            c = c + self.composition_embedder(composition)
        return c

    def _sampled_mask_ratio(self, device):
        extra = torch.rand(1, device=device).item() * self.mask_ratio_range
        return min(self.mask_ratio + extra, 0.95)

    def _random_masking(self, x, token_mask):
        """Hide a random subset of visible tokens.

        Returns compacted tokens ``[B, K, D]``, gather indices ``[B, K]`` and the
        boolean ``[B, K]`` mask marking real (non-padding) compacted slots.
        """
        batch, num_tokens, dim = x.shape
        lengths = token_mask.sum(dim=-1)
        ratio = self._sampled_mask_ratio(x.device)
        num_keep = torch.floor((1.0 - ratio) * lengths.float()).long().clamp(min=1)
        num_keep = torch.minimum(num_keep, lengths)
        keep_len = int(num_keep.max().item())

        noise = torch.rand(batch, num_tokens, device=x.device)
        noise = noise.masked_fill(~token_mask, 2.0)  # padding sorts last, never kept
        order = torch.argsort(noise, dim=1)

        slot = torch.arange(keep_len, device=x.device).unsqueeze(0)
        keep_mask = slot < num_keep.unsqueeze(1)
        dummy = torch.zeros(batch, keep_len, dtype=torch.long, device=x.device)
        ids_keep = torch.where(keep_mask, order[:, :keep_len], dummy)

        x_keep = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, dim))
        x_keep = x_keep * keep_mask.unsqueeze(-1)
        return x_keep, ids_keep, keep_mask

    def _merge_masked(self, x_keep, ids_keep, keep_mask, pad_mask, positions, pos_emb, c):
        """Scatter back, fill removed slots with ``mask_token``, run SI, apply shortcut."""
        batch, _, dim = x_keep.shape
        num_tokens = pad_mask.shape[1]

        # scatter_add_ is required: padded slots all point at index 0 and must be
        # additive no-ops (their features are exactly zero), not overwrites.
        filled = x_keep.new_zeros(batch, num_tokens, dim)
        filled.scatter_add_(1, ids_keep.unsqueeze(-1).expand(-1, -1, dim), x_keep)
        hit = x_keep.new_zeros(batch, num_tokens, 1)
        hit.scatter_add_(1, ids_keep.unsqueeze(-1), keep_mask.unsqueeze(-1).to(x_keep.dtype))
        removed = 1.0 - hit  # 1 marks a slot whose token was hidden

        x_before = filled + removed * self.mask_token + pos_emb
        x = x_before
        for block in self.side_blocks:
            x = block(x, c, positions, pad_mask)
        # masked shortcut: visible tokens keep the encoder output, hidden tokens
        # take the SI prediction
        return x * removed + x_before * (1.0 - removed)

    # ------------------------------------------------------------- diversity
    def _selected_encoder_features(self, features):
        if self.div_block_indices is None:
            return features
        return [features[i] for i in self.div_block_indices]

    def _compute_diversity(self, encoder_features, keep_mask):
        selected = self._selected_encoder_features(encoder_features)
        return self.div_loss_fn(selected, keep_mask)

    def diversity_loss(self):
        """Last diversity loss dict (tensors), or ``None`` if not computed."""
        return self._last_div

    def diversity_components(self):
        """Scalar snapshot of the last diversity loss, for logging."""
        if self._last_div is None:
            return {}
        return {
            key: float(value.detach())
            for key, value in self._last_div.items()
            if isinstance(value, torch.Tensor)
        }

    def diversity_term(self, step, diffusion_loss=None):
        """Weighted diversity loss to add to the training objective.

        Returns ``None`` when the last forward produced no diversity loss. The
        weight is a detached float, so only the loss itself carries gradients.
        """
        if self._last_div is None:
            return None
        div = self._last_div["div_loss"]
        weight = self.div_weights.weight(step, div.detach(), diffusion_loss)
        return weight * div

    def diversity_weight(self):
        return self.div_weights.last_weight

    # --------------------------------------------------------------- forward
    def forward(
        self,
        x,
        t,
        dataset_idx,
        spacegroup,
        mask,
        x_sc=None,
        composition=None,
        enable_mask=False,
        return_aux=False,
    ):
        """Predict the flow-matching target for ``x``.

        Args:
            x: ``[B, N, d_x]`` noisy latent tokens.
            t: ``[B, 1]`` flow-matching time.
            dataset_idx: ``[B]`` dataset label (>0; 0 is the null class).
            spacegroup: ``[B]`` spacegroup label.
            mask: ``[B, N]`` boolean, True for real tokens.
            x_sc: ``[B, N, d_x]`` self-conditioning prediction.
            composition: ``[B, composition_dim]`` element counts.
            enable_mask: run the masked pass (training only).
            return_aux: also return the diversity loss dict.
        """
        self._last_div = None

        token_index = torch.cumsum(mask, dim=-1, dtype=torch.int64) - 1
        token_index = token_index.clamp(min=0)
        pos_emb = _get_pos_embedding(token_index, self.d_model)

        if x_sc is None:
            x_sc = torch.zeros_like(x)
        x = self.x_embedder(torch.cat([x, x_sc], dim=-1)) + pos_emb

        c = self._build_condition(t, dataset_idx, spacegroup, composition, x)
        pad_mask = ~mask
        input_skip = x

        masked_pass = bool(enable_mask)
        if masked_pass:
            x, ids_keep, keep_mask = self._random_masking(x, mask)
            encoder_mask = ~keep_mask
            encoder_positions = torch.gather(token_index, 1, ids_keep)
        else:
            keep_mask = mask
            encoder_mask = pad_mask
            encoder_positions = token_index

        encoder_features = []
        skips = []
        for block in self.en_in_blocks:
            x = block(x, c, encoder_positions, encoder_mask)
            skips.append(x)
            encoder_features.append(x)
        for block in self.en_out_blocks:
            x = block(x, c, encoder_positions, encoder_mask, skip=skips.pop())
            encoder_features.append(x)

        if masked_pass or self.div_on_plain:
            self._last_div = self._compute_diversity(encoder_features, keep_mask)

        if masked_pass:
            x = self._merge_masked(
                x, ids_keep, keep_mask, pad_mask, token_index, pos_emb, c
            )
        else:
            x = x + pos_emb

        for block in self.de_blocks:
            x = block(x, c, token_index, pad_mask, skip=input_skip)

        x = self.final_layer(x, c) * mask[..., None]
        if return_aux:
            return x, self._last_div
        return x

    def forward_with_cfg(
        self,
        x,
        t,
        dataset_idx,
        spacegroup,
        mask,
        cfg_scale,
        x_sc=None,
        composition=None,
    ):
        """Classifier-free guided forward, as expected by the sampler."""
        half_x = x[: len(x) // 2]
        combined_x = torch.cat([half_x, half_x], dim=0)
        if (
            self.composition_embedder is not None
            and composition is not None
            and composition.size(0) * 2 == x.size(0)
        ):
            composition = torch.cat([composition, torch.zeros_like(composition)], dim=0)
        model_out = self.forward(
            combined_x,
            t,
            dataset_idx,
            spacegroup,
            mask,
            x_sc,
            composition=composition,
            enable_mask=False,
        )
        cond_eps, uncond_eps = torch.split(model_out, len(model_out) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        return torch.cat([half_eps, half_eps], dim=0)

    # ------------------------------------------------------------------ cka
    def register_cka_hooks(self):
        """Capture embedding / block / output features for the CKA callback."""
        try:
            from src.models.components.cka import CKAFeatureCollector
        except ImportError as error:  # pragma: no cover - standalone use
            raise ImportError(
                "register_cka_hooks needs ADITcsp's src.models.components.cka on "
                "sys.path; it is only available after integrating the file."
            ) from error

        modules = {"embedding": self.x_embedder}
        modules.update(
            {
                f"block_{index + 1:02d}": block
                for index, block in enumerate(self.cka_blocks)
            }
        )
        modules["output"] = self.final_layer
        return CKAFeatureCollector(modules)
