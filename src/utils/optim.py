"""Optimizer helpers shared by VAE and latent diffusion training."""

import torch


def build_optimizer(params, learning_rate: float, weight_decay: float):
    """Build AdamW groups without weight decay on biases, norms, and embeddings."""
    decay = []
    no_decay = []
    for name, parameter in params:
        if not parameter.requires_grad:
            continue
        name_lower = name.lower()
        if (
            parameter.ndim < 2
            or name_lower.endswith(".bias")
            or "norm" in name_lower
            or "embed" in name_lower
        ):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
    )
