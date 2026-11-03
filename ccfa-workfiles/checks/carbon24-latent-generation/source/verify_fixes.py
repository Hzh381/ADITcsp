"""Verify the two pipeline fixes without trained checkpoints."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from lightning import LightningModule

from src.models.interpolants.flow_matching import FlowMatchingInterpolant
from src.models.ldm_module import LatentDiffusionLitModule


def main():
    torch.manual_seed(0)
    interp = FlowMatchingInterpolant()

    mask = torch.arange(24)[None, :] < 6
    noise = interp._centered_gaussian(1, 24, 8, token_mask=mask)
    valid = noise[mask]
    assert torch.all(noise[~mask] == 0.0), "padding must stay zero"
    assert valid.mean(0).abs().max().item() < 1e-5, "valid tokens must be centered"

    noise_flat = interp._centered_gaussian(1, 6, 8)
    assert noise_flat.mean(1).abs().max().item() < 1e-5, "no-padding noise centered"

    ldm = LatentDiffusionLitModule.__new__(LatentDiffusionLitModule)
    LightningModule.__init__(ldm)
    ldm.autoencoder = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Dropout(0.5))
    ldm.autoencoder.requires_grad_(False)
    ldm.train()
    assert ldm.autoencoder.training is False, "frozen autoencoder must stay eval"
    assert ldm.autoencoder[1].training is False, "dropout must stay off"
    assert not any(p.requires_grad for p in ldm.autoencoder.parameters())
    print("verification passed")


if __name__ == "__main__":
    main()
