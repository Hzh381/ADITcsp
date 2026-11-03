"""Reproduce pipeline issues without loading or changing trained checkpoints.

Run from the repository root with: python -B <path-to-this-file>
The synthetic checks verify implementation behavior, not generation accuracy.
"""

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from lightning import LightningModule

from src.data.components.carbon24_dataset import Carbon24Dataset
from src.models.components.posterior_mmd import PosteriorMMD
from src.models.interpolants.flow_matching import FlowMatchingInterpolant
from src.models.ldm_module import LatentDiffusionLitModule


def main():
    torch.manual_seed(0)
    torch.set_num_threads(2)
    result = {"scope": "Synthetic implementation checks; no trained checkpoints used."}

    # Use the actual inherited train() on the actual LDM class. Constructor is
    # bypassed only to avoid requiring a checkpoint for this mode-recursion test.
    ldm = LatentDiffusionLitModule.__new__(LatentDiffusionLitModule)
    LightningModule.__init__(ldm)
    ldm.autoencoder = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Dropout(0.05))
    ldm.autoencoder.requires_grad_(False)
    ldm.autoencoder.eval()
    before = ldm.autoencoder.training
    ldm.train()
    result["frozen_vae_train_mode"] = {
        "train_implementation": LatentDiffusionLitModule.train.__qualname__,
        "before_parent_train": before,
        "after_parent_train": ldm.autoencoder.training,
        "dropout_after_parent_train": ldm.autoencoder[1].training,
        "parameters_require_grad": any(p.requires_grad for p in ldm.autoencoder.parameters()),
    }

    interpolant = FlowMatchingInterpolant()
    noise = interpolant._centered_gaussian(20000, 24, 8)
    result["noise_mean_variance"] = {}
    for n in (6, 12, 24):
        padded_train_mean = noise[:, :n].mean(1)
        no_padding_test_mean = interpolant._centered_gaussian(20000, n, 8).mean(1)
        result["noise_mean_variance"][str(n)] = {
            "training_M24": padded_train_mean.var(0).mean().item(),
            "theoretical_training": 1.0 / n - 1.0 / 24,
            "test_M_equals_N": no_padding_test_mean.var(0).mean().item(),
        }

    # A huge change to token arrangement which preserves each crystal's mean.
    mask = torch.arange(6)[None, :] < torch.tensor([6, 4, 5, 3])[:, None]
    posterior = torch.randn(4, 6, 8) * mask[..., None]
    delta = torch.randn_like(posterior) * mask[..., None]
    delta -= (delta.sum(1) / mask.sum(1)[:, None])[:, None, :]
    generated = posterior + 10 * delta * mask[..., None]
    mmd = PosteriorMMD()
    result["mixed_length_mmd"] = {
        "valid_token_rmse": (generated - posterior)[mask].square().mean().sqrt().item(),
        "pooled_feature_max_error": (mmd._features(generated, mask) - mmd._features(posterior, mask)).abs().max().item(),
        "mmd": mmd(generated, posterior, mask).item(),
    }

    # For a median-adaptive RBF kernel, rescaling both samples should preserve
    # relative distances. The current bandwidth convention does not do that.
    base_posterior = torch.randn(16, 8)
    base_generated = torch.randn(16, 8) + 1.0
    result["mmd_joint_rescaling"] = {
        str(scale): mmd(base_generated * scale, base_posterior * scale).item()
        for scale in (0.1, 1.0, 10.0, 100.0)
    }

    result["local_dataset"] = {}
    for split in ("train", "val", "test"):
        dataset = Carbon24Dataset(ROOT / "data/carbon_24", split)
        result["local_dataset"][split] = {
            "count": len(dataset),
            "atom_counts": dict(sorted(Counter(r["num_atoms"] for r in dataset.records).items())),
            "elements": sorted({int(z) for r in dataset.records for z in r["atom_types"]}),
        }

    # Confirm mixed lengths in typical training batches: MMD defaults to the
    # first 16 examples of an ordinary shuffled batch of 32.
    dataset = Carbon24Dataset(ROOT / "data/carbon_24", "train")
    counts = torch.tensor([r["num_atoms"] for r in dataset.records])
    indices = torch.randperm(len(counts))[: len(counts) // 32 * 32].reshape(-1, 32)
    groups = counts[indices]
    result["local_training_batch_example"] = {
        "seed": 0,
        "batches": len(groups),
        "fraction_of_atoms_counts_below_batch_max": (groups < groups.max(1).values[:, None]).float().mean().item(),
        "fraction_mixed_lengths_in_first_16": (groups[:, :16].min(1).values != groups[:, :16].max(1).values).float().mean().item(),
    }
    output = Path(__file__).resolve().parents[1] / "mechanics.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
