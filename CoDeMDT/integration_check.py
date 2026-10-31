"""Integration check for the already-wired CoDeMDT + ADITcsp pipeline.

It imports the real Lightning module, the real VAE checkpoint and a real
carbon_24 batch, then exercises both training paths, diversity backpropagation,
classifier-free sampling, and CKA hooks.

Usage:
    python -B integration_check.py --vae-ckpt <path to last.ckpt>
"""

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from torch_geometric.loader import DataLoader  # noqa: E402

from src.data.crystal_datamodule import build_crystal_dataset  # noqa: E402
from src.models.denoisers.code_mdt import CoDeMDT  # noqa: E402
from src.models.interpolants.flow_matching import FlowMatchingInterpolant  # noqa: E402
from src.models.ldm_module import LatentDiffusionLitModule  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae-ckpt", required=True)
    parser.add_argument("--steps", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(0)

    denoiser = CoDeMDT(
        d_x=8,
        d_model=384,
        nhead=6,
        num_layers=6,
        num_datasets=2,
        composition_dim=100,
        composition_dropout_prob=0.1,
        dropout=0.05,
        mask_ratio=0.3,
        mask_ratio_range=0.2,
        decode_layer=2,
    )
    interpolant = FlowMatchingInterpolant(
        min_t=1e-2,
        corrupt=True,
        num_timesteps=8,
        self_condition=True,
        self_condition_prob=0.5,
    )
    module = LatentDiffusionLitModule(
        autoencoder_ckpt=args.vae_ckpt,
        denoiser=denoiser,
        interpolant=interpolant,
        optimizer=torch.optim.AdamW,
        scheduler=None,
        scheduler_frequency=10,
        mask_loss_weight=1.0,
    )
    module.eval()  # dropout off; training_step is replayed manually below
    module.autoencoder.eval()
    print(
        f"[check] denoiser params={sum(p.numel() for p in denoiser.parameters())/1e6:.2f}M "
        f"blocks={denoiser.num_blocks} cka_blocks={len(denoiser.cka_blocks)}"
    )
    assert module.masking_enabled, "masking_enabled must be True for CoDeMDT"

    dataset = build_crystal_dataset(REPO / "data" / "carbon_24", "carbon_24", "train")
    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))

    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=1e-4)
    for step in range(args.steps):
        optimizer.zero_grad()
        # --- exactly the patched training_step, minus Lightning logging ---
        noisy, cond = module._prepare_batch(batch)
        plain = module.criterion(noisy, module._denoise(noisy, cond))
        masked = module.criterion(
            noisy, module._denoise(noisy, cond, enable_mask=True)
        )
        total = plain["loss"] + masked["loss"]
        div_term = denoiser.diversity_term(step, plain["loss"].detach())
        assert div_term is not None, "diversity term missing after the masked pass"
        total = total + div_term
        # ------------------------------------------------------------------
        total.backward()
        optimizer.step()
        components = denoiser.diversity_components()
        print(
            f"[check] step {step}: loss={plain['loss'].item():.4f} "
            f"m_loss={masked['loss'].item():.4f} "
            f"div={components['div_loss']:.4f} "
            f"orth={components['loss_orth']:.4f} "
            f"mi={components['loss_mi']:.4f} "
            f"disp={components['loss_disp']:.4f} "
            f"w={denoiser.diversity_weight():.3f}"
        )
        assert torch.isfinite(total).all()

    # inference path: no mask, no SI, no diversity
    composition = batch.composition.float()[:1].repeat(2, 1)
    module.interpolant.num_timesteps = 8
    with torch.no_grad():
        out, meta, samples = module.sample_and_decode(composition, cfg_scale=2.0)
    assert torch.isfinite(out["frac_coords"]).all()
    assert torch.isfinite(out["lengths"]).all()
    assert denoiser.diversity_loss() is None, "sampling must not compute diversity"
    print(
        f"[check] sampling ok: atoms={meta['num_atoms'].tolist()} "
        f"traj={len(samples['clean_traj'])} "
        f"frac={tuple(out['frac_coords'].shape)}"
    )

    # CKA hooks resolve against ADITcsp's collector and fire on the plain pass
    collector = denoiser.register_cka_hooks()
    with torch.no_grad():
        module._denoise(*module._prepare_batch(batch))
    features = collector.latest_features()
    collector.remove()
    expected = 1 + len(denoiser.cka_blocks) + 1
    assert len(features) == expected, (sorted(features), expected)
    assert all(v.shape[1] == batch.num_atoms.max().item() for v in features.values())
    print(f"[check] cka hooks ok: {sorted(features)}")
    print("INTEGRATION_OK")


if __name__ == "__main__":
    main()
