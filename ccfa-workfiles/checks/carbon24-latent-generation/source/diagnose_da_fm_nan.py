"""Reproduce decoder-aware loss stability on a tiny CPU model."""

from functools import partial
from pathlib import Path
import runpy
import tempfile

import rootutils
import torch

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.decoder_struct_loss import compute_cell_loss, compute_periodic_loss
from src.models.ldm_module import LatentDiffusionLitModule


def main():
    helpers = runpy.run_path("tests/test_decoder_aware_and_mmfm.py")
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "vae.ckpt"
        helpers["_checkpoint"](checkpoint)
        module = LatentDiffusionLitModule(
            autoencoder_ckpt=str(checkpoint),
            denoiser=helpers["_denoiser"](),
            interpolant=helpers["FlowMatchingInterpolant"](
                num_timesteps=2,
                self_condition=False,
            ),
            optimizer=partial(torch.optim.AdamW, lr=1e-4),
            scheduler=None,
            scheduler_frequency=10,
            decoder_aware=True,
            cell_loss_weight=0.1,
            geo_loss_weight=0.1,
        )
        optimizer = torch.optim.AdamW(module.denoiser.parameters(), lr=1e-4)
        for step in range(10):
            noisy, conditioning = module._prepare_batch(helpers["_batch"]())
            pred = module._denoise(noisy, conditioning)
            plain = module.criterion(noisy, pred)
            structure = module._decoder_structure_loss(pred, conditioning, noisy)
            if step == 0:
                mask = conditioning["mask"]
                num_atoms = conditioning["num_atoms"]
                flat_batch = torch.repeat_interleave(
                    torch.arange(pred.size(0)), num_atoms
                )
                flat = {
                    "x": pred[mask],
                    "num_atoms": num_atoms,
                    "batch": flat_batch,
                    "token_idx": (
                        torch.cumsum(mask, dim=-1, dtype=torch.int64) - 1
                    )[mask],
                    "composition": conditioning["composition"],
                }
                decoded = module.autoencoder.decode(flat)
                print("decoded lengths", decoded["lengths"].detach())
                print("decoded angles", decoded["angles"].detach())
                print(
                    "frac minmax",
                    float(decoded["frac_coords"].min()),
                    float(decoded["frac_coords"].max()),
                )
                from torch_geometric.utils import to_dense_batch

                pred_frac, _ = to_dense_batch(
                    decoded["frac_coords"],
                    flat_batch,
                    max_num_nodes=mask.size(1),
                )
                scale = num_atoms.to(pred.dtype).pow(1.0 / 3.0)
                cell_only = compute_cell_loss(
                    decoded["lengths"] * scale[:, None],
                    decoded["angles"],
                    noisy["target_lengths"],
                    noisy["target_angles"],
                )
                geo_only = compute_periodic_loss(
                    pred_frac,
                    decoded["lengths"] * scale[:, None],
                    decoded["angles"],
                    noisy["target_frac"],
                    noisy["target_lengths"],
                    noisy["target_angles"],
                    mask,
                )
                for name, value in (("cell", cell_only), ("geo", geo_only)):
                    value.backward(retain_graph=True)
                    grads = [
                        p.grad.detach()
                        for p in module.denoiser.parameters()
                        if p.grad is not None
                    ]
                    print(
                        name,
                        float(value),
                        "grad_max",
                        max(float(g.abs().max()) for g in grads),
                        "grad_nan",
                        any(not torch.isfinite(g).all() for g in grads),
                    )
                    module.denoiser.zero_grad(set_to_none=True)
            total = (
                plain["loss"]
                + 0.1 * structure["cell_loss"]
                + 0.1 * structure["geo_loss"]
            )
            optimizer.zero_grad()
            total.backward()
            grads = [
                p.grad.detach()
                for p in module.denoiser.parameters()
                if p.grad is not None
            ]
            grad_max = max(float(g.abs().max()) for g in grads)
            grad_nan = any(not torch.isfinite(g).all() for g in grads)
            print(
                step,
                "loss", float(total),
                "plain", float(plain["loss"]),
                "cell", float(structure["cell_loss"]),
                "geo", float(structure["geo_loss"]),
                "grad_max", grad_max,
                "grad_nan", grad_nan,
                "pred_absmax", float(pred.abs().max()),
            )
            optimizer.step()
            finite = all(
                torch.isfinite(p).all() for p in module.denoiser.parameters()
            )
            print("  params_finite", bool(finite))


if __name__ == "__main__":
    main()
