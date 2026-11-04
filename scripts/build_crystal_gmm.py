"""Fit Crystal-MM-FM mode priors from the training split only.

Example:
    python scripts/build_crystal_gmm.py \
        --dataset carbon_24 \
        --vae-ckpt runs/carbon_24_hilo_vae/checkpoints/last.ckpt \
        --num-modes 8 \
        --output runs/carbon_24_hilo_mmfm_prior.pt
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import rootutils
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.components.composition_utils import composition_from_batch  # noqa: E402
from src.data.crystal_datamodule import (  # noqa: E402
    SUPPORTED_DATASETS,
    build_crystal_dataset,
)
from src.models.crystal_mmfm_prior import (  # noqa: E402
    CrystalMMFMPrior,
    build_global_descriptor,
    composition_key,
    estimate_token_priors,
    fit_global_gmm,
)
from src.models.vae_module import VariationalAutoencoderLitModule  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Build Crystal-MM-FM GMM prior.")
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, default="carbon_24")
    parser.add_argument("--vae-ckpt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-modes", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--rdf-bins", type=int, default=16)
    parser.add_argument("--rdf-max", type=float, default=6.0)
    parser.add_argument(
        "--composition-smoothing",
        type=float,
        default=1e-3,
        help="uniform mixture smoothing for per-composition mode weights",
    )
    parser.add_argument("--limit", type=int, default=0, help="0 uses the full train split")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["autodetect", "cpu", "cuda"], default="autodetect")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_modes < 1:
        raise ValueError("--num-modes must be positive.")
    if args.rdf_bins < 1:
        raise ValueError("--rdf-bins must be positive.")
    if not 0.0 <= args.composition_smoothing < 1.0:
        raise ValueError("--composition-smoothing must be in [0, 1).")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "autodetect":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    dataset = build_crystal_dataset(
        ROOT / "data" / args.dataset,
        args.dataset,
        "train",
    )
    if args.limit > 0:
        dataset = torch.utils.data.Subset(dataset, range(min(args.limit, len(dataset))))
    if len(dataset) < args.num_modes:
        raise ValueError(
            f"Need at least --num-modes samples; got {len(dataset)} "
            f"samples and {args.num_modes} modes."
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    vae = VariationalAutoencoderLitModule.load_from_checkpoint(
        args.vae_ckpt,
        map_location="cpu",
    )
    vae.to(device)
    vae.eval()

    descriptors = []
    token_means = []
    token_vars = []
    token_counts = []
    compositions = []
    material_ids = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            encoded = vae.encode(batch)
            posterior = encoded["posterior"]
            mean_dense, mask = to_dense_batch(posterior.mean, encoded["batch"])
            var_dense, _ = to_dense_batch(posterior.var, encoded["batch"])
            std_dense = torch.sqrt(var_dense.clamp_min(0.0))
            frac_dense, _ = to_dense_batch(batch.frac_coords, batch.batch)
            descriptor = build_global_descriptor(
                mean_dense,
                std_dense,
                batch.lengths,
                batch.angles_radians,
                frac_dense,
                batch.cell,
                mask,
                num_rdf_bins=args.rdf_bins,
                r_max=args.rdf_max,
            )
            descriptors.append(descriptor.cpu())
            token_means.append(mean_dense[mask].cpu())
            token_vars.append(var_dense[mask].cpu())
            token_counts.append(mask.sum(dim=-1).cpu())
            compositions.append(
                composition_from_batch(batch, vae.num_elements).cpu()
            )
            material_ids.extend(list(batch.id))

    descriptor = torch.cat(descriptors, dim=0)
    normalized, responsibilities, model, descriptor_mean, descriptor_std = fit_global_gmm(
        descriptor,
        args.num_modes,
        seed=args.seed,
    )

    valid_means = torch.cat(token_means, dim=0)
    valid_variances = torch.cat(token_vars, dim=0)
    token_count = torch.cat(token_counts, dim=0)
    mode_means, mode_vars, denominator = estimate_token_priors(
        valid_means,
        valid_variances,
        token_count,
        responsibilities,
    )

    composition_sums = defaultdict(lambda: torch.zeros(args.num_modes))
    all_compositions = torch.cat(compositions, dim=0)
    for index, material_id in enumerate(material_ids):
        composition_sums[composition_key(all_compositions[index])] += responsibilities[index]
    composition_mode_weights = {}
    for key, weights in composition_sums.items():
        normalized = weights / weights.sum().clamp_min(1e-8)
        composition_mode_weights[key] = (
            (1.0 - args.composition_smoothing) * normalized
            + args.composition_smoothing / args.num_modes
        )

    responsibility_map = {
        str(material_id): responsibilities[index]
        for index, material_id in enumerate(material_ids)
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    prior = CrystalMMFMPrior(
        mode_means=mode_means,
        mode_vars=mode_vars,
        composition_mode_weights=composition_mode_weights,
        responsibilities=responsibility_map,
        metadata={
            "dataset": args.dataset,
            "num_modes": args.num_modes,
            "descriptor_dim": int(descriptor.size(1)),
            "rdf_bins": args.rdf_bins,
            "rdf_max": args.rdf_max,
            "seed": args.seed,
            "vae_ckpt": str(args.vae_ckpt),
            "train_size": len(dataset),
            "descriptor_mean": descriptor_mean.tolist(),
            "descriptor_std": descriptor_std.tolist(),
            "gmm_weights": model.weights_.tolist(),
        },
    )
    prior.save(output)

    summary = {
        "dataset": args.dataset,
        "train_size": len(dataset),
        "num_modes": args.num_modes,
        "descriptor_dim": int(descriptor.size(1)),
        "mode_token_mass": denominator.tolist(),
        "composition_keys": len(composition_mode_weights),
        "output": str(output),
    }
    summary_path = output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
