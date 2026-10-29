"""Compute layer-wise CKA for a trained Carbon-24 DiT checkpoint."""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import rootutils
import torch
from lightning import Trainer
from torch_geometric.loader import DataLoader

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.components.carbon24_dataset import Carbon24Dataset  # noqa: E402
from src.models.components.cka import CKAVisualizationCallback  # noqa: E402
from src.models.ldm_module import LatentDiffusionLitModule  # noqa: E402


CKA_BATCH_SIZE = 32
CKA_MAX_VECTORS = 512


def parse_args():
    parser = argparse.ArgumentParser(
        description="Use forward hooks to compute layer-wise CKA for a trained DiT."
    )
    parser.add_argument("--ckpt", required=True, help="DiT checkpoint")
    parser.add_argument("--vae-ckpt", required=True, help="matching VAE checkpoint")
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default="cpu", help="evaluation device"
    )
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument(
        "--batches",
        type=int,
        default=4,
        help="number of data batches used to collect layer activations",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "runs/carbon24_cka"),
        help="directory for heatmap, matrix, and metadata",
    )
    args = parser.parse_args()
    if args.batches <= 0:
        parser.error("--batches must be positive")
    return args


def main():
    args = parse_args()
    dit_checkpoint = Path(args.ckpt)
    vae_checkpoint = Path(args.vae_ckpt)
    for checkpoint in (dit_checkpoint, vae_checkpoint):
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    device = torch.device(args.device)
    model = LatentDiffusionLitModule.load_from_checkpoint(
        dit_checkpoint,
        map_location="cpu",
        autoencoder_ckpt=str(vae_checkpoint),
    )
    model.to(device)
    model.eval()
    model.autoencoder.eval()
    model.interpolant.num_timesteps = 100

    dataset = Carbon24Dataset(ROOT / "data/carbon_24", args.split)
    loader = DataLoader(
        dataset,
        batch_size=CKA_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    callback = CKAVisualizationCallback(
        output_dir,
        max_batches=args.batches,
        max_vectors=CKA_MAX_VECTORS,
        epoch_label="standalone evaluation",
    )
    trainer = Trainer(
        accelerator="gpu" if args.device == "cuda" else "cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        num_sanity_val_steps=0,
        limit_val_batches=args.batches,
        callbacks=[callback],
    )
    trainer.validate(model, dataloaders=loader)

    heatmap_path = output_dir / "cka_heatmap.png"
    matrix_path = output_dir / "cka_matrix.csv"
    if not heatmap_path.is_file() or not matrix_path.is_file():
        raise RuntimeError("CKA evaluation finished without producing output files.")

    with matrix_path.open("r", encoding="utf-8", newline="") as handle:
        layer_names = next(csv.reader(handle))[1:]
    metadata = {
        "dit_checkpoint": str(dit_checkpoint.resolve()),
        "vae_checkpoint": str(vae_checkpoint.resolve()),
        "split": args.split,
        "batches": args.batches,
        "batch_size": CKA_BATCH_SIZE,
        "max_vectors": CKA_MAX_VECTORS,
        "seed": 0,
        "device": args.device,
        "d_model": model.denoiser.d_model,
        "nhead": model.denoiser.nhead,
        "num_blocks": len(model.denoiser.blocks),
        "layers": layer_names,
    }
    (output_dir / "cka_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(f"CKA heatmap: {heatmap_path}")
    print(f"CKA matrix: {matrix_path}")
    print(f"CKA metadata: {output_dir / 'cka_metadata.json'}")


if __name__ == "__main__":
    main()
