"""Minimal Carbon-24 CSP training entry point.

Examples:
    python src/train_csp.py --stage vae
    python src/train_csp.py --stage dit --vae-ckpt runs/carbon24_vae/checkpoints/last.ckpt
"""

import argparse
import random
from functools import partial
from pathlib import Path

import numpy as np
import rootutils
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import OmegaConf

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.carbon24_datamodule import Carbon24DataModule  # noqa: E402
from src.models.decoders.transformer import TransformerDecoder  # noqa: E402
from src.models.denoisers.dit import DiT  # noqa: E402
from src.models.encoders.transformer import TransformerEncoder  # noqa: E402
from src.models.interpolants.flow_matching import FlowMatchingInterpolant  # noqa: E402
from src.models.ldm_module import LatentDiffusionLitModule  # noqa: E402
from src.models.vae_module import VariationalAutoencoderLitModule  # noqa: E402


DIT_SIZES = {
    "s": (384, 6, 12),
    "b": (768, 12, 12),
    "l": (1024, 16, 24),
}

CONSOLE_LOG_INTERVAL = 10


class ConsoleTrainingCallback(Callback):
    """Print compact training and validation information to the console."""

    @staticmethod
    def _scalar(trainer, key):
        value = trainer.callback_metrics.get(key)
        return None if value is None else float(value.detach().cpu())

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = batch_idx + 1
        if step != 1 and step % CONSOLE_LOG_INTERVAL != 0:
            return
        loss = self._scalar(trainer, "train/loss")
        if loss is None and torch.is_tensor(outputs):
            loss = float(outputs.detach().cpu())
        if loss is None:
            return
        learning_rate = trainer.optimizers[0].param_groups[0]["lr"]
        print(
            f"[训练] epoch {trainer.current_epoch + 1}/{trainer.max_epochs} "
            f"step {trainer.global_step} loss={loss:.6f} lr={learning_rate:.2e}",
            flush=True,
        )

    def on_validation_end(self, trainer, pl_module):
        metrics = []
        for key, label in (
            ("val_mp20/loss", "loss"),
            ("val_mp20/match_rate", "match_rate"),
            ("val_mp20/rms_dist", "rms_dist"),
        ):
            value = self._scalar(trainer, key)
            if value is not None:
                metrics.append(f"{label}={value:.6f}")
        if metrics:
            print(
                f"[验证] epoch {trainer.current_epoch + 1}/{trainer.max_epochs} "
                + " ".join(metrics),
                flush=True,
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["vae", "dit"], default="vae")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--vae-ckpt", default=None)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dit-size", choices=["s", "b", "l"], default="s")
    parser.add_argument("--limit-batches", type=int, default=0, help="0 = all batches")
    return parser.parse_args()


def build_vae(args, output_dir):
    common = dict(
        max_num_elements=100,
        d_model=512,
        nhead=8,
        dim_feedforward=2048,
        activation="gelu",
        dropout=0.0,
        norm_first=True,
        bias=True,
        num_layers=8,
    )
    encoder = TransformerEncoder(**common)
    decoder = TransformerDecoder(**common)
    loss_weights = {
        "loss_atom_types": {"mp20": 1.0},
        "loss_lengths": {"mp20": 1.0},
        "loss_angles": {"mp20": 10.0},
        "loss_frac_coords": {"mp20": 10.0},
        "loss_pos": {"mp20": 0.0},
        "loss_kl": {"mp20": 1e-5},
    }
    return VariationalAutoencoderLitModule(
        encoder=encoder,
        decoder=decoder,
        latent_dim=8,
        optimizer=partial(torch.optim.AdamW, lr=1e-4, weight_decay=0.0),
        scheduler=None,
        scheduler_frequency=10,
        loss_weights=loss_weights,
        augmentations=OmegaConf.create(
            {"noise": 0.1, "frac_coords": True, "pos": True}
        ),
        visualization=OmegaConf.create({"visualize": False, "save_dir": str(output_dir)}),
        compile=False,
        composition_conditioning=True,
        num_elements=100,
        composition_dropout=0.0,
        composition_loss_weight=1.0,
    )


def build_ldm(args, output_dir):
    d_model, nhead, num_layers = DIT_SIZES[args.dit_size]
    denoiser = DiT(
        d_x=8,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        num_datasets=2,
        composition_dim=100,
        composition_dropout_prob=0.1,
    )
    interpolant = FlowMatchingInterpolant(
        min_t=1e-2,
        corrupt=True,
        num_timesteps=100,
        self_condition=True,
        self_condition_prob=0.5,
    )
    return LatentDiffusionLitModule(
        autoencoder_ckpt=args.vae_ckpt,
        denoiser=denoiser,
        interpolant=interpolant,
        optimizer=partial(torch.optim.AdamW, lr=1e-4, weight_decay=0.0),
        scheduler=None,
        scheduler_frequency=10,
        compile=False,
    )


def main():
    args = parse_args()
    if args.stage == "dit" and not args.vae_ckpt:
        raise ValueError("--vae-ckpt is required for --stage dit")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    output_dir = Path(
        args.output_dir or ROOT / f"runs/carbon24_{args.stage}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    datamodule = Carbon24DataModule(
        root=str(ROOT / "data/carbon_24"),
        batch_size={"train": args.batch_size, "val": args.batch_size, "test": args.batch_size},
        num_workers={"train": 0, "val": 0, "test": 0},
    )
    model = build_vae(args, output_dir) if args.stage == "vae" else build_ldm(args, output_dir)

    accelerator = "gpu" if args.device == "cuda" or (
        args.device == "auto" and torch.cuda.is_available()
    ) else "cpu"
    monitor = "val_mp20/match_rate" if args.stage == "vae" else "val_mp20/loss"
    checkpoint = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename=f"{args.stage}-{{epoch:03d}}",
        monitor=monitor,
        mode="max" if args.stage == "vae" else "min",
        save_last=True,
        save_top_k=1,
    )
    trainer = Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=args.epochs,
        gradient_clip_val=10.0,
        accumulate_grad_batches=1,
        limit_train_batches=args.limit_batches or 1.0,
        limit_val_batches=args.limit_batches or 1.0,
        check_val_every_n_epoch=10,
        num_sanity_val_steps=0,
        enable_progress_bar=True,
        log_every_n_steps=CONSOLE_LOG_INTERVAL,
        logger=CSVLogger(str(output_dir / "logs"), name=args.stage),
        callbacks=[checkpoint, ConsoleTrainingCallback()],
    )
    trainer.fit(model, datamodule=datamodule)
    print(f"best checkpoint: {checkpoint.best_model_path}")
    print(f"last checkpoint: {checkpoint.last_model_path}")


if __name__ == "__main__":
    main()
