"""Minimal Carbon-24 CSP training entry point.

Examples:
    python src/train_csp.py --stage vae
    python src/train_csp.py --stage dit --vae-ckpt runs/carbon24_vae/checkpoints/last.ckpt
"""

import argparse
import csv
import os
import random
from functools import partial
from pathlib import Path

import matplotlib
import numpy as np
import rootutils
import torch
from lightning import Trainer
from lightning.fabric.plugins.io.torch_io import TorchCheckpointIO
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger
from omegaconf import OmegaConf

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.carbon24_datamodule import Carbon24DataModule  # noqa: E402
from src.models.decoders.transformer import TransformerDecoder  # noqa: E402
from src.models.denoisers.dit import DiT  # noqa: E402
from src.models.encoders.transformer import TransformerEncoder  # noqa: E402
from src.models.interpolants.flow_matching import FlowMatchingInterpolant  # noqa: E402
from src.models.ldm_module import LatentDiffusionLitModule  # noqa: E402
from src.models.components.cka import CKAVisualizationCallback  # noqa: E402
from src.models.vae_module import VariationalAutoencoderLitModule  # noqa: E402


DIT_SIZES = {
    "s": {"d_model": 384, "nhead": 6, "num_layers": 12},
    "b": {"d_model": 768, "nhead": 12, "num_layers": 12},
    "l": {"d_model": 1024, "nhead": 16, "num_layers": 24},
}

CONSOLE_LOG_INTERVAL = 10
VALIDATION_INTERVAL = 10
EARLY_STOPPING_MIN_DELTA = 1e-4
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 2
LR_SCHEDULER_THRESHOLD = 1e-4
MIN_LEARNING_RATE = 1e-6


def build_lr_scheduler():
    return partial(
        torch.optim.lr_scheduler.ReduceLROnPlateau,
        mode="min",
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
        threshold=LR_SCHEDULER_THRESHOLD,
        threshold_mode="rel",
        min_lr=MIN_LEARNING_RATE,
    )


class AtomicCheckpointIO(TorchCheckpointIO):
    """用同目录临时文件 + os.replace 覆盖写入 checkpoint。

    Lightning 默认走 fsspec 原子写：临时文件落在系统盘，提交时跨盘 move 会失败；
    目标文件已存在时 Windows 的 rename 也不能覆盖。
    """

    def save_checkpoint(self, checkpoint, path, storage_options=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(path.name + ".tmp")
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, path)


class ConsoleTrainingCallback(Callback):
    """Print training information and save loss curves."""

    def __init__(self, early_stopping_patience: int, output_dir: Path, stage: str):
        super().__init__()
        self.early_stopping_patience = early_stopping_patience
        self.output_dir = Path(output_dir)
        self.stage = stage
        self.last_learning_rate = None
        self.train_epochs = []
        self.train_losses = []
        self.val_epochs = []
        self.val_losses = []
        self.epoch_loss_sum = 0.0
        self.epoch_loss_count = 0

    @staticmethod
    def _scalar(trainer, key):
        value = trainer.callback_metrics.get(key)
        return None if value is None else float(value.detach().cpu())

    @staticmethod
    def _output_loss(outputs):
        value = outputs.get("loss") if isinstance(outputs, dict) else outputs
        if torch.is_tensor(value) and value.numel() == 1:
            return float(value.detach().cpu())
        return None

    def on_train_start(self, trainer, pl_module):
        self.last_learning_rate = trainer.optimizers[0].param_groups[0]["lr"]
        self.train_epochs = []
        self.train_losses = []
        self.val_epochs = []
        self.val_losses = []
        self.epoch_loss_sum = 0.0
        self.epoch_loss_count = 0
        if trainer.current_epoch > 0:
            self._load_history_until(trainer.current_epoch)
        print(
            "[配置] 提前停止监控 val_mp20/loss；"
            f"连续 {self.early_stopping_patience} 次验证未改善后停止 "
            f"(约 {self.early_stopping_patience * VALIDATION_INTERVAL} 个 epoch)，"
            f"min_delta={EARLY_STOPPING_MIN_DELTA}",
            flush=True,
        )
        print(
            "[配置] 学习率动态调整监控 val_mp20/loss；"
            f"允许连续 {LR_SCHEDULER_PATIENCE} 次验证未改善，"
            "再次未改善时乘以 "
            f"{LR_SCHEDULER_FACTOR}，最低学习率 {MIN_LEARNING_RATE:.1e}",
            flush=True,
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = self._output_loss(outputs)
        if loss is None:
            loss = self._scalar(trainer, "train/loss")
        if loss is not None:
            self.epoch_loss_sum += loss
            self.epoch_loss_count += 1

        step = batch_idx + 1
        if step != 1 and step % CONSOLE_LOG_INTERVAL != 0:
            return
        if loss is None:
            return
        learning_rate = trainer.optimizers[0].param_groups[0]["lr"]
        if (
            self.last_learning_rate is not None
            and learning_rate != self.last_learning_rate
        ):
            print(
                f"[学习率] {self.last_learning_rate:.2e} -> {learning_rate:.2e}",
                flush=True,
            )
        self.last_learning_rate = learning_rate
        print(
            f"[训练] epoch {trainer.current_epoch + 1}/{trainer.max_epochs} "
            f"step {trainer.global_step} loss={loss:.6f} lr={learning_rate:.2e}",
            flush=True,
        )

    def on_train_epoch_end(self, trainer, pl_module):
        if self.epoch_loss_count == 0:
            return
        self.train_epochs.append(trainer.current_epoch + 1)
        self.train_losses.append(self.epoch_loss_sum / self.epoch_loss_count)
        self.epoch_loss_sum = 0.0
        self.epoch_loss_count = 0

    def on_validation_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        val_loss = self._scalar(trainer, "val_mp20/loss")
        if val_loss is not None:
            if self.val_epochs and self.val_epochs[-1] == epoch:
                self.val_losses[-1] = val_loss
            else:
                self.val_epochs.append(epoch)
                self.val_losses.append(val_loss)

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

    def on_train_end(self, trainer, pl_module):
        self._save_loss_history()
        self._plot_loss_curve()

    def _load_history_until(self, epoch):
        history_path = self.output_dir / "loss_history.csv"
        if not history_path.exists():
            return
        with history_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    row_epoch = int(row["epoch"])
                except (KeyError, TypeError, ValueError):
                    continue
                if row_epoch > epoch:
                    continue
                if row.get("train_loss"):
                    self.train_epochs.append(row_epoch)
                    self.train_losses.append(float(row["train_loss"]))
                if row.get("val_loss"):
                    self.val_epochs.append(row_epoch)
                    self.val_losses.append(float(row["val_loss"]))

    def _save_loss_history(self):
        epochs = sorted(set(self.train_epochs) | set(self.val_epochs))
        if not epochs:
            return
        train_map = dict(zip(self.train_epochs, self.train_losses))
        val_map = dict(zip(self.val_epochs, self.val_losses))
        history_path = self.output_dir / "loss_history.csv"
        with history_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["epoch", "train_loss", "val_loss"]
            )
            writer.writeheader()
            for epoch in epochs:
                writer.writerow(
                    {
                        "epoch": epoch,
                        "train_loss": train_map.get(epoch, ""),
                        "val_loss": val_map.get(epoch, ""),
                    }
                )

    def _plot_loss_curve(self):
        if not self.train_epochs and not self.val_epochs:
            return
        figure, axis = plt.subplots(figsize=(8, 5))
        if self.train_epochs:
            axis.plot(
                self.train_epochs,
                self.train_losses,
                color="#2563EB",
                linewidth=1.8,
                label="Train loss",
            )
        if self.val_epochs:
            axis.plot(
                self.val_epochs,
                self.val_losses,
                color="#DC2626",
                linewidth=1.8,
                marker="o",
                markersize=4,
                label="Validation loss",
            )
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.set_title(f"{self.stage.upper()} Loss vs Epoch")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        plot_path = self.output_dir / "loss_curve.png"
        figure.savefig(plot_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        print(f"[图表] loss 曲线已保存: {plot_path}", flush=True)


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
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="连续多少次验证指标未改善后停止训练",
    )
    parser.add_argument("--resume", default=None, help="checkpoint path to resume training from")
    args = parser.parse_args()
    if args.patience < 0:
        parser.error("--patience must be non-negative")
    return args


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
        scheduler=build_lr_scheduler(),
        scheduler_frequency=VALIDATION_INTERVAL,
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
    size = DIT_SIZES[args.dit_size]
    denoiser = DiT(
        d_x=8,
        d_model=size["d_model"],
        nhead=size["nhead"],
        num_layers=size["num_layers"],
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
        scheduler=build_lr_scheduler(),
        scheduler_frequency=VALIDATION_INTERVAL,
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
    early_stopping = EarlyStopping(
        monitor="val_mp20/loss",
        mode="min",
        patience=args.patience,
        min_delta=EARLY_STOPPING_MIN_DELTA,
        verbose=True,
        check_finite=True,
    )
    learning_rate_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks = [
        checkpoint,
        early_stopping,
        learning_rate_monitor,
        ConsoleTrainingCallback(args.patience, output_dir, args.stage),
    ]
    if args.stage == "dit":
        callbacks.append(CKAVisualizationCallback(output_dir))
    trainer = Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=args.epochs,
        gradient_clip_val=10.0,
        accumulate_grad_batches=1,
        limit_train_batches=args.limit_batches or 1.0,
        limit_val_batches=args.limit_batches or 1.0,
        check_val_every_n_epoch=VALIDATION_INTERVAL,
        num_sanity_val_steps=0,
        enable_progress_bar=True,
        log_every_n_steps=CONSOLE_LOG_INTERVAL,
        logger=CSVLogger(str(output_dir / "logs"), name=args.stage),
        callbacks=callbacks,
        plugins=[AtomicCheckpointIO()],
    )
    trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume)
    print(f"best checkpoint: {checkpoint.best_model_path}")
    print(f"last checkpoint: {checkpoint.last_model_path}")


if __name__ == "__main__":
    main()
