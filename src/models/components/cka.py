"""Forward-hook based CKA analysis for latent Transformer layers."""

import csv
from pathlib import Path
from typing import Mapping

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import Callback

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def linear_cka(features_x: np.ndarray, features_y: np.ndarray) -> float:
    """Return linear CKA between two feature matrices with shape ``[N, D]``."""
    if features_x.ndim != 2 or features_y.ndim != 2:
        raise ValueError("CKA inputs must be two-dimensional feature matrices.")
    if features_x.shape[0] != features_y.shape[0]:
        raise ValueError("CKA inputs must contain the same number of samples.")
    if features_x.shape[0] < 2:
        raise ValueError("CKA requires at least two samples.")

    x = np.nan_to_num(features_x.astype(np.float32, copy=True))
    y = np.nan_to_num(features_y.astype(np.float32, copy=True))
    x -= x.mean(axis=0, keepdims=True)
    y -= y.mean(axis=0, keepdims=True)

    covariance_xy = x.T @ y
    covariance_xx = x.T @ x
    covariance_yy = y.T @ y
    denominator = np.linalg.norm(covariance_xx) * np.linalg.norm(covariance_yy)
    if denominator <= 1e-12:
        return 0.0
    similarity = np.square(np.linalg.norm(covariance_xy)) / denominator
    return float(np.clip(similarity, 0.0, 1.0))


def compute_cka_matrix(features: Mapping[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    """Compute a symmetric layer-by-layer linear CKA matrix."""
    names = list(features)
    matrix = np.eye(len(names), dtype=np.float32)
    for row, name_x in enumerate(names):
        for col in range(row + 1, len(names)):
            similarity = linear_cka(features[name_x], features[names[col]])
            matrix[row, col] = similarity
            matrix[col, row] = similarity
    return names, matrix


class CKAFeatureCollector:
    """Capture module outputs through PyTorch forward hooks."""

    def __init__(self, modules: Mapping[str, nn.Module]):
        self._current = {}
        self._handles = [
            module.register_forward_hook(self._hook(name))
            for name, module in modules.items()
        ]

    def _hook(self, name):
        def capture(module, inputs, output):
            value = output[0] if isinstance(output, (tuple, list)) else output
            if torch.is_tensor(value):
                self._current[name] = value

        return capture

    def begin_capture(self):
        self._current = {}

    def latest_features(self) -> dict[str, torch.Tensor]:
        return dict(self._current)

    def remove(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []


class CKAVisualizationCallback(Callback):
    """Collect DiT activations during validation and save a CKA heatmap."""

    def __init__(
        self,
        output_dir: Path,
        max_batches: int = 4,
        max_vectors: int = 512,
        epoch_label: str | None = None,
    ):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_batches = max_batches
        self.max_vectors = max_vectors
        self.epoch_label = epoch_label
        self._collector = None
        self._chunks = {}
        self._captured_batches = 0

    def on_validation_start(self, trainer, pl_module):
        denoiser = getattr(pl_module, "denoiser", None)
        if denoiser is None or not hasattr(denoiser, "register_cka_hooks"):
            return
        self._collector = denoiser.register_cka_hooks()
        self._chunks = {}
        self._captured_batches = 0

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ):
        if self._collector is not None and self._captured_batches < self.max_batches:
            self._collector.begin_capture()

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        if self._collector is None or self._captured_batches >= self.max_batches:
            return
        activations = self._collector.latest_features()
        if not activations:
            return

        lengths = batch.num_atoms.reshape(-1)
        for name, activation in activations.items():
            if activation.ndim != 3:
                continue
            device = activation.device
            lengths_dev = lengths.to(device)
            token_mask = torch.arange(
                activation.shape[1], device=device
            ).unsqueeze(0) < lengths_dev.unsqueeze(1)
            chunk = activation[token_mask].detach().float().cpu().numpy()
            self._chunks.setdefault(name, []).append(chunk)
        self._captured_batches += 1

    def on_validation_end(self, trainer, pl_module):
        if self._collector is not None:
            self._collector.remove()
            self._collector = None
        if not self._chunks:
            return

        features = {}
        total_vectors = min(
            sum(part.shape[0] for part in parts)
            for parts in self._chunks.values()
        )
        if total_vectors < 2:
            return
        indices = np.unique(
            np.linspace(
                0,
                total_vectors - 1,
                num=min(total_vectors, self.max_vectors),
                dtype=np.int64,
            )
        )
        for name, parts in self._chunks.items():
            values = np.concatenate(parts, axis=0)
            features[name] = values[indices]

        names, matrix = compute_cka_matrix(features)
        self._save_matrix(names, matrix)
        self._save_heatmap(names, matrix, trainer.current_epoch + 1)
        print(
            f"[CKA] epoch {trainer.current_epoch + 1} 层间 CKA 热力图已保存: "
            f"{self.output_dir / 'cka_heatmap.png'}",
            flush=True,
        )

    def on_exception(self, trainer, pl_module, exception):
        if self._collector is not None:
            self._collector.remove()
            self._collector = None

    def _save_matrix(self, names, matrix):
        with (self.output_dir / "cka_matrix.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["layer", *names])
            for name, row in zip(names, matrix):
                writer.writerow([name, *[f"{value:.6f}" for value in row]])

    def _save_heatmap(self, names, matrix, epoch):
        size = max(7.5, len(names) * 0.45 + 3.5)
        figure, axis = plt.subplots(figsize=(size, size))
        image = axis.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)
        colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        colorbar.set_label("Linear CKA")
        axis.set_xticks(np.arange(len(names)), labels=names, rotation=45, ha="right")
        axis.set_yticks(np.arange(len(names)), labels=names)
        axis.tick_params(axis="both", labelsize=7)
        label = self.epoch_label or f"Epoch {epoch}"
        axis.set_title(f"DiT Layer-wise CKA ({label})")

        if len(names) <= 16:
            for row in range(len(names)):
                for col in range(len(names)):
                    color = "white" if matrix[row, col] < 0.55 else "black"
                    axis.text(
                        col,
                        row,
                        f"{matrix[row, col]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color=color,
                    )

        figure.tight_layout()
        figure.savefig(
            self.output_dir / "cka_heatmap.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(figure)
