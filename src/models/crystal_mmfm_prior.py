"""Crystal-MM-FM: crystal-level GMM modes as the flow-matching source.

The prior is fitted only on training crystals.  A crystal-level descriptor
combines the VAE posterior mean/std, normalized cell parameters and a periodic
radial histogram.  Each mixture component defines a token-level diagonal
Gaussian source ``N(m_k, diag(s_k^2))``; all valid tokens of one crystal share
the same sampled mode, while their noise is drawn independently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch

from src.models.decoder_struct_loss import periodic_pair_distances


def composition_key(composition_row: torch.Tensor | list[float]) -> str:
    """Stable key for a composition count vector."""
    if isinstance(composition_row, torch.Tensor):
        values = composition_row.detach().cpu().tolist()
    else:
        values = list(composition_row)
    return ",".join(str(int(round(float(value)))) for value in values)


def build_global_descriptor(
    mean_latent: torch.Tensor,
    std_latent: torch.Tensor,
    lengths: torch.Tensor,
    angles_radians: torch.Tensor,
    frac_coords: torch.Tensor,
    cell: torch.Tensor,
    token_mask: torch.Tensor,
    num_rdf_bins: int = 16,
    r_max: float = 6.0,
) -> torch.Tensor:
    """Build one crystal-level descriptor per graph.

    Args:
        mean_latent/std_latent: ``[B, N, D]`` dense posterior statistics.
        lengths: ``[B, 3]`` actual lattice lengths.
        angles_radians: ``[B, 3]`` lattice angles in radians.
        frac_coords: ``[B, N, 3]`` fractional coordinates.
        cell: ``[B, 3, 3]`` lattice matrices (vectors as rows).
        token_mask: ``[B, N]`` valid-atom mask.
    """
    if mean_latent.shape != std_latent.shape:
        raise ValueError("mean_latent and std_latent must share a shape.")
    if frac_coords.shape[:2] != token_mask.shape:
        raise ValueError("frac_coords and token_mask shapes disagree.")

    eps = 1e-6
    mask = token_mask.to(dtype=mean_latent.dtype).unsqueeze(-1)
    counts = token_mask.sum(dim=-1).clamp_min(1).to(mean_latent.dtype)
    mean = (mean_latent * mask).sum(dim=1) / counts.unsqueeze(-1)
    var = ((mean_latent - mean[:, None, :]).square() * mask).sum(dim=1) / counts.unsqueeze(
        -1
    )
    std = torch.sqrt(var.clamp_min(0.0))

    lengths_scaled = lengths / counts.unsqueeze(-1).pow(1.0 / 3.0)
    log_lengths = torch.log(lengths_scaled.clamp_min(eps))
    cos_angles = torch.cos(angles_radians)
    cell_features = torch.cat([log_lengths, cos_angles], dim=-1)
    rdf_features = periodic_distance_histogram(
        frac_coords,
        cell,
        token_mask,
        num_bins=num_rdf_bins,
        r_max=r_max,
    )
    return torch.cat([mean, std, cell_features, rdf_features], dim=-1)


def periodic_distance_histogram(
    frac_coords: torch.Tensor,
    cell: torch.Tensor,
    token_mask: torch.Tensor,
    num_bins: int,
    r_max: float,
) -> torch.Tensor:
    """Normalized periodic pair-distance histogram for each graph."""
    distances = periodic_pair_distances(frac_coords, cell)
    n = token_mask.size(1)
    index = torch.arange(n, device=token_mask.device)
    pairs = (token_mask[:, :, None] & token_mask[:, None, :]) & (
        index[:, None] < index[None, :]
    ).unsqueeze(0)

    histograms = []
    for batch_index in range(frac_coords.size(0)):
        pair_distances = (
            distances[batch_index][pairs[batch_index]].detach().float().cpu()
        )
        if pair_distances.numel() == 0:
            histograms.append(
                torch.zeros(num_bins, dtype=frac_coords.dtype)
            )
            continue
        bins = (
            (pair_distances / max(float(r_max), 1e-6)) * num_bins
        ).floor().long().clamp(0, num_bins - 1)
        histogram = torch.bincount(bins, minlength=num_bins).to(frac_coords.dtype)
        histograms.append(histogram / histogram.sum().clamp_min(1.0))
    return torch.stack(histograms, dim=0).to(frac_coords.device)


def balanced_mode_assignment(weights: torch.Tensor) -> torch.Tensor:
    """Allocate modes proportionally with largest-remainder stratification."""
    batch_size, num_modes = weights.shape
    average = weights.mean(dim=0).clamp_min(0.0)
    total = average.sum()
    if total <= 0:
        average = torch.full_like(average, 1.0 / max(num_modes, 1))
    else:
        average = average / total

    raw = average * batch_size
    counts = raw.floor().long()
    remainder = int(batch_size - counts.sum().item())
    if remainder > 0:
        fractions = raw - counts
        top = torch.topk(fractions, k=remainder).indices
        counts[top] += 1
    modes = torch.repeat_interleave(torch.arange(num_modes, device=weights.device), counts)
    if modes.numel() != batch_size:
        modes = modes[:batch_size]
    return modes[torch.randperm(modes.numel(), device=weights.device)]


def fit_global_gmm(
    descriptors: torch.Tensor,
    num_modes: int,
    seed: int = 0,
    n_init: int = 3,
    reg_covar: float = 1e-5,
):
    """Fit a diagonal-covariance GMM and return soft responsibilities."""
    from sklearn.mixture import GaussianMixture

    if descriptors.dim() != 2:
        raise ValueError("descriptors must have shape [num_crystals, dim].")
    if num_modes < 1 or descriptors.size(0) < num_modes:
        raise ValueError("num_modes must be positive and no larger than samples.")
    mean = descriptors.mean(dim=0)
    std = descriptors.std(dim=0).clamp_min(1e-6)
    normalized = (descriptors - mean) / std
    model = GaussianMixture(
        n_components=num_modes,
        covariance_type="diag",
        random_state=seed,
        n_init=n_init,
        reg_covar=reg_covar,
    )
    model.fit(normalized.detach().cpu().numpy())
    responsibilities = torch.as_tensor(
        model.predict_proba(normalized.detach().cpu().numpy()),
        dtype=torch.float32,
    )
    return normalized, responsibilities, model, mean, std


def estimate_token_priors(
    valid_means: torch.Tensor,
    valid_variances: torch.Tensor,
    token_counts: torch.Tensor,
    responsibilities: torch.Tensor,
):
    """Estimate per-mode token means and variances from soft assignments."""
    token_responsibilities = torch.repeat_interleave(
        responsibilities,
        token_counts,
        dim=0,
    )
    denominator = token_responsibilities.sum(dim=0).clamp_min(1e-8)
    mode_means = (
        token_responsibilities.transpose(0, 1) @ valid_means
    ) / denominator.unsqueeze(-1)
    squared_deviation = (
        valid_means[:, None, :] - mode_means[None, :, :]
    ).square() + valid_variances[:, None, :]
    mode_vars = (
        token_responsibilities[..., None] * squared_deviation
    ).sum(dim=0) / denominator[:, None]
    empty_modes = denominator <= 1e-6
    if empty_modes.any():
        global_mean = valid_means.mean(dim=0)
        global_var = valid_variances.mean(dim=0) + valid_means.var(dim=0)
        mode_means[empty_modes] = global_mean
        mode_vars[empty_modes] = global_var
    return mode_means, mode_vars, denominator


class CrystalMMFMPrior:
    """Container for GMM modes and the corresponding token-level sources."""

    def __init__(
        self,
        mode_means: torch.Tensor,
        mode_vars: torch.Tensor,
        composition_mode_weights: Mapping[str, torch.Tensor],
        responsibilities: Mapping[str, torch.Tensor] | None = None,
        metadata: Mapping | None = None,
    ) -> None:
        if mode_means.shape != mode_vars.shape:
            raise ValueError("mode_means and mode_vars must share a shape.")
        self.mode_means = mode_means.detach().cpu().float()
        self.mode_vars = mode_vars.detach().cpu().float().clamp_min(0.0)
        self.composition_mode_weights = {
            str(key): torch.as_tensor(value, dtype=torch.float32).detach().cpu()
            for key, value in composition_mode_weights.items()
        }
        self.responsibilities = {
            str(key): torch.as_tensor(value, dtype=torch.float32).detach().cpu()
            for key, value in (responsibilities or {}).items()
        }
        self.metadata = dict(metadata or {})
        self.num_modes = self.mode_means.size(0)
        self.latent_dim = self.mode_means.size(1)

    @classmethod
    def load(cls, path: str | Path, map_location: str = "cpu") -> "CrystalMMFMPrior":
        payload = torch.load(str(path), map_location=map_location, weights_only=False)
        return cls(
            mode_means=payload["mode_means"],
            mode_vars=payload["mode_vars"],
            composition_mode_weights=payload["composition_mode_weights"],
            responsibilities=payload.get("responsibilities"),
            metadata=payload.get("metadata"),
        )

    def save(self, path: str | Path) -> None:
        payload = {
            "mode_means": self.mode_means,
            "mode_vars": self.mode_vars,
            "composition_mode_weights": self.composition_mode_weights,
            "responsibilities": self.responsibilities,
            "metadata": self.metadata,
        }
        torch.save(payload, str(path))

    def mode_probabilities(
        self,
        composition_row: torch.Tensor,
        batch_id: str | None = None,
        mode_coupling: bool = False,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        if mode_coupling and batch_id is not None:
            responsibility = self.responsibilities.get(str(batch_id))
            if responsibility is not None:
                return responsibility.to(device=device, dtype=torch.float32)
        key = composition_key(composition_row)
        weights = self.composition_mode_weights.get(key)
        if weights is None:
            weights = torch.full((self.num_modes,), 1.0 / self.num_modes)
        weights = weights.clamp_min(0.0)
        total = weights.sum()
        if total <= 0:
            weights = torch.full_like(weights, 1.0 / self.num_modes)
        else:
            weights = weights / total
        return weights.to(device=device, dtype=torch.float32)

    def sample_source(
        self,
        composition: torch.Tensor,
        token_mask: torch.Tensor,
        batch_ids: list[str] | None = None,
        mode_coupling: bool = False,
        balanced: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample ``[B, N, D]`` source latents with one shared mode per crystal."""
        if composition.size(0) != token_mask.size(0):
            raise ValueError("composition and token_mask batch sizes disagree.")
        device = token_mask.device
        batch_size = token_mask.size(0)
        profiles = torch.stack(
            [
                self.mode_probabilities(
                    composition[index],
                    batch_id=batch_ids[index] if batch_ids is not None else None,
                    mode_coupling=mode_coupling,
                    device=device,
                )
                for index in range(batch_size)
            ],
            dim=0,
        )
        if balanced and batch_size >= self.num_modes:
            modes = balanced_mode_assignment(profiles)
        else:
            modes = torch.multinomial(
                profiles.clamp_min(1e-8),
                1,
                generator=generator,
            ).squeeze(-1)

        means = self.mode_means.to(device=device, dtype=torch.float32)[modes]
        stds = self.mode_vars.to(device=device, dtype=torch.float32).sqrt()[modes]
        shape = (batch_size, token_mask.size(1), self.latent_dim)
        noise = torch.randn(shape, device=device, generator=generator)
        source = means[:, None, :] + stds[:, None, :] * noise
        return source * token_mask[..., None].to(source.dtype)

    def sample_train_source(
        self,
        composition: torch.Tensor,
        token_mask: torch.Tensor,
        batch_ids: list[str] | None = None,
        mode_coupling: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return self.sample_source(
            composition,
            token_mask,
            batch_ids=batch_ids,
            mode_coupling=mode_coupling,
            balanced=False,
            generator=generator,
        )

    def sample_inference_source(
        self,
        composition: torch.Tensor,
        token_mask: torch.Tensor,
        balanced: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return self.sample_source(
            composition,
            token_mask,
            mode_coupling=False,
            balanced=balanced,
            generator=generator,
        )
