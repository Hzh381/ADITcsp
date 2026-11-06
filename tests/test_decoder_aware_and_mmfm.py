"""CPU smoke tests for decoder-aware FM and Crystal-MM-FM integration."""

import tempfile
from functools import partial
from pathlib import Path

import lightning
import rootutils
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Batch, Data

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.crystal_mmfm_prior import (
    CrystalMMFMPrior,
    balanced_mode_assignment,
    build_global_descriptor,
    composition_key,
    estimate_token_priors,
    fit_global_gmm,
)
from src.models.decoders.transformer import TransformerDecoder
from src.models.decoder_struct_loss import compute_periodic_loss, compute_struct_loss
from src.models.periodic_coord_loss import periodic_coord_loss
from src.models.denoisers.dit import DiT
from src.models.encoders.transformer import TransformerEncoder
from src.models.interpolants.flow_matching import FlowMatchingInterpolant
from src.models.ldm_module import LatentDiffusionLitModule
from src.models.vae_module import VariationalAutoencoderLitModule


def _batch():
    data_list = []
    for index, length in enumerate((5, 3)):
        frac = torch.rand(length, 3)
        lengths = torch.tensor([2.0 + index, 2.2 + index, 2.4 + index])
        angles = torch.tensor([90.0, 90.0, 90.0 + index])
        cell = torch.eye(3).unsqueeze(0).repeat(length, 1, 1)
        data_list.append(
            Data(
                id=f"m{index}",
                atom_types=torch.ones(length, dtype=torch.long),
                pos=frac @ cell[0],
                frac_coords=frac,
                cell=cell,
                lengths=lengths.view(1, -1),
                lengths_scaled=(lengths / length ** (1 / 3)).view(1, -1),
                angles=angles.view(1, -1),
                angles_radians=torch.deg2rad(angles).view(1, -1),
                token_idx=torch.arange(length, dtype=torch.long),
                num_atoms=torch.tensor([length], dtype=torch.long),
                dataset_idx=torch.tensor([0], dtype=torch.long),
                spacegroup=torch.tensor([1], dtype=torch.long),
                composition=torch.eye(16)[1].view(1, -1),
            )
        )
    return Batch.from_data_list(data_list)


def _checkpoint(path: Path):
    common = dict(
        max_num_elements=16,
        d_model=32,
        nhead=4,
        dim_feedforward=64,
        dropout=0.0,
        num_layers=2,
    )
    vae = VariationalAutoencoderLitModule(
        encoder=TransformerEncoder(
            **common,
            attention_type="hi_lo",
            low_freq_heads=1,
            hilo_cutoff=2.0,
            hilo_pool_grid_size=2,
        ),
        decoder=TransformerDecoder(**common, attention_type="standard"),
        latent_dim=8,
        optimizer=partial(torch.optim.AdamW, lr=1e-4),
        scheduler=None,
        scheduler_frequency=10,
        loss_weights={
            "loss_atom_types": {"mp20": 1.0},
            "loss_lengths": {"mp20": 1.0},
            "loss_angles": {"mp20": 10.0},
            "loss_frac_coords": {"mp20": 10.0},
            "loss_pos": {"mp20": 0.0},
            "loss_kl": {"mp20": 1e-5},
        },
        augmentations=OmegaConf.create(
            {"noise": 0.0, "frac_coords": False, "pos": False}
        ),
        visualization=OmegaConf.create(
            {"visualize": False, "save_dir": str(path.parent)}
        ),
        compile=False,
        composition_conditioning=True,
        num_elements=16,
        composition_dropout=0.0,
        composition_loss_weight=1.0,
    )
    torch.save(
        {
            "state_dict": vae.state_dict(),
            "hyper_parameters": dict(vae.hparams),
            "pytorch-lightning_version": lightning.__version__,
        },
        path,
    )


def _denoiser():
    return DiT(
        d_x=8,
        d_model=32,
        nhead=4,
        num_layers=2,
        num_datasets=2,
        composition_dim=16,
        composition_dropout_prob=0.1,
        dropout=0.0,
    )


def _prior(path: Path):
    means = torch.tensor(
        [
            [0.5] * 8,
            [-0.5] * 8,
        ]
    )
    vars = torch.full_like(means, 0.01)
    key = composition_key(torch.eye(16)[1])
    responsibilities = {
        "m0": torch.tensor([0.9, 0.1]),
        "m1": torch.tensor([0.1, 0.9]),
    }
    prior = CrystalMMFMPrior(
        mode_means=means,
        mode_vars=vars,
        composition_mode_weights={key: torch.tensor([0.5, 0.5])},
        responsibilities=responsibilities,
    )
    prior.save(path)
    return key


def test_decoder_aware_loss_backpropagates():
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "vae.ckpt"
        _checkpoint(checkpoint)
        module = LatentDiffusionLitModule(
            autoencoder_ckpt=str(checkpoint),
            denoiser=_denoiser(),
            interpolant=FlowMatchingInterpolant(
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
        noisy, conditioning = module._prepare_batch(_batch())
        pred = module._denoise(noisy, conditioning)
        plain = module.criterion(noisy, pred)
        structure = module._decoder_structure_loss(pred, conditioning, noisy)
        total = (
            plain["loss"]
            + 0.1 * structure["cell_loss"]
            + 0.1 * structure["geo_loss"]
            + 0.1 * structure["coord_loss"]
        )
        total.backward()
        assert torch.isfinite(total)
        assert (
            structure["cell_loss"] >= 0
            and structure["geo_loss"] >= 0
            and structure["coord_loss"] >= 0
        )
        assert any(p.grad is not None for p in module.denoiser.parameters())
        assert all(
            torch.isfinite(p.grad).all()
            for p in module.denoiser.parameters()
            if p.grad is not None
        )
        assert not any(p.requires_grad for p in module.autoencoder.parameters())


def test_struct_loss_is_zero_for_identical_structures():
    frac = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.2, 0.3, 0.4], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.3, 0.4, 0.5], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    )
    mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
    lengths = torch.tensor([[2.0, 2.2, 2.4], [2.1, 2.3, 2.5]])
    angles = torch.deg2rad(torch.tensor([[90.0, 90.0, 90.0], [80.0, 90.0, 100.0]]))
    result = compute_struct_loss(
        frac, lengths, angles, frac, lengths, angles, mask
    )
    assert result["cell_loss"].abs().item() < 1e-5
    assert result["geo_loss"].abs().item() < 1e-5


def test_periodic_loss_is_finite_for_coincident_atoms():
    pred_frac = torch.zeros(1, 2, 3, requires_grad=True)
    target_frac = torch.zeros(1, 2, 3)
    mask = torch.ones(1, 2, dtype=torch.bool)
    lengths = torch.tensor([[2.0, 2.1, 2.2]])
    angles = torch.deg2rad(torch.tensor([[90.0, 90.0, 90.0]]))
    loss = compute_periodic_loss(
        pred_frac,
        lengths,
        angles,
        target_frac,
        lengths,
        angles,
        mask,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(pred_frac.grad).all()


def test_periodic_coord_loss_handles_permutation_and_translation():
    target_frac = torch.tensor(
        [
            [[0.0, 0.0, 0.0],
             [0.5, 0.5, 0.5],
             [0.2, 0.3, 0.4],
             [0.7, 0.1, 0.6]]
        ],
        dtype=torch.float32,
    )
    target_types = torch.tensor([[1, 2, 1, 2]])
    permutation = torch.tensor([2, 0, 3, 1])
    shift = torch.tensor([0.2, -0.1, 0.3])
    pred_frac = (
        target_frac[:, permutation] + shift.view(1, 1, 3)
    ).requires_grad_(True)
    pred_types = target_types[:, permutation]
    lattice = torch.diag(torch.tensor([2.0, 2.2, 2.4])).unsqueeze(0)
    mask = torch.ones(1, 4, dtype=torch.bool)
    loss = periodic_coord_loss(
        pred_frac,
        target_frac,
        lattice,
        pred_types=pred_types,
        target_types=target_types,
        atom_mask=mask,
        max_shift_candidates=0,
    )
    loss.backward()
    assert loss.item() < 1e-8
    assert torch.isfinite(pred_frac.grad).all()


def test_periodic_coord_loss_skips_incompatible_composition():
    pred_frac = torch.zeros(1, 2, 3, requires_grad=True)
    target_frac = torch.zeros(1, 2, 3)
    lattice = torch.eye(3).unsqueeze(0)
    mask = torch.ones(1, 2, dtype=torch.bool)
    loss = periodic_coord_loss(
        pred_frac,
        target_frac,
        lattice,
        pred_types=torch.tensor([[1, 1]]),
        target_types=torch.tensor([[1, 2]]),
        atom_mask=mask,
    )
    assert loss.item() == 0.0


def test_global_descriptor_shapes():
    batch_size, num_tokens, latent_dim = 2, 4, 8
    mask = torch.tensor(
        [[True, True, True, True], [True, True, False, False]]
    )
    means = torch.randn(batch_size, num_tokens, latent_dim)
    stds = torch.rand_like(means)
    frac = torch.rand(batch_size, num_tokens, 3)
    cell = torch.eye(3).repeat(batch_size, 1, 1)
    lengths = torch.tensor([[2.0, 2.1, 2.2], [2.3, 2.4, 2.5]])
    angles = torch.full((batch_size, 3), torch.pi / 2)
    descriptor = build_global_descriptor(
        means,
        stds,
        lengths,
        angles,
        frac,
        cell,
        mask,
        num_rdf_bins=4,
        r_max=3.0,
    )
    assert descriptor.shape == (batch_size, 2 * latent_dim + 6 + 4)
    assert torch.isfinite(descriptor).all()


def test_gmm_and_token_prior_helpers():
    descriptors = torch.randn(16, 5)
    normalized, responsibilities, model, mean, std = fit_global_gmm(
        descriptors, num_modes=2, seed=0
    )
    assert normalized.shape == descriptors.shape
    assert responsibilities.shape == (16, 2)
    assert torch.allclose(responsibilities.sum(dim=1), torch.ones(16), atol=1e-5)
    valid_means = torch.randn(20, 8)
    valid_vars = torch.rand(20, 8)
    token_counts = torch.tensor([10, 10])
    mode_means, mode_vars, mass = estimate_token_priors(
        valid_means,
        valid_vars,
        token_counts,
        responsibilities[:2],
    )
    assert mode_means.shape == mode_vars.shape == (2, 8)
    assert mass.shape == (2,)
    assert torch.isfinite(mode_means).all() and torch.isfinite(mode_vars).all()


def test_mmfm_prior_source_and_sampling():
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "vae.ckpt"
        prior_path = Path(directory) / "prior.pt"
        _checkpoint(checkpoint)
        _prior(prior_path)
        module = LatentDiffusionLitModule(
            autoencoder_ckpt=str(checkpoint),
            denoiser=_denoiser(),
            interpolant=FlowMatchingInterpolant(
                num_timesteps=2,
                self_condition=False,
            ),
            optimizer=partial(torch.optim.AdamW, lr=1e-4),
            scheduler=None,
            scheduler_frequency=10,
            mmfm_prior_path=str(prior_path),
            mmfm_mode_coupling=True,
            mmfm_balanced=True,
        )
        noisy, conditioning = module._prepare_batch(_batch())
        assert noisy["x_0"].shape == noisy["x_1"].shape
        assert torch.count_nonzero(noisy["x_0"][~conditioning["mask"]]) == 0
        module.eval()
        composition = torch.eye(16)[1].view(1, -1).repeat(4, 1)
        decoded, metadata, samples = module.sample_and_decode(
            composition, cfg_scale=1.0
        )
        assert metadata["num_atoms"].shape == (4,)
        assert torch.isfinite(decoded["frac_coords"]).all()
        assert torch.isfinite(samples["clean_traj"][-1]).all()
        uniform = torch.full((4, 2), 0.5)
        balanced = balanced_mode_assignment(uniform)
        assert sorted(balanced.tolist()) == [0, 0, 1, 1]


def main():
    test_decoder_aware_loss_backpropagates()
    test_struct_loss_is_zero_for_identical_structures()
    test_periodic_loss_is_finite_for_coincident_atoms()
    test_periodic_coord_loss_handles_permutation_and_translation()
    test_periodic_coord_loss_skips_incompatible_composition()
    test_global_descriptor_shapes()
    test_gmm_and_token_prior_helpers()
    test_mmfm_prior_source_and_sampling()
    print("PASS decoder-aware and Crystal-MM-FM smoke tests")


if __name__ == "__main__":
    main()
