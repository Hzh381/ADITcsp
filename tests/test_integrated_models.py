"""CPU smoke tests for the integrated HiLo VAE and CoDeMDT denoiser."""

import tempfile
from functools import partial
from pathlib import Path

import lightning
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Batch, Data

from src.models.decoders.transformer import TransformerDecoder
from src.models.denoisers.code_mdt import CoDeMDT
from src.models.encoders.transformer import TransformerEncoder
from src.models.interpolants.flow_matching import FlowMatchingInterpolant
from src.models.ldm_module import LatentDiffusionLitModule
from src.models.vae_module import VariationalAutoencoderLitModule


def _batch(lengths=(7, 5)):
    data_list = []
    for index, length in enumerate(lengths):
        frac = torch.rand(length, 3)
        cell_matrix = torch.eye(3)
        cell = cell_matrix.unsqueeze(0).repeat(length, 1, 1).unsqueeze(1)
        data_list.append(
            Data(
                atom_types=torch.full((length,), index + 1, dtype=torch.long),
                pos=frac @ cell_matrix,
                frac_coords=frac,
                cell=cell,
                token_idx=torch.arange(length, dtype=torch.long),
                num_atoms=torch.tensor([length], dtype=torch.long),
                dataset_idx=torch.tensor([index], dtype=torch.long),
                spacegroup=torch.tensor([1], dtype=torch.long),
            )
        )
    return Batch.from_data_list(data_list)


def test_hilo_encoder_and_decoder_preserve_flat_token_contract():
    common = dict(
        max_num_elements=16,
        d_model=32,
        nhead=4,
        dim_feedforward=64,
        dropout=0.0,
        num_layers=2,
        attention_type="hi_lo",
        low_freq_heads=1,
        hilo_cutoff=2.0,
        hilo_pool_grid_size=2,
    )
    encoder = TransformerEncoder(**common).eval()
    decoder = TransformerDecoder(**{**common, "attention_type": "standard"}).eval()
    batch = _batch()

    with torch.no_grad():
        encoded = encoder(batch)
        decoded = decoder(encoded)

    assert encoded["x"].shape == (batch.num_nodes, common["d_model"])
    assert torch.isfinite(encoded["x"]).all()
    assert decoded["atom_types"].shape == (batch.num_nodes, common["max_num_elements"])
    assert decoded["lengths"].shape == (batch.num_graphs, 3)
    assert torch.isfinite(decoded["frac_coords"]).all()


def test_codemdt_masked_branch_adds_diversity_loss():
    lengths = (6, 4)
    batch_size = len(lengths)
    num_tokens = max(lengths)
    mask = torch.arange(num_tokens).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)
    x = torch.randn(batch_size, num_tokens, 8) * mask[..., None]
    t = torch.full((batch_size, 1), 0.5)
    composition = torch.zeros(batch_size, 100)
    model = CoDeMDT(
        d_x=8,
        d_model=32,
        nhead=4,
        num_layers=6,
        num_datasets=2,
        num_spacegroups=8,
        composition_dim=100,
        dropout=0.0,
        mask_ratio=0.3,
        mask_ratio_range=0.2,
        decode_layer=2,
    )

    plain = model(
        x,
        t,
        dataset_idx=torch.ones(batch_size, dtype=torch.long),
        spacegroup=torch.zeros(batch_size, dtype=torch.long),
        mask=mask,
        composition=composition,
    )
    masked = model(
        x,
        t,
        dataset_idx=torch.ones(batch_size, dtype=torch.long),
        spacegroup=torch.zeros(batch_size, dtype=torch.long),
        mask=mask,
        composition=composition,
        enable_mask=True,
    )
    diversity = model.diversity_loss()

    assert plain.shape == x.shape and masked.shape == x.shape
    assert torch.count_nonzero(masked[~mask]) == 0
    assert diversity is not None and torch.isfinite(diversity["div_loss"])
    assert model.diversity_term(0, plain.detach().square().mean()) is not None


def test_latent_module_combines_plain_masked_and_diversity_losses():
    runtime_dir = Path(__file__).resolve().parent
    temporary = tempfile.NamedTemporaryFile(
        dir=runtime_dir, suffix=".ckpt", delete=False
    )
    temporary.close()
    checkpoint = Path(temporary.name)
    try:
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
                {"visualize": False, "save_dir": str(runtime_dir)}
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
            checkpoint,
        )

        denoiser = CoDeMDT(
            d_x=8,
            d_model=32,
            nhead=4,
            num_layers=6,
            num_datasets=2,
            num_spacegroups=8,
            composition_dim=16,
            dropout=0.0,
            mask_ratio=0.3,
            mask_ratio_range=0.2,
            decode_layer=2,
        )
        module = LatentDiffusionLitModule(
            autoencoder_ckpt=str(checkpoint),
            denoiser=denoiser,
            interpolant=FlowMatchingInterpolant(
                num_timesteps=2,
                self_condition=False,
            ),
            optimizer=partial(torch.optim.AdamW, lr=1e-4),
            scheduler=None,
            scheduler_frequency=10,
            mask_loss_weight=1.0,
        )
        noisy, conditioning = module._prepare_batch(_batch())
        plain = module.criterion(noisy, module._denoise(noisy, conditioning))
        masked = module.criterion(
            noisy,
            module._denoise(noisy, conditioning, enable_mask=True),
        )
        diversity = denoiser.diversity_term(0, plain["loss"].detach())
        total = plain["loss"] + masked["loss"] + diversity
        total.backward()

        assert module.masking_enabled and module.has_diversity
        assert torch.isfinite(total)
        assert any(
            parameter.grad is not None
            for block in denoiser.encoder_blocks
            for parameter in block.parameters()
        )

        module.cpu().eval()
        composition = torch.zeros(1, 16)
        composition[0, 1] = 3
        with torch.no_grad():
            decoded, metadata, samples = module.sample_and_decode(
                composition, cfg_scale=1.5
            )
        assert metadata["num_atoms"].tolist() == [3]
        assert len(samples["clean_traj"]) == 2
        assert torch.isfinite(decoded["frac_coords"]).all()
        assert torch.isfinite(decoded["lengths"]).all()

        collector = denoiser.register_cka_hooks()
        with torch.no_grad():
            module._denoise(*module._prepare_batch(_batch()))
        features = collector.latest_features()
        collector.remove()
        assert len(features) == 1 + len(denoiser.cka_blocks) + 1
    finally:
        checkpoint.unlink(missing_ok=True)
