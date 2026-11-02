"""CPU tests for the DiverseDiT denoiser."""

import rootutils
import torch

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.denoisers.diverse_dit import DiverseDiT


def _batch(lengths=(6, 4)):
    batch_size = len(lengths)
    num_tokens = max(lengths)
    mask = torch.arange(num_tokens).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(batch_size, num_tokens, 8, generator=generator) * mask[..., None]
    composition = torch.zeros(batch_size, 16)
    for index, length in enumerate(lengths):
        composition[index, 1] = float(length)
    return x, mask, composition


def _model(**overrides):
    kwargs = dict(
        d_x=8,
        d_model=32,
        nhead=4,
        num_layers=4,
        num_datasets=2,
        num_spacegroups=8,
        composition_dim=16,
        composition_dropout_prob=0.1,
        dropout=0.0,
        long_residual=True,
        diversity_loss=True,
        diversity_blocks=4,
        diversity_max_pairs=10,
    )
    kwargs.update(overrides)
    torch.manual_seed(0)
    return DiverseDiT(**kwargs)


def _forward(model, x, mask, composition):
    return model(
        x=x,
        t=torch.full((x.size(0), 1), 0.5),
        dataset_idx=torch.ones(x.size(0), dtype=torch.long),
        spacegroup=torch.zeros(x.size(0), dtype=torch.long),
        mask=mask,
        composition=composition,
    )


def test_diverse_dit_forward_and_diversity_backward():
    model = _model().train()
    x, mask, composition = _batch()
    output = _forward(model, x, mask, composition)
    divergence = model.diversity_loss()
    term = model.diversity_term(0, output.detach().square().mean())

    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output[~mask]) == 0
    assert divergence is not None
    assert term is not None and term.requires_grad
    assert 0.0 <= model.diversity_weight() <= 1.0

    (output.square().mean() + term).backward()
    assert any(
        parameter.grad is not None
        for parameter in model.long_residual_fusions.parameters()
    )
    assert any(parameter.grad is not None for parameter in model.blocks.parameters())


def test_diverse_dit_eval_skips_diversity_and_cfg_shapes():
    model = _model().eval()
    x, mask, composition = _batch()
    doubled = 2 * x.size(0)
    kwargs = dict(
        t=torch.full((doubled, 1), 0.5),
        dataset_idx=torch.ones(doubled, dtype=torch.long),
        spacegroup=torch.zeros(doubled, dtype=torch.long),
        mask=torch.cat([mask, mask], dim=0),
        cfg_scale=2.0,
        composition=torch.cat([composition, composition], dim=0),
    )
    output = model.forward_with_cfg(torch.cat([x, x], dim=0), **kwargs)

    assert output.shape == (doubled, x.size(1), x.size(2))
    assert torch.isfinite(output).all()
    assert model.diversity_loss() is None


def test_padding_does_not_leak_into_valid_tokens():
    model = _model().eval()
    x, mask, composition = _batch()
    padded = (~mask)[..., None]
    with torch.no_grad():
        clean = _forward(model, x, mask, composition)
        dirty = _forward(model, x + 1000.0 * padded, mask, composition)
    assert torch.allclose(clean, dirty, atol=1e-6)


def test_even_layer_requirement_for_long_residual():
    try:
        _model(num_layers=3)
    except ValueError:
        return
    raise AssertionError("DiverseDiT accepted an odd layer count")


if __name__ == "__main__":
    failures = 0
    for name, function in sorted(globals().items()):
        if not name.startswith("test_") or not callable(function):
            continue
        try:
            function()
        except Exception as error:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(error).__name__}: {error}")
        else:
            print(f"PASS {name}")
    raise SystemExit(1 if failures else 0)
