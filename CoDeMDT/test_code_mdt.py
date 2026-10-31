"""CPU self-tests for CoDeMDT.

Runs standalone (no ADITcsp, no pytest):
    python -B test_code_mdt.py
Pytest also picks up the ``test_*`` functions.
"""

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from code_mdt import CoDeMDT, DiversityWeighter  # noqa: E402

NUM_TOKENS = 7
D_MODEL = 32


def assert_raises(exception_type, callable_):
    try:
        callable_()
    except exception_type:
        return
    raise AssertionError(f"{exception_type.__name__} was not raised")


def build_model(**overrides):
    kwargs = dict(
        d_x=8,
        d_model=D_MODEL,
        num_layers=6,
        nhead=4,
        num_datasets=2,
        num_spacegroups=8,
        composition_dim=100,
        composition_dropout_prob=0.1,
        dropout=0.0,
        mask_ratio=0.3,
        mask_ratio_range=0.2,
        decode_layer=2,
        div_weight=0.2,
    )
    kwargs.update(overrides)
    torch.manual_seed(0)
    return CoDeMDT(**kwargs)


def build_batch(lengths=(7, 5, 3), seed=0):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(len(lengths), NUM_TOKENS, 8, generator=generator)
    mask = torch.arange(NUM_TOKENS).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)
    return x * mask[..., None], mask, torch.full((len(lengths), 1), 0.5)


def default_kwargs(mask, t, batch_size=None):
    batch_size = mask.size(0) if batch_size is None else batch_size
    return dict(
        t=t,
        dataset_idx=torch.ones(batch_size, dtype=torch.long),
        spacegroup=torch.zeros(batch_size, dtype=torch.long),
        mask=mask,
        composition=torch.zeros(batch_size, 100),
    )


# ---------------------------------------------------------------- backbone
def test_backbone_split_matches_mdtv2():
    model = build_model()
    assert len(model.en_in_blocks) == 2
    assert len(model.en_out_blocks) == 2
    assert len(model.side_blocks) == 1
    assert len(model.de_blocks) == 2
    assert model.num_blocks == 7
    assert len(model.encoder_blocks) == 4
    assert len(model.cka_blocks) == 6
    assert model.supports_masking and model.has_diversity


def test_invalid_layer_split_is_rejected():
    assert_raises(ValueError, lambda: build_model(num_layers=6, decode_layer=3))
    assert_raises(ValueError, lambda: build_model(num_layers=6, decode_layer=6))
    assert_raises(ValueError, lambda: build_model(rel_pos_bias="2d"))


def test_rel_pos_bias_is_optional():
    model = build_model(rel_pos_bias="none")
    assert model.en_in_blocks[0].attn.rel_bias is None
    x, mask, t = build_batch()
    with torch.no_grad():
        out = model(x, enable_mask=False, **default_kwargs(mask, t))
    assert torch.isfinite(out).all()


def test_unmasked_pass_keeps_shape_and_zeroes_padding():
    model = build_model().eval()
    x, mask, t = build_batch()
    with torch.no_grad():
        out = model(x, enable_mask=False, **default_kwargs(mask, t))
    assert out.shape == x.shape
    assert torch.count_nonzero(out[~mask]) == 0


def test_masked_pass_hides_tokens_and_restores_full_length():
    model = build_model().eval()
    x, mask, t = build_batch()
    with torch.no_grad():
        embedded = model.x_embedder(torch.cat([x, torch.zeros_like(x)], dim=-1))
        x_keep, ids_keep, keep_mask = model._random_masking(embedded, mask)
        out = model(x, enable_mask=True, **default_kwargs(mask, t))

    lengths = mask.sum(dim=-1)
    keep_counts = keep_mask.sum(dim=-1)
    assert torch.all(keep_counts >= 1)
    assert torch.all(keep_counts < lengths)  # 30-50% really removed
    assert x_keep.shape[1] == int(keep_counts.max())
    assert torch.all(mask.gather(1, ids_keep)[keep_mask])
    assert out.shape == x.shape
    assert torch.count_nonzero(out[~mask]) == 0


def test_padding_never_leaks_into_valid_tokens():
    model = build_model().eval()
    x, mask, t = build_batch()
    padded = (~mask)[..., None]
    for enable_mask in (False, True):
        with torch.no_grad():
            torch.manual_seed(0)
            clean = model(x, enable_mask=enable_mask, **default_kwargs(mask, t))
            torch.manual_seed(0)
            dirty = model(
                x + 1000.0 * padded, enable_mask=enable_mask, **default_kwargs(mask, t)
            )
        assert torch.allclose(clean, dirty, atol=1e-6)


def test_single_atom_sample_survives_masking():
    model = build_model().eval()
    x, mask, t = build_batch(lengths=(7, 1, 3))
    with torch.no_grad():
        out = model(x, enable_mask=True, **default_kwargs(mask, t))
    assert torch.isfinite(out).all()


def test_classifier_free_guidance_halves_and_reshapes():
    model = build_model().eval()
    x, mask, t = build_batch()
    doubled = 2 * mask.size(0)
    kwargs = default_kwargs(mask, t, batch_size=doubled)
    kwargs["mask"] = torch.cat([mask, mask], dim=0)
    kwargs["t"] = torch.cat([t, t], dim=0)
    with torch.no_grad():
        out = model.forward_with_cfg(torch.cat([x, x], dim=0), cfg_scale=2.0, **kwargs)
    assert out.shape == (doubled, NUM_TOKENS, 8)
    assert torch.allclose(out[: mask.size(0)], out[mask.size(0):])


def test_inference_is_deterministic_in_eval_mode():
    model = build_model().eval()
    x, mask, t = build_batch()
    kwargs = default_kwargs(mask, t)
    with torch.no_grad():
        first = model(x, enable_mask=False, **kwargs)
        second = model(x, enable_mask=False, **kwargs)
    assert torch.equal(first, second)


# --------------------------------------------------------------- diversity
def test_diversity_loss_only_on_masked_pass():
    model = build_model().eval()
    x, mask, t = build_batch()
    kwargs = default_kwargs(mask, t)
    with torch.no_grad():
        model(x, enable_mask=False, **kwargs)
        assert model.diversity_loss() is None
        model(x, enable_mask=True, **kwargs)
        losses = model.diversity_loss()
    assert losses is not None
    assert losses["num_pairs"] == 6  # 4 encoder blocks -> C(4,2)
    for key in ("div_loss", "loss_orth", "loss_mi", "loss_disp"):
        assert torch.isfinite(losses[key]), key
    assert model.diversity_components()


def test_diversity_loss_touches_encoder_only():
    model = build_model().train()
    x, mask, t = build_batch()
    model(x, enable_mask=True, **default_kwargs(mask, t))
    div = model.diversity_loss()["div_loss"]
    assert div.requires_grad
    div.backward()

    assert model.x_embedder.weight.grad is not None
    assert any(p.grad is not None for p in model.en_in_blocks.parameters())
    # decoder representations are not part of the diversity graph
    assert all(p.grad is None for p in model.de_blocks.parameters())
    assert all(p.grad is None for p in model.side_blocks.parameters())


def test_diversity_loss_ignores_padding():
    model = build_model().eval()
    x, mask, t = build_batch()
    padded = (~mask)[..., None]
    with torch.no_grad():
        torch.manual_seed(0)
        model(x, enable_mask=True, **default_kwargs(mask, t))
        clean = float(model.diversity_loss()["div_loss"])
        torch.manual_seed(0)
        model(x + 1000.0 * padded, enable_mask=True, **default_kwargs(mask, t))
        dirty = float(model.diversity_loss()["div_loss"])
    assert abs(clean - dirty) < 1e-6


def test_return_aux_and_diversity_term():
    model = build_model().train()
    x, mask, t = build_batch()
    kwargs = default_kwargs(mask, t)

    pred, aux = model(x, enable_mask=True, return_aux=True, **kwargs)
    assert pred.shape == x.shape and aux is not None

    term = model.diversity_term(10, torch.tensor(5.0))
    assert term is not None and term.requires_grad
    weight = model.diversity_weight()
    assert 0.1 <= weight <= 0.5

    model(x, enable_mask=False, **kwargs)
    assert model.diversity_term(10, torch.tensor(5.0)) is None


def test_diversity_weighter_bounds_and_warmup():
    weighter = DiversityWeighter(adaptive=False, warmup_steps=0)
    assert 0.1 <= weighter.weight(0, torch.tensor(1.0), torch.tensor(1.0)) <= 0.5

    weighter = DiversityWeighter(adaptive=True, warmup_steps=0)
    # huge diffusion loss relative to diversity -> saturates at the upper bound
    assert weighter.weight(0, torch.tensor(0.01), torch.tensor(1000.0)) == 0.5
    assert weighter.weight(0, torch.tensor(1000.0), torch.tensor(0.01)) == 0.1

    warmed = DiversityWeighter(adaptive=False, warmup_steps=100)
    early = warmed.weight(0, torch.tensor(1.0), torch.tensor(1.0))
    late = warmed.weight(99, torch.tensor(1.0), torch.tensor(1.0))
    assert early < late
    assert 0.0 <= early < 0.1


def test_gradients_reach_new_mechanisms():
    """adaLN-Zero needs a few steps before block-internal gradients appear."""
    model = build_model().train()
    x, mask, t = build_batch()
    kwargs = default_kwargs(mask, t)
    target = torch.randn_like(x)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    for step in range(3):
        optimizer.zero_grad()
        plain = model(x, enable_mask=False, **kwargs)
        masked = model(x, enable_mask=True, **kwargs)
        loss = (plain - target).square().mean() + (masked - target).square().mean()
        term = model.diversity_term(step, loss.detach())
        if term is not None:
            loss = loss + term
        loss.backward()

        assert model.mask_token.grad is not None
        assert any(
            block.attn.rel_bias.table.grad is not None
            for block in model.encoder_blocks
        )
        if step == 2:
            assert torch.count_nonzero(model.mask_token.grad) > 0
            assert torch.count_nonzero(
                model.en_in_blocks[0].attn.rel_bias.table.grad
            ) > 0
            for block in model.de_blocks:
                assert torch.count_nonzero(block.skip_linear.weight.grad) > 0
        optimizer.step()


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
