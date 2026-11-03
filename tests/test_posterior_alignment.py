"""CPU tests for minibatch OT coupling and terminal posterior MMD."""

import rootutils
import torch

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.components.ot_coupling import MinibatchOTCoupler
from src.models.components.posterior_mmd import PosteriorMMD
from src.models.interpolants.flow_matching import FlowMatchingInterpolant


class ToyFlow(torch.nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.linear = torch.nn.Linear(dim, dim)

    def forward(
        self,
        x,
        t,
        dataset_idx,
        spacegroup,
        mask,
        x_sc=None,
        composition=None,
    ):
        del t, dataset_idx, spacegroup, x_sc, composition
        return torch.tanh(self.linear(x)) * mask[..., None]


def test_hungarian_coupling_reorders_posterior_batch():
    coupler = MinibatchOTCoupler()
    z0 = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
    z1 = torch.tensor([[10.0, 10.0], [0.0, 0.0]])
    matched = coupler(z0, z1, group_ids=torch.zeros(2, dtype=torch.long))

    assert torch.allclose(matched, z0)
    assert coupler.last_cost == 0.0


def test_hungarian_coupling_respects_groups():
    coupler = MinibatchOTCoupler()
    z0 = torch.tensor(
        [[0.0, 0.0], [1.0, 1.0], [10.0, 10.0], [11.0, 11.0]]
    )
    z1 = torch.tensor(
        [[1.0, 1.0], [0.0, 0.0], [11.0, 11.0], [10.0, 10.0]]
    )
    matched = coupler(
        z0,
        z1,
        group_ids=torch.tensor([0, 0, 1, 1], dtype=torch.long),
    )
    assert torch.allclose(matched, z0)


def test_posterior_mmd_zero_for_identical_samples():
    mmd = PosteriorMMD()
    values = torch.randn(8, 6)
    loss = mmd(values, values.clone())
    assert loss.item() >= 0.0
    assert loss.item() < 1e-6


def test_posterior_mmd_detects_shift_and_provides_gradient():
    mmd = PosteriorMMD()
    posterior = torch.randn(8, 6)
    generated = (posterior + 2.0).requires_grad_()
    loss = mmd(generated, posterior)
    assert loss.item() > 1e-4
    assert loss.requires_grad
    loss.backward()
    assert generated.grad is not None
    assert torch.isfinite(generated.grad).all()


def test_posterior_mmd_handles_padded_tokens_and_mixed_lengths():
    mmd = PosteriorMMD()
    posterior = torch.randn(4, 5, 8)
    generated = (posterior + 0.1).requires_grad_()
    mask = torch.tensor(
        [
            [True, True, True, True, True],
            [True, True, True, False, False],
            [True, True, True, True, False],
            [True, True, False, False, False],
        ]
    )
    loss = mmd(generated, posterior, mask)
    assert torch.isfinite(loss)
    assert loss.requires_grad


def test_alignment_ode_retains_autograd_graph():
    interpolant = FlowMatchingInterpolant(
        min_t=0.1,
        corrupt=True,
        num_timesteps=3,
        self_condition=True,
        device="cpu",
    )
    model = ToyFlow()
    mask = torch.ones(2, 4, dtype=torch.bool)
    generated = interpolant.sample_for_alignment(
        batch_size=2,
        num_tokens=4,
        emb_dim=8,
        model=model,
        dataset_idx=torch.ones(2, dtype=torch.long),
        spacegroup=torch.zeros(2, dtype=torch.long),
        num_timesteps=3,
        token_mask=mask,
        composition=torch.zeros(2, 16),
    )

    assert generated.shape == (2, 4, 8)
    assert generated.requires_grad
    generated.square().mean().backward()
    assert model.linear.weight.grad is not None
    assert torch.isfinite(model.linear.weight.grad).all()


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
