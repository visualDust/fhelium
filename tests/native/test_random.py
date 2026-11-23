"""Native ChaCha20 keystream and CKKS samplers."""

from __future__ import annotations

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import fhelium.native  # noqa: F401 - loads native torch operator libraries
from fhelium.native.wrapper import random_ops

MODULI = (1152921504606846883, 1152921504606830593, 1152921504606748673)
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _signed(value: int) -> int:
    return value - (1 << 64) if value >= 1 << 63 else value


def _rns_params(device: str) -> torch.Tensor:
    params = torch.zeros((8, len(MODULI)), dtype=torch.int64)
    params[0] = torch.tensor([2 * q for q in MODULI])
    return params.to(device)


def _samplers(table: torch.Tensor):
    return {
        "gaussian": lambda seed, shape, params: (
            random_ops.sample_discrete_gaussian(
                seed, shape, table.to(seed.device), params
            )
        ),
        "binomial": lambda seed, shape, params: (
            random_ops.sample_centered_binomial(seed, shape, 21, params)
        ),
        "ternary": lambda seed, shape, params: random_ops.sample_ternary(
            seed, shape, 0.5, params
        ),
        "fixed_weight": lambda seed, shape, params: (
            random_ops.sample_fixed_weight_ternary(seed, shape, 64, params)
        ),
    }


@pytest.mark.parametrize("device", DEVICES)
def test_chacha20_block_matches_rfc8439(device: str) -> None:
    key = bytes(range(32))
    words = [int.from_bytes(key[8 * i : 8 * i + 8], "little") for i in range(4)]
    seed = torch.tensor(
        [_signed(w) for w in words] + [0x4A000000], device=device
    )
    block = random_ops.chacha20_keystream(
        seed, _signed(1 | 0x09000000 << 32), 1
    )
    assert block.to(torch.int64).cpu().tolist()[0] == [
        0xE4E7F110, 0x15593BD1, 0x1FDD0F50, 0xC47120A3,
        0xC7F4D1C7, 0x0368C033, 0x9AAA2204, 0x4E6CD4C3,
        0x466482D2, 0x09AA9F07, 0x05D7C214, 0xA2028BD9,
        0xD19C12B5, 0xB94E16DE, 0xE883D0CB, 0x4E3C50A2,
    ]  # fmt: skip


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cpu_and_cuda_samples_are_identical() -> None:
    seed = random_ops.generate_seed()
    table = random_ops.gaussian_cdt_table(3.19)
    shape = [3, 1000]
    for sample in _samplers(table).values():
        for params in (None, _rns_params("cpu")):
            cpu = sample(seed, shape, params)
            cuda = sample(
                seed.cuda(), shape, None if params is None else params.cuda()
            )
            assert torch.equal(cpu, cuda.cpu())
    uniform = random_ops.sample_uniform_residues(
        seed, shape, _rns_params("cpu")
    )
    assert torch.equal(
        uniform,
        random_ops.sample_uniform_residues(
            seed.cuda(), shape, _rns_params("cuda")
        ).cpu(),
    )
    values = torch.linspace(-50, 50, 997, dtype=torch.float64)
    assert torch.equal(
        random_ops.stochastic_round(seed, values),
        random_ops.stochastic_round(seed.cuda(), values.cuda()).cpu(),
    )


@pytest.mark.parametrize("device", DEVICES)
def test_sample_distributions(device: str) -> None:
    seed = random_ops.generate_seed().to(device)
    table = random_ops.gaussian_cdt_table(3.19)
    shape = [4, 1 << 14]
    samplers = _samplers(table)
    moduli = torch.tensor(MODULI, device=device).view(-1, 1)
    for name, expected_std in (
        ("gaussian", 3.19),
        ("binomial", (21 / 2) ** 0.5),
        ("ternary", 0.5**0.5),
    ):
        centered = samplers[name](seed, shape, None)
        assert centered.shape == tuple(shape)
        assert abs(centered.double().std().item() / expected_std - 1) < 0.03
        assert abs(centered.double().mean().item()) < 0.1
        residues = samplers[name](seed, shape, _rns_params(device))
        lifted = centered.unsqueeze(-2)
        assert torch.equal(
            residues, torch.where(lifted < 0, lifted + moduli, lifted)
        )
    fixed = samplers["fixed_weight"](seed, shape, None)
    assert torch.all((fixed != 0).sum(-1) == 64)
    assert set(fixed.unique().tolist()) == {-1, 0, 1}
    uniform = random_ops.sample_uniform_residues(
        seed, shape, _rns_params(device)
    )
    assert torch.all((uniform >= 0) & (uniform < moduli))
    values = torch.full((1 << 16,), 0.25, dtype=torch.float64, device=device)
    rounded = random_ops.stochastic_round(seed, values)
    assert abs(rounded.double().mean().item() - 0.25) < 0.01


@pytest.mark.parametrize("device", DEVICES)
def test_fused_error_addition_and_strided_rounding(device: str) -> None:
    seed = random_ops.generate_seed().to(device)
    table = random_ops.gaussian_cdt_table(3.19).to(device)
    params = _rns_params(device)
    moduli = torch.tensor(MODULI, device=device).view(-1, 1)
    lazy = random_ops.sample_uniform_residues(seed, [2, 512], params) * 2
    expected = torch.remainder(
        lazy
        + random_ops.sample_discrete_gaussian(seed, [2, 512], table, params),
        moduli,
    )
    random_ops.add_discrete_gaussian_(lazy, seed, table, params)
    assert torch.equal(lazy, expected)
    full = torch.zeros(2, 512, dtype=torch.float64, device=device)
    full[:, ::8] = torch.linspace(-9.5, 9.5, 128, device=device).view(2, 64)
    assert torch.equal(
        random_ops.stochastic_round(seed, full[:, ::8].contiguous(), 8),
        random_ops.stochastic_round(seed, full)[:, ::8],
    )


def test_sampler_fake_shapes() -> None:
    with FakeTensorMode() as mode:
        seed = mode.from_tensor(torch.zeros(5, dtype=torch.int64))
        params = mode.from_tensor(_rns_params("cpu"))
        residues = random_ops.sample_ternary(seed, [2, 16], 0.5, params)
        centered = random_ops.sample_ternary(seed, [2, 16], 0.5, None)
        keystream = random_ops.chacha20_keystream(seed, 0, 7)
    assert residues.shape == (2, len(MODULI), 16)
    assert centered.shape == (2, 16)
    assert keystream.shape == (7, 16) and keystream.dtype == torch.uint32
