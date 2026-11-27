"""Exported artifacts reproduce the compiled callable in a loading process."""

from __future__ import annotations

import inspect

import pytest
import torch

from fhelium import compile as fc

from test_binding_time import _fixture as _binding_fixture
from test_callable_cache import _fixture


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA"
            ),
        ),
    ],
)
def test_exported_callable_loads_with_caller_keys_and_built_tables(
    tmp_path, device
):
    engine, secret, x, source = _fixture(device)
    compiled = fc.compile(source, cache=False)
    expected = compiled(x)
    path = tmp_path / "rotate.fhelium"
    fc.export(compiled, path, signatures=[(x,)])
    rotation = inspect.getclosurevars(source).nonlocals["rotation"]
    with pytest.raises(ValueError, match="No supplied key"):
        fc.load(path)
    for graph in (False, "auto"):
        loaded = fc.load(path, keys=[rotation], engine=engine, graph=graph)
        actual = loaded(x)
        torch.testing.assert_close(actual.data, expected.data, rtol=0, atol=0)
        assert (actual.depth, actual.scale, actual.prime_ids) == (
            expected.depth,
            expected.scale,
            expected.prime_ids,
        )


@pytest.mark.parametrize("weights", ["raw", "embedded"])
def test_exported_plaintext_preparation_runs_at_load(tmp_path, weights):
    engine, secret, clear, value = _binding_fixture()
    weight = torch.tensor([0.125, 0.25, 0.375, 0.5], dtype=torch.float64)

    def source(current):
        plain = engine.prepare_compressed_plaintext(
            weight,
            depth=current.depth,
            scale=current.scale,
            device=current.device,
        )
        return engine.ntt_domain_to_coefficient_domain(
            engine.multiply_plaintext(
                engine.coefficient_domain_to_ntt_domain(current), plain
            )
        )

    compiled = fc.compile(source, cache=False, graph=False)
    expected = engine.decrypt_message(compiled(value), secret).real
    path = tmp_path / f"{weights}.fhelium"
    fc.export(compiled, path, signatures=[(value,)], weights=weights)
    loaded = fc.load(path, engine=engine, graph=False)
    actual = engine.decrypt_message(loaded(value), secret).real
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-8)


def test_loaded_callable_prepares_argument_weights_per_call(tmp_path):
    engine, secret, clear, value = _binding_fixture()

    def source(current, weight):
        plain = engine.prepare_compressed_plaintext(
            weight,
            depth=current.depth,
            scale=current.scale,
            device=current.device,
        )
        return engine.ntt_domain_to_coefficient_domain(
            engine.multiply_plaintext(
                engine.coefficient_domain_to_ntt_domain(current), plain
            )
        )

    compiled = fc.compile(source, cache=False, graph=False)
    weight = torch.tensor([0.125, 0.25, 0.375, 0.5], dtype=torch.float64)
    path = tmp_path / "weights.fhelium"
    fc.export(compiled, path, signatures=[(value, weight)])
    loaded = fc.load(path, engine=engine, graph=False)
    for current in (weight, weight + 0.125):
        expected = clear * current.repeat(clear.numel() // current.numel())
        actual = engine.decrypt_message(loaded(value, current), secret).real
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-8)
    weight.add_(0.25)
    actual = engine.decrypt_message(loaded(value, weight), secret).real
    torch.testing.assert_close(
        actual,
        clear * weight.repeat(clear.numel() // weight.numel()),
        rtol=0,
        atol=1e-8,
    )
