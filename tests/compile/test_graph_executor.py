"""Graph-segment execution of compiled callables keeps prepared results and fallbacks."""

from collections.abc import Sequence
from typing import cast

import pytest
import torch

import fhelium.compile as fc


def _source(x, y):
    z = x * 2 + y
    return {"value": z, "nested": ([z - 1], None, "static")}


@pytest.mark.parametrize("graph", ["auto", True, False])
def test_cpu_inputs_execute_prepared(graph):
    compiled = fc.compile(_source, cache=False, graph=graph)
    for value in (1.0, 4.0):
        x, y = torch.tensor([value]), torch.tensor([2.0])
        result = compiled(x, y)
        torch.testing.assert_close(result["value"], x * 2 + y)
        assert result["nested"][1:] == (None, "static")
    (stats,) = cast(Sequence[dict], compiled.stats()["graphs"])
    assert stats["state"] == ("disabled" if graph is False else "fallback")
    assert stats["prepared_calls"] == 2
    if graph is not False:
        assert "non-CUDA" in stats["fallback_reason"]
    compiled.clear()
    assert compiled.stats()["graphs"] == ()


@pytest.mark.gpu
def test_inputs_requiring_gradients_execute_prepared():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    compiled = fc.compile(lambda x: x * x, cache=False, graph=True)
    source = torch.tensor([3.0], device="cuda:0", requires_grad=True)
    for _ in range(2):
        compiled(source).sum().backward()
    torch.testing.assert_close(
        source.grad, torch.tensor([12.0], device="cuda:0")
    )
    assert (
        cast(Sequence[dict], compiled.stats()["graphs"])[-1]["fallback_reason"]
        == "Autograd outputs require prepared execution"
    )


def _graph_stats(compiled) -> dict:
    (stats,) = cast(Sequence[dict], compiled.stats()["graphs"])
    return stats


@pytest.mark.gpu
def test_kept_graph_results_are_detached_and_dropped_results_are_not():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    compiled = fc.compile(_source, cache=False, graph=True)
    x = torch.tensor([1.0, 2.0], device="cuda:0")
    y = torch.tensor([3.0, 5.0], device="cuda:0")
    try:
        for step in range(4):
            compiled(x + step, y)
        dropped = _graph_stats(compiled)
        assert dropped["state"] == "captured"
        assert dropped["capture_stats"]["replays"] >= 2
        assert dropped["capture_stats"]["detached_outputs"] == 0
        results = [compiled(x + step, y) for step in range(4)]
        for step, result in enumerate(results):
            torch.testing.assert_close(result["value"], (x + step) * 2 + y)
            torch.testing.assert_close(
                result["nested"][0][0], (x + step) * 2 + y - 1
            )
            assert result["nested"][1:] == (None, "static")
        kept = _graph_stats(compiled)["capture_stats"]
        assert kept["detached_outputs"] >= 3
    finally:
        compiled.clear()
    assert compiled.stats()["graphs"] == ()


@pytest.mark.gpu
def test_chained_callables_share_the_arena_and_read_results_in_place():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from fhelium.backend.runner import shared_arena

    first = fc.compile(lambda x: (x * 3 + 1) * x, cache=False, graph=True)
    second = fc.compile(lambda z: (z - 2) * z + z, cache=False, graph=True)
    x = torch.linspace(-1.0, 1.0, 4096, device="cuda:0")
    try:
        staged = []
        for step in range(5):
            source = x + step
            z = first(source)
            result = second(z)
            expected = (source * 3 + 1) * source
            torch.testing.assert_close(
                result, (expected - 2) * expected + expected
            )
            staged.append(
                _graph_stats(second)["capture_stats"]["staged_copies"]
            )
        stats = [_graph_stats(c)["capture_stats"] for c in (first, second)]
        assert all(item["replays"] >= 2 for item in stats)
        # After capture, the second callable reads the first one's result in place.
        assert staged[-1] == staged[2]
        assert shared_arena(0).bytes() >= max(
            item["arena_bytes"] for item in stats
        )
    finally:
        first.clear()
        second.clear()


def test_refreshed_plaintext_materials_keep_graph_captures():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from fhelium import CkksConfig
    from fhelium.eager import Engine

    config = CkksConfig.catalog_suffix_chain(
        logN=8,
        scale_bits=40,
        depth=2,
        p_count=1,
        catalog_logN=14,
        enforce_security_budget=False,
    )
    engine = Engine(config, rng_seed=761)
    secret = engine.create_secret_key(device="cuda:0")
    public = engine.create_public_key(secret, device="cuda:0")
    clear = torch.linspace(-0.5, 0.5, config.num_slots, dtype=torch.float64)
    value = engine.encrypt_message(clear, public, device="cuda:0")

    def source(current, weight):
        plain = engine.prepare_compressed_plaintext(
            weight.repeat(2),
            depth=current.depth,
            scale=current.scale,
            device=current.device,
        )
        return engine.ntt_domain_to_coefficient_domain(
            engine.multiply_plaintext(
                engine.coefficient_domain_to_ntt_domain(current), plain
            )
        )

    compiled = fc.compile(source, cache=False, graph=True)
    try:
        for step in range(7):
            # A new weight Tensor at every call makes preparation rebind its plaintext.
            weight = torch.tensor(
                [0.5, -0.25, 0.125, 1.0], dtype=torch.float64, device="cuda:0"
            ) * (step + 1)
            torch.testing.assert_close(
                engine.decrypt_message(compiled(value, weight), secret).real,
                engine.decrypt_message(source(value, weight), secret).real,
                rtol=0,
                atol=1e-9,
            )
        stats = _graph_stats(compiled)["capture_stats"]
        assert stats["captures"] == 1
        assert stats["replays"] >= 3
    finally:
        compiled.clear()


@pytest.mark.gpu
def test_auto_segments_decide_and_keep_results():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    compiled = fc.compile(_source, cache=False)
    x = torch.tensor([1.0, 2.0], device="cuda:0")
    y = torch.tensor([3.0, 5.0], device="cuda:0")
    try:
        for step in range(8):
            result = compiled(x + step, y)
            torch.testing.assert_close(result["value"], (x + step) * 2 + y)
        torch.cuda.synchronize()
        compiled(x, y)
        segments = _graph_stats(compiled)["segments"]
        assert segments
        for segment in segments:
            assert segment["choice"] in ("graph", "direct")
            assert segment["variants"] == (segment["choice"] == "graph")
    finally:
        compiled.clear()


@pytest.mark.gpu
def test_alternating_key_arguments_are_read_in_place():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from fhelium import CkksConfig, Preset
    from fhelium.eager import Engine

    device = torch.device("cuda:0")
    engine = Engine(
        CkksConfig.parse(Preset.slots8192_scale40_depth7), rng_seed=3
    )
    secrets = [engine.create_secret_key(device=device) for _ in range(2)]
    keys = [engine.create_rotation_key(1, sk, device=device) for sk in secrets]
    message = torch.linspace(0, 1, engine.num_slots, device=device)
    inputs = [
        engine.encrypt_message(
            message, engine.create_public_key(sk, device=device), device=device
        )
        for sk in secrets
    ]

    def rotate_twice(ct, key):
        x = engine.rotate_with_key(ct, key)
        return engine.add(x, engine.rotate_with_key(x, key))

    expected = [rotate_twice(inputs[i], keys[i]) for i in range(2)]
    compiled = fc.compile(rotate_twice, cache=False, graph=True)
    try:
        for step in range(8):
            i = step % 2
            out = compiled(inputs[i], keys[i])
            assert torch.equal(out.data, expected[i].data)
            del out
        stats = _graph_stats(compiled)["capture_stats"]
        # One capture per key set; each replay stages only the ciphertext.
        assert stats["captures"] == 2
        assert stats["staged_copies"] == stats["replays"]
    finally:
        compiled.clear()
