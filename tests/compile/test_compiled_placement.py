"""Compiled functions follow the placement written in the program and interact with Residency between calls."""

from typing import cast

import pytest
import torch

from fhelium import CkksConfig, compile as fc
from fhelium.compile import CaptureError
from fhelium.eager import Engine
from fhelium.residency import PAGEABLE_HOST, ResidencyManager, cuda_location


def _engine(device):
    config = CkksConfig.catalog_suffix_chain(
        logN=8,
        scale_bits=40,
        depth=2,
        p_count=1,
        catalog_logN=14,
        enforce_security_budget=False,
    )
    engine = Engine(config, rng_seed=761)
    secret = engine.create_secret_key(device=device)
    public = engine.create_public_key(secret, device=device)
    clear = torch.linspace(-0.02, 0.03, config.num_slots, dtype=torch.float64)
    return engine, public, engine.encrypt_message(clear, public, device=device)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("graph", [False, "auto"])
def test_host_offload_inside_a_compiled_function_matches_eager(graph):
    engine, _, value = _engine("cuda:0")

    def source(current):
        pageable = current.to("cpu")
        pinned = pageable.pin_memory()
        host = engine.add(pageable, engine.negate(pinned))
        return engine.add(engine.add(current, current), host.to("cuda:0"))

    expected = source(value)
    compiled = fc.compile(source, cache=False, graph=graph)
    for _ in range(3):
        actual = compiled(value)
        assert torch.equal(actual.data, expected.data)
        assert actual.data.device == value.data.device
    # The host transfers run on copy streams.
    stats = cast(dict[str, float], compiled.stats()["overlap"])
    assert stats["h2d_transfers"] > 0 and stats["d2h_transfers"] > 0


def test_residency_calls_inside_capture_raise():
    engine, public, value = _engine("cpu")
    manager = ResidencyManager()
    handle = manager.adopt(value.clone(), at=PAGEABLE_HOST)

    def source(current):
        manager.ensure(handle, PAGEABLE_HOST)
        return engine.add(current, current)

    with pytest.raises(CaptureError, match="between compiled calls"):
        fc.compile(source, cache=False, graph=False)(value)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_residency_moves_values_between_compiled_calls():
    engine, public, value = _engine("cuda:0")
    device = cuda_location("cuda:0")
    manager = ResidencyManager({device: 1 << 30})
    handle = manager.adopt(value.cpu(), at=PAGEABLE_HOST)

    def source(current):
        # The NTT-domain intermediates are temporaries in the shared arena.
        return engine.ntt_domain_to_coefficient_domain(
            engine.add(
                engine.coefficient_domain_to_ntt_domain(current),
                engine.coefficient_domain_to_ntt_domain(
                    engine.add(current, current)
                ),
            )
        )

    compiled = fc.compile(source, cache=False)
    stream = torch.cuda.current_stream()
    results = []
    for _ in range(3):
        manager.move(handle, device)
        lease = manager.acquire((handle,), at=device, consumer_stream=stream)
        results.append(compiled(lease.values[handle]))
        lease.release()
        manager.move(handle, PAGEABLE_HOST)
    expected = source(value)
    for result in results:
        assert torch.equal(result.data, expected.data)
    (location,) = (
        item for item in manager.snapshot().locations if item.location == device
    )
    assert location.compiled_execution_bytes > 0
