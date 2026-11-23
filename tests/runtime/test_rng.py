import torch

from fhelium import CkksConfig, Preset
from fhelium.backend.random.stream import RandomStream
from fhelium.eager import Engine


def test_seeded_streams_reproduce_and_advance() -> None:
    first = RandomStream("cpu", sigma=3.19, seed=7, nonce=11)
    second = RandomStream("cpu", sigma=3.19, seed=7, nonce=11)
    other = RandomStream("cpu", sigma=3.19, seed=7, nonce=11, stream_id=1)
    sample = first.discrete_gaussian([64])
    assert torch.equal(sample, second.discrete_gaussian([64]))
    assert not torch.equal(sample, other.discrete_gaussian([64]))
    assert int(first.seed[4]) == 12
    assert not torch.equal(sample, first.discrete_gaussian([64]))
    entropy = [RandomStream("cpu", sigma=3.19).seed[:4] for _ in range(2)]
    assert not torch.equal(*entropy)


def test_cpu_engine_rng_uses_the_configured_int32_dtype() -> None:
    config = CkksConfig.parse(Preset.slots8192_scale25_depth14_int32)
    engine = Engine(config)
    ciphertext = engine.encrypt_message(
        torch.zeros(16, dtype=torch.float64), device="cpu"
    )
    assert ciphertext.data.dtype is torch.int32


def test_engine_owns_seeded_rng_configuration() -> None:
    config = CkksConfig.parse(Preset.slots8192_scale40_depth7_int64)
    first = Engine(config, rng_seed=7, rng_nonce=11)
    second = Engine(config, rng_seed=7, rng_nonce=11)
    assert torch.equal(first._rng_for("cpu").seed, second._rng_for("cpu").seed)
    assert torch.equal(
        first.create_secret_key(device="cpu").data,
        second.create_secret_key(device="cpu").data,
    )


def test_engine_passes_configured_gaussian_sigma() -> None:
    config = CkksConfig.parse(
        Preset.slots8192_scale40_depth7_int64,
        sigma=4.25,
        enforce_security_budget=False,
    )
    assert Engine(config)._rng_for("cpu").sigma == 4.25
