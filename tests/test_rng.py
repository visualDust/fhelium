import pytest
import torch

from fhelium import CkksConfig, CkksEngine, Preset
from fhelium.rng import Csprng


def _require_cuda_devices(count: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if torch.cuda.device_count() < count:
        pytest.skip(f"requires at least {count} CUDA devices")


@pytest.mark.gpu
def test_csprng_adapter_shapes_ranges_and_repeated_channels() -> None:
    _require_cuda_devices(2)
    generator = Csprng(
        num_coefs=64,
        num_channels=[2, 3],
        num_repeating_channels=2,
        devices=["cuda:0", "cuda:1"],
        seed=20260530,
        nonce=7,
    )

    integers = generator.randint(
        [[17, 19, 23, 29], [31, 37, 23, 29]],
        repeats=2,
    )
    assert [tuple(value.shape) for value in integers] == [(4, 64), (4, 64)]
    assert integers[0].device == torch.device("cuda:0")
    assert integers[1].device == torch.device("cuda:1")
    assert torch.equal(integers[0][-2:].cpu(), integers[1][-2:].cpu())
    assert torch.all(integers[0][0] < 17)
    assert torch.all(integers[1][1] < 37)

    gaussian = generator.discrete_gaussian(non_repeats=[1, 2], repeats=1)
    assert [tuple(value.shape) for value in gaussian] == [(2, 64), (3, 64)]
    assert torch.equal(gaussian[0][-1].cpu(), gaussian[1][-1].cpu())

    raw = generator.randbytes(shares=[1, 1], repeats=1, reshape=True)
    assert [tuple(value.shape) for value in raw] == [(2, 16, 16), (2, 16, 16)]
    assert raw[0].dtype == torch.int64


def test_cpu_csprng_shapes_ranges_and_engine_integration() -> None:
    generator = Csprng(
        num_coefs=64,
        num_channels=[3],
        num_repeating_channels=2,
        devices=["cpu"],
        seed=20260530,
        nonce=7,
    )
    integers = generator.randint([[17, 19, 23]], repeats=2)
    gaussian = generator.discrete_gaussian(non_repeats=1, repeats=2)
    raw = generator.randbytes(shares=[1], repeats=1, reshape=True)
    assert integers[0].shape == (3, 64)
    assert integers[0].device.type == "cpu"
    assert torch.all(integers[0][0] < 17)
    assert gaussian[0].shape == (3, 64)
    assert raw[0].shape == (2, 16, 16)

    config = CkksConfig.parse(Preset.slots8192_scale40_levels7_int64)
    engine = CkksEngine(config, device="cpu")
    assert isinstance(engine.rng, Csprng)
    assert engine.device.type == "cpu"


def test_cpu_engine_rng_uses_the_configured_int32_dtype() -> None:
    config = CkksConfig.parse(
        Preset.slots8192_scale30_levels9_int64,
        buffer_bit_length=30,
        scale_bits=25,
        num_scale_primes=3,
        enforce_security_budget=False,
    )
    engine = CkksEngine(config, device="cpu")
    assert isinstance(engine.rng, Csprng)
    assert engine.rng.torch_dtype is torch.int32
    ciphertext = engine.encrypt_message(torch.zeros(16, dtype=torch.float64))
    assert ciphertext.data.dtype is torch.int32


def test_engine_canonicalizes_unindexed_local_devices() -> None:
    config = CkksConfig.parse(Preset.slots8192_scale40_levels7_int64)
    assert CkksEngine(config, device="cpu:0").device == torch.device("cpu")
    if torch.cuda.is_available():
        engine = CkksEngine(config, device="cuda")
        assert engine.device == torch.device(
            "cuda", torch.cuda.current_device()
        )


@pytest.mark.parametrize("num_coefs", [0, 15])
def test_csprng_rejects_invalid_coefficient_counts(num_coefs: int) -> None:
    with pytest.raises(ValueError, match="positive multiple of 4"):
        Csprng(num_coefs=num_coefs, devices=["cpu"])


def test_csprng_rejects_mixed_device_types_and_negative_channels() -> None:
    with pytest.raises(ValueError, match="all CPU or all CUDA"):
        Csprng(
            num_coefs=16,
            num_channels=[1, 1],
            devices=["cpu", "cuda:0"],
        )
    with pytest.raises(ValueError, match="num_channels"):
        Csprng(num_coefs=16, num_channels=[-1], devices=["cpu"])
    with pytest.raises(ValueError, match="num_repeating_channels"):
        Csprng(
            num_coefs=16,
            num_channels=[1],
            num_repeating_channels=-1,
            devices=["cpu"],
        )


def test_ckks_engine_owns_seeded_rng_configuration() -> None:
    config = CkksConfig.parse(Preset.slots8192_scale40_levels7_int64)
    first = CkksEngine(
        config,
        device="cpu",
        rng_seed=7,
        rng_nonce=11,
    )
    second = CkksEngine(
        config,
        device="cpu",
        rng_seed=7,
        rng_nonce=11,
    )

    assert isinstance(first.rng, Csprng)
    assert torch.equal(first.rng.randbytes()[0], second.rng.randbytes()[0])


def test_ckks_engine_rejects_noninteger_rng_stream_material() -> None:
    with pytest.raises(TypeError, match="rng_seed must be an integer"):
        CkksEngine(
            Preset.slots8192_scale40_levels7_int64,
            device="cpu",
            rng_seed=True,  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="rng_seed must be an integer"):
        CkksEngine(
            Preset.slots8192_scale40_levels7_int64,
            device="cpu",
            rng_seed="7",  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="rng_nonce must be an integer"):
        CkksEngine(
            Preset.slots8192_scale40_levels7_int64,
            device="cpu",
            rng_nonce=1.5,  # type: ignore[arg-type]
        )


@pytest.mark.gpu
def test_ckks_engine_passes_configured_gaussian_sigma() -> None:
    _require_cuda_devices(1)
    config = CkksConfig.parse(
        Preset.slots8192_scale40_levels7_int64,
        sigma=4.25,
        enforce_security_budget=False,
    )
    engine = CkksEngine(config, device="cuda:0")
    assert isinstance(engine.rng, Csprng)
    assert engine.rng.sigma == 4.25
