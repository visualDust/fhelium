"""Essential tests for backend resources and eager specialization."""

from __future__ import annotations

import gc
import weakref
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from fhelium import (
    Preset,
)
from fhelium.eager import Engine
from fhelium.runtime import ReusableValueBuffer
from fhelium.values import Ciphertext, RelinearizationKey


def test_eager_factories_follow_torch_default_device() -> None:
    engine = Engine(Preset.slots8192_scale40_depth7)

    encoded = engine.encode([0.125])
    secret_key = engine.create_secret_key()
    public_key = engine.create_public_key(secret_key)

    assert encoded.device == torch.get_default_device()
    assert secret_key.device == torch.get_default_device()
    assert public_key.device == torch.get_default_device()


def test_first_device_use_shares_one_serial_random_stream_across_threads() -> (
    None
):
    engine = Engine(
        Preset.slots8192_scale40_depth7,
        rng_seed=47,
        rng_nonce=19,
    )

    with ThreadPoolExecutor(max_workers=4) as pool:
        keys = tuple(pool.map(lambda _: engine.create_secret_key(), range(4)))

    payloads = {key.data.cpu().numpy().tobytes() for key in keys}
    assert len(payloads) == len(keys)


@pytest.mark.gpu
def test_eager_boundary_device_selection_is_caller_controlled() -> None:
    engine = Engine(Preset.slots8192_scale40_depth7)
    message = torch.tensor([0.03125], dtype=torch.float64)
    secret_key = engine.create_secret_key()
    public_key = engine.create_public_key(secret_key)
    encoded_cpu = engine.encode(message)

    ciphertext_cuda = engine.encrypt(
        encoded_cpu,
        public_key.to("cuda:0"),
        device="cuda:0",
    )
    assert ciphertext_cuda.device == torch.device("cuda:0")
    with pytest.raises(ValueError, match="automatic_key_replication"):
        engine.decrypt(ciphertext_cuda, secret_key)
    decrypted_cuda = engine.decrypt(ciphertext_cuda, secret_key.to("cuda:0"))
    decoded_cpu = engine.decode(decrypted_cuda, device="cpu", is_real=True)
    torch.testing.assert_close(
        decoded_cpu[0],
        message[0],
        atol=1e-5,
        rtol=0.0,
    )
    relinearization_key = engine.create_relinearization_key(secret_key)
    ciphertext_ntt = engine.coefficient_domain_to_ntt_domain(ciphertext_cuda)
    product = engine.multiply(ciphertext_ntt, ciphertext_ntt)
    with pytest.raises(ValueError, match="automatic_key_replication"):
        engine.relinearize(product, relinearization_key)


@pytest.mark.gpu
def test_eager_automatic_key_replication_is_opt_in() -> None:
    engine = Engine(
        Preset.slots8192_scale40_depth7,
        allow_automatic_key_replication=True,
    )
    secret_key = engine.create_secret_key(device="cpu")
    public_key = engine.create_public_key(secret_key)
    engine.set_secret_key(secret_key)
    engine.set_public_key(public_key)
    plaintext = engine.encode([0.03125], device="cpu")

    ciphertext = engine.encrypt(plaintext, device="cuda:0")
    decrypted = engine.decrypt(ciphertext)
    relinearization_key = engine.create_relinearization_key(secret_key)
    engine.set_relinearization_key(relinearization_key)
    ciphertext_ntt = engine.coefficient_domain_to_ntt_domain(ciphertext)
    product = engine.multiply(ciphertext_ntt, ciphertext_ntt)
    relinearized = engine.relinearize(product)

    assert ciphertext.device == torch.device("cuda:0")
    assert decrypted.device == torch.device("cuda:0")
    assert relinearized.device == torch.device("cuda:0")


def test_eager_key_lifecycle_controls_dependent_operations() -> None:
    preset = Preset.slots8192_scale40_depth7
    engine = Engine(preset, rng_seed=41, rng_nonce=13)
    runtime = Engine(
        preset,
        allow_automatic_key_generation=False,
    )
    runtime.set_secret_key(engine.secret_key)
    runtime.set_relinearization_key(engine.relinearization_key)
    ciphertext = engine.encrypt_message(
        torch.full((engine.num_slots,), 0.01, dtype=torch.float64),
        engine.public_key,
    )
    ntt = engine.coefficient_domain_to_ntt_domain(ciphertext)
    product = engine.multiply(ntt, ntt)
    runtime.relinearize(product)

    runtime.remove_evaluation_key("relinearization-key")
    with pytest.raises(KeyError):
        runtime.relinearize(product)

    runtime.set_relinearization_key(engine.relinearization_key)
    runtime.relinearize(product)
    replacement = runtime.create_secret_key()
    runtime.set_secret_key(replacement)
    with pytest.raises(KeyError):
        runtime.relinearize(product)


def test_eager_repeated_calls_and_mutated_key_binding_behavior() -> None:
    preset = Preset.slots8192_scale40_depth7
    engine = Engine(preset, rng_seed=43, rng_nonce=17)
    runtime = Engine(preset)
    ciphertext = engine.encrypt_message(
        torch.full((engine.num_slots,), 0.01, dtype=torch.float64),
        engine.public_key,
    )
    ntt = engine.coefficient_domain_to_ntt_domain(ciphertext)
    product = engine.multiply(ntt, ntt)
    key = engine.create_relinearization_key(engine.secret_key)

    assert torch.equal(
        runtime.add(ciphertext, ciphertext).data,
        runtime.add(ciphertext, ciphertext).data,
    )
    message = torch.zeros(runtime.num_slots, dtype=torch.float64)
    first_encoding = runtime.encode(message)
    second_encoding = runtime.encode(message)
    assert first_encoding.data is not None
    assert second_encoding.data is not None
    assert torch.equal(first_encoding.data, second_encoding.data)
    assert torch.equal(
        runtime.relinearize(product, key).data,
        runtime.relinearize(product, key).data,
    )

    buffer = ReusableValueBuffer.like(key)
    buffer.copy_from(key)
    stable_key = buffer.value
    assert isinstance(stable_key, RelinearizationKey)
    before_refresh = runtime.relinearize(product, stable_key)
    key.data.add_(1)
    buffer.copy_from(key)
    after_refresh = runtime.relinearize(product, stable_key)
    assert before_refresh.data.shape == product.data[:2].shape
    assert after_refresh.data.shape == product.data[:2].shape


def test_eager_boundary_calls_reuse_dynamic_state_and_release_replaced_key() -> (
    None
):
    runtime = Engine(Preset.slots8192_scale40_depth7)
    message = torch.zeros(runtime.num_slots, dtype=torch.float64)
    depth_zero = runtime.encode(message, depth=0)
    depth_one = runtime.encode(message, depth=1)
    assert depth_zero.depth == 0
    assert depth_one.depth == 1

    secret = runtime.create_secret_key()
    first_key = runtime.create_public_key(secret)
    runtime.encrypt(depth_zero, first_key)
    first_key_reference = weakref.ref(first_key)
    second_key = runtime.create_public_key(secret)
    runtime.encrypt(depth_zero, second_key)
    del first_key
    gc.collect()
    assert first_key_reference() is None


def test_eager_metadata_operations_preserve_payload_and_update_state() -> None:
    runtime = Engine(Preset.slots8192_scale40_depth7)
    ciphertext = runtime.encrypt_message(
        torch.full((runtime.num_slots,), 0.01, dtype=torch.float64)
    )
    switched = runtime.mod_switch_to_next_depth(ciphertext)
    rows = list(runtime.config.q_prime_ids(1))
    assert torch.equal(switched.data, ciphertext.data[..., rows, :])

    reinterpreted = runtime.reinterpret_at_scale(
        ciphertext, ciphertext.scale * 2
    )
    assert reinterpreted.scale == ciphertext.scale * 2
    assert torch.equal(reinterpreted.data, ciphertext.data)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_eager_limb_arithmetic_uses_declared_prime_rows(device: str) -> None:
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    engine = Engine(Preset.slots8192_scale40_depth7)
    moduli = torch.tensor(
        engine.config.q_moduli, dtype=torch.int64, device=device
    )
    data = (
        moduli.view(1, -1, 1)
        - torch.randint(
            1,
            65536,
            (2, len(moduli), engine.config.N),
            dtype=torch.int64,
            device=device,
        )
    ).to(engine.dtype)
    source = Ciphertext(
        data=data, depth=0, scale=2.0**40, prime_ids=tuple(range(len(moduli)))
    )

    def evaluate(value: Ciphertext) -> Ciphertext:
        combined = engine.add(value, value)
        engine.subtract_(combined, engine.negate(value))
        scaled = engine.multiply_integer_scalar(combined, 3)
        transformed = engine.coefficient_domain_to_ntt_domain(scaled)
        return engine.ntt_domain_to_coefficient_domain(
            engine.multiply(transformed, transformed)
        )

    expected = evaluate(source)
    for start, stop in ((0, 4), (2, 6), (4, 8)):
        shard = source.slice_limbs(start, stop)
        shard = shard.with_data(shard.data.contiguous())
        actual = evaluate(shard)
        assert actual.prime_ids == shard.prime_ids
        torch.testing.assert_close(
            actual.data, expected.data[..., start:stop, :], rtol=0, atol=0
        )


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_secret_key_weight_and_basis_conversion(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from fhelium import CkksConfig

    config = CkksConfig.suffix_chain(
        logN=8,
        q_depth_groups=((254279681, 253493249), (205651969, 204865537)),
        p_moduli=(257949697, 258605057),
        scale=2.0**40,
        galois_generator=5,
        enforce_security_budget=False,
    )
    target = CkksConfig.catalog_suffix_chain(
        logN=8,
        scale_bits=40,
        depth=2,
        p_count=1,
        catalog_logN=14,
        galois_generator=5,
        enforce_security_budget=False,
    )
    engine, other = Engine(config, rng_seed=327), Engine(target)
    for weight in (0, 32, config.N):
        key = engine.create_secret_key(hamming_weight=weight, device=device)
        from fhelium.backend.ntt import NttContext

        coefficients = key.data.clone()
        NttContext(engine.rns_context(device)).inverse_to_standard_(
            coefficients, include_p=True
        )
        row = coefficients[0]
        assert int((row != 0).sum()) == weight
        assert bool(
            ((row == 0) | (row == 1) | (row == config.q_moduli[0] - 1)).all()
        )
        converted = engine.convert_secret_key(key, target, dtype=torch.int64)
        returned = other.convert_secret_key(
            converted, config, dtype=torch.int32
        )
        moduli = torch.tensor(
            config.moduli, dtype=torch.int64, device=device
        ).view(-1, 1)
        torch.testing.assert_close(
            returned.data.to(torch.int64) % moduli,
            key.data.to(torch.int64) % moduli,
            rtol=0,
            atol=0,
        )
