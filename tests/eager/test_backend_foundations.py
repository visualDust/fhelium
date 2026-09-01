"""Essential tests for backend resources and eager specialization."""

from __future__ import annotations

import gc
import weakref
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from fhelium import (
    CompressedPlaintext,
    Plaintext,
    Preset,
)
from fhelium.legacy.engine import CkksEngine
from fhelium.eager import Engine
from fhelium.runtime import ReusableValueBuffer
from fhelium.values import Ciphertext, RelinearizationKey


def test_eager_factories_follow_torch_default_device() -> None:
    engine = Engine(Preset.slots8192_scale40_levels7_int64)

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
        Preset.slots8192_scale40_levels7_int64,
        rng_seed=47,
        rng_nonce=19,
    )

    with ThreadPoolExecutor(max_workers=4) as pool:
        keys = tuple(pool.map(lambda _: engine.create_secret_key(), range(4)))

    payloads = {key.data.cpu().numpy().tobytes() for key in keys}
    assert len(payloads) == len(keys)


@pytest.mark.gpu
def test_eager_boundary_device_selection_is_caller_controlled() -> None:
    engine = Engine(Preset.slots8192_scale40_levels7_int64)
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
        Preset.slots8192_scale40_levels7_int64,
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


@pytest.mark.gpu
def test_eager_cuda_add_matches_handwritten_engine() -> None:
    engine = CkksEngine(
        Preset.slots8192_scale40_levels7_int64,
        device="cuda:0",
    )
    runtime = Engine(
        engine.config,
        ntt_backend=engine.ntt_backend_name,
    )
    value = engine.encrypt_message(
        torch.full(
            (engine.num_slots,),
            0.03125,
            dtype=torch.float64,
            device=engine.device,
        ),
        engine.public_key,
    )

    assert torch.equal(
        runtime.add(value, value).data, engine.add(value, value).data
    )


def test_eager_key_graphs_match_handwritten_engine() -> None:
    engine = CkksEngine(
        Preset.slots8192_scale40_levels7_int64,
        device="cpu",
        rng_seed=17,
    )
    runtime = Engine(
        engine.config,
        ntt_backend=engine.ntt_backend_name,
    )
    secret_key = engine.secret_key
    message = torch.linspace(
        -0.05,
        0.05,
        engine.num_slots,
        dtype=torch.float64,
    )
    ciphertext = engine.encrypt(engine.encode(message))

    relinearization_key = engine.create_relinearization_key(secret_key)
    ntt = engine.coefficient_domain_to_ntt_domain(ciphertext)
    product = engine.multiply(ntt, ntt)
    assert torch.equal(
        runtime.relinearize(product, relinearization_key).data,
        engine.relinearize(product, relinearization_key).data,
    )

    key_switch_key = engine.create_key_switch_key(secret_key, secret_key)
    assert torch.equal(
        runtime.switch_key(ciphertext, key_switch_key).data,
        engine.switch_key(ciphertext, key_switch_key).data,
    )

    rotation_key = engine.create_rotation_key(1, secret_key)
    assert torch.equal(
        runtime.rotate_with_key(ciphertext, rotation_key).data,
        engine.rotate_with_key(ciphertext, rotation_key).data,
    )

    conjugation_key = engine.create_conjugation_key(secret_key)
    assert torch.equal(
        runtime.conjugate(ciphertext, conjugation_key).data,
        engine.conjugate(ciphertext, conjugation_key).data,
    )


def test_eager_runtime_matches_boundary_and_computation_sequence_on_cpu() -> (
    None
):
    preset = Preset.slots8192_scale40_levels7_int64
    engine = CkksEngine(
        preset,
        device="cpu",
        rng_seed=29,
        rng_nonce=7,
    )
    runtime = Engine(
        preset,
        rng_seed=29,
        rng_nonce=7,
    )
    message = torch.linspace(-0.03, 0.03, engine.num_slots, dtype=torch.float64)

    encoded_engine = engine.encode(message)
    encoded_runtime = runtime.encode(message)
    assert encoded_engine.data is not None
    assert encoded_runtime.data is not None
    assert torch.equal(encoded_runtime.data, encoded_engine.data)
    secret_engine = engine.create_secret_key()
    secret_runtime = runtime.create_secret_key()
    assert torch.equal(secret_runtime.data, secret_engine.data)
    public_engine = engine.create_public_key(secret_engine)
    public_runtime = runtime.create_public_key(secret_runtime)
    assert torch.equal(public_runtime.data, public_engine.data)
    public_qp_engine = engine.create_public_key(
        secret_engine,
        modulus_basis="QP",
    )
    public_qp_runtime = runtime.create_public_key(
        secret_runtime,
        modulus_basis="QP",
    )
    assert torch.equal(public_qp_runtime.data, public_qp_engine.data)
    ciphertext_engine = engine.encrypt(encoded_engine, public_engine)
    ciphertext_runtime = runtime.encrypt(encoded_runtime, public_runtime)
    assert torch.equal(ciphertext_runtime.data, ciphertext_engine.data)
    decrypted_runtime = runtime.decrypt(ciphertext_runtime, secret_runtime)
    decrypted_engine = engine.decrypt(ciphertext_engine, secret_engine)
    assert decrypted_runtime.data is not None
    assert decrypted_engine.data is not None
    assert torch.equal(decrypted_runtime.data, decrypted_engine.data)

    rhs = engine.encrypt_message(
        torch.full((engine.num_slots,), 0.01, dtype=torch.float64),
        public_engine,
    )
    for observed, expected in (
        (
            runtime.add(ciphertext_engine, rhs),
            engine.add(ciphertext_engine, rhs),
        ),
        (
            runtime.subtract(ciphertext_engine, rhs),
            engine.subtract(ciphertext_engine, rhs),
        ),
        (runtime.negate(ciphertext_engine), engine.negate(ciphertext_engine)),
    ):
        assert torch.equal(observed.data, expected.data)

    ntt = engine.coefficient_domain_to_ntt_domain(ciphertext_engine)
    product = runtime.multiply(ntt, ntt)
    expected_product = engine.multiply(ntt, ntt)
    assert torch.equal(product.data, expected_product.data)

    prepared_plaintext = engine.prepare_plaintext_for_multiplication(
        engine.encode(
            torch.full((engine.num_slots,), 0.02, dtype=torch.float64)
        )
    )
    assert prepared_plaintext.data is not None
    unique_count = 32
    compact = prepared_plaintext.data[..., :unique_count].clone()
    dense_data = compact.repeat(
        *([1] * (compact.ndim - 1)),
        prepared_plaintext.data.size(-1) // unique_count,
    )
    dense_plaintext = Plaintext(
        message=None,
        level=prepared_plaintext.level,
        scale=prepared_plaintext.scale,
        data=dense_data,
        representation="rns",
        polynomial_domain=prepared_plaintext.polynomial_domain,
        modulus_basis=prepared_plaintext.modulus_basis,
        residue_representation=prepared_plaintext.residue_representation,
        prime_ids=prepared_plaintext.prime_ids,
    )
    compressed = CompressedPlaintext.from_plaintext(
        dense_plaintext,
        unique_count=unique_count,
        compression_layout="cyclic",
    )
    assert torch.equal(
        runtime.multiply_plaintext(ntt, compressed).data,
        engine.multiply_plaintext(ntt, compressed).data,
    )
    relinearization_key = engine.create_relinearization_key(secret_engine)
    relinearized = runtime.relinearize(product, relinearization_key)
    expected_relinearized = engine.relinearize(
        expected_product,
        relinearization_key,
    )
    assert torch.equal(relinearized.data, expected_relinearized.data)
    assert torch.equal(
        runtime.rescale_to_next_level(relinearized).data,
        engine.rescale_to_next_level(expected_relinearized).data,
    )
    assert torch.equal(
        runtime.mod_switch_to_level(ciphertext_engine, 2).data,
        engine.mod_switch_to_level(ciphertext_engine, 2).data,
    )
    assert runtime.reinterpret_at_scale(ciphertext_engine, 2.0**39).scale == (
        2.0**39
    )

    rotation_keys = [
        engine.create_rotation_key(step, secret_engine) for step in (1, -3)
    ]
    actual_rotations = runtime.rotate_many_with_keys(
        ciphertext_engine,
        rotation_keys,
    )
    expected_rotations = engine.rotate_many_with_keys(
        ciphertext_engine,
        rotation_keys,
        use_hoisting=True,
    )
    assert all(
        torch.equal(actual.data, expected.data)
        for actual, expected in zip(
            actual_rotations,
            expected_rotations,
            strict=True,
        )
    )
    singleton = runtime.sum_ciphertexts([ciphertext_engine])
    assert torch.equal(singleton.data, ciphertext_engine.data)
    assert singleton.data.data_ptr() != ciphertext_engine.data.data_ptr()


@pytest.mark.gpu
def test_eager_runtime_cuda_exact_differential_sequence() -> None:
    preset = Preset.slots8192_scale40_levels7_int64
    engine = CkksEngine(
        preset,
        device="cuda:0",
        rng_seed=37,
        rng_nonce=11,
    )
    runtime = Engine(
        preset,
        rng_seed=37,
        rng_nonce=11,
    )
    message = torch.linspace(-0.02, 0.02, engine.num_slots, dtype=torch.float64)
    encoded_engine = engine.encode(message)
    secret_engine = engine.create_secret_key()
    public_engine = engine.create_public_key(secret_engine)
    ciphertext_engine = engine.encrypt(encoded_engine, public_engine)
    ciphertext_runtime = ciphertext_engine

    ntt_engine = engine.coefficient_domain_to_ntt_domain(ciphertext_engine)
    ntt_runtime = runtime.coefficient_domain_to_ntt_domain(ciphertext_runtime)
    assert isinstance(ntt_runtime, Ciphertext)
    assert torch.equal(ntt_runtime.data, ntt_engine.data)
    product_engine = engine.multiply(ntt_engine, ntt_engine)
    product_runtime = runtime.multiply(ntt_runtime, ntt_runtime)
    assert torch.equal(product_runtime.data, product_engine.data)
    key_engine = engine.create_relinearization_key(secret_engine)
    actual = runtime.rescale_to_next_level(
        runtime.relinearize(product_runtime, key_engine)
    )
    expected = engine.rescale_to_next_level(
        engine.relinearize(product_engine, key_engine)
    )
    assert torch.equal(actual.data, expected.data)
    rotation_keys = [
        engine.create_rotation_key(step, secret_engine) for step in (1, -3)
    ]
    actual_rotations = runtime.rotate_many_with_keys(
        ciphertext_runtime,
        rotation_keys,
        use_hoisting=True,
    )
    expected_rotations = engine.rotate_many_with_keys(
        ciphertext_engine,
        rotation_keys,
        use_hoisting=True,
    )
    assert all(
        torch.equal(actual_rotation.data, expected_rotation.data)
        for actual_rotation, expected_rotation in zip(
            actual_rotations,
            expected_rotations,
            strict=True,
        )
    )


def test_eager_key_lifecycle_controls_dependent_operations() -> None:
    preset = Preset.slots8192_scale40_levels7_int64
    engine = CkksEngine(preset, device="cpu", rng_seed=41, rng_nonce=13)
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
    preset = Preset.slots8192_scale40_levels7_int64
    engine = CkksEngine(preset, device="cpu", rng_seed=43, rng_nonce=17)
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
    runtime = Engine(Preset.slots8192_scale40_levels7_int64)
    message = torch.zeros(runtime.num_slots, dtype=torch.float64)
    level_zero = runtime.encode(message, level=0)
    level_one = runtime.encode(message, level=1)
    assert level_zero.level == 0
    assert level_one.level == 1

    secret = runtime.create_secret_key()
    first_key = runtime.create_public_key(secret)
    runtime.encrypt(level_zero, first_key)
    first_key_reference = weakref.ref(first_key)
    second_key = runtime.create_public_key(secret)
    runtime.encrypt(level_zero, second_key)
    del first_key
    gc.collect()
    assert first_key_reference() is None


def test_eager_metadata_operations_preserve_payload_and_update_state() -> None:
    runtime = Engine(Preset.slots8192_scale40_levels7_int64)
    ciphertext = runtime.encrypt_message(
        torch.full((runtime.num_slots,), 0.01, dtype=torch.float64)
    )
    switched = runtime.mod_switch_to_next_level(ciphertext)
    assert torch.equal(switched.data, ciphertext.data[..., 1:, :])

    reinterpreted = runtime.reinterpret_at_scale(
        ciphertext, ciphertext.scale * 2
    )
    assert reinterpreted.scale == ciphertext.scale * 2
    assert torch.equal(reinterpreted.data, ciphertext.data)
