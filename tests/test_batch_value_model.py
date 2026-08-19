"""Shape, compatibility, and aliasing invariants for dense CKKS batches."""

import pytest
import torch

from fhelium import Ciphertext, CkksEngine, Plaintext, Preset
from fhelium.engine import slot_embedding


def _ciphertext(value: int = 0, *, batch_shape=()) -> Ciphertext:
    data = torch.full((2, *batch_shape, 3, 8), value, dtype=torch.int64)
    return Ciphertext(
        data=data,
        level=2,
        scale=float(2**40),
        context_id="batch-test",
        prime_ids=(2, 3, 4),
    )


def test_ciphertext_batch_shape_uses_dimensions_between_components_and_rns():
    single = _ciphertext()
    batch = _ciphertext(batch_shape=(4, 5))

    assert single.batch_shape == ()
    assert single.batch_size == 1
    assert not single.is_batched
    assert batch.batch_shape == (4, 5)
    assert batch.batch_size == 20
    assert batch.is_batched
    assert batch.component_count == 2
    assert batch.limb_count == 3
    assert batch.ring_dimension == 8
    assert batch.c0.shape == (4, 5, 3, 8)


def test_ciphertext_stack_batch_is_explicit_copy_and_unbind_returns_views():
    values = [_ciphertext(1), _ciphertext(2), _ciphertext(3)]

    batch = Ciphertext.stack_batch(values)

    assert batch.data.shape == (2, 3, 3, 8)
    assert batch.batch_shape == (3,)
    assert batch.data.data_ptr() != values[0].data.data_ptr()

    items = batch.unbind_batch()
    assert len(items) == 3
    assert all(item.batch_shape == () for item in items)
    assert torch.equal(items[1].data, values[1].data)

    items[1].data[0, 0, 0] = 99
    assert batch.data[0, 1, 0, 0].item() == 99


def test_ciphertext_stack_batch_rejects_heterogeneous_exact_state():
    first = _ciphertext()
    second = _ciphertext()
    second.level += 1

    with pytest.raises(ValueError, match="level"):
        Ciphertext.stack_batch([first, second])


def test_stack_batch_adds_outer_axis_to_already_batched_values():
    values = [_ciphertext(value, batch_shape=(2,)) for value in (1, 2, 3)]

    stacked = Ciphertext.stack_batch(values)
    selected = stacked.select_batch(-1, dim=-1)

    assert stacked.batch_shape == (3, 2)
    assert selected.batch_shape == (3,)
    assert (
        selected.data.untyped_storage().data_ptr()
        == stacked.data.untyped_storage().data_ptr()
    )
    torch.testing.assert_close(selected.data, stacked.data[:, :, -1])


def test_ciphertext_limb_slice_preserves_all_batch_dimensions():
    batch = _ciphertext(batch_shape=(2, 4))

    sliced = batch.slice_limbs(1, 3)

    assert sliced.data.shape == (2, 2, 4, 2, 8)
    assert sliced.batch_shape == (2, 4)
    assert sliced.prime_ids == (3, 4)
    sliced.data[0, 0, 0, 0, 0] = 17
    assert batch.data[0, 0, 0, 1, 0].item() == 17


def test_plaintext_batch_shape_depends_on_layout():
    slots = Plaintext(
        message=torch.zeros(6, 4, dtype=torch.complex128),
        level=0,
        scale=float(2**40),
    )
    coefficients = Plaintext(
        message=None,
        level=0,
        scale=float(2**40),
        data=torch.zeros(6, 8, dtype=torch.int64),
        context_id="batch-test",
        representation="integer_coefficients",
        polynomial_domain="coefficient",
    )
    rns = Plaintext(
        message=None,
        level=0,
        scale=float(2**40),
        data=torch.zeros(6, 3, 8, dtype=torch.int64),
        context_id="batch-test",
        representation="rns",
        polynomial_domain="ntt",
        modulus_basis="Q",
        residue_representation="montgomery",
        prime_ids=(2, 3, 4),
    )

    assert slots.batch_shape == (6,)
    assert coefficients.batch_shape == (6,)
    assert rns.batch_shape == (6,)


def test_plaintext_stack_batch_is_explicit_copy_and_unbind_returns_views():
    values = [
        Plaintext(
            message=torch.full((4,), value, dtype=torch.float64),
            level=0,
            scale=float(2**40),
        )
        for value in (1, 2, 3)
    ]

    batch = Plaintext.stack_batch(values)
    items = batch.unbind_batch()

    assert batch.batch_shape == (3,)
    assert batch.message is not None
    assert batch.message.shape == (3, 4)
    assert batch.message.data_ptr() != values[0].message.data_ptr()
    assert len(items) == 3
    items[1].message[0] = 17
    assert batch.message[1, 0].item() == 17


def test_scalar_slot_plaintexts_cannot_be_batch_stacked_ambiguously():
    values = [
        Plaintext(message=torch.tensor(value), level=0, scale=float(2**40))
        for value in (2.0, 3.0)
    ]

    with pytest.raises(ValueError, match="repeat-to-all-slots semantics"):
        Plaintext.stack_batch(values)


def test_zero_sized_batch_dimensions_are_rejected():
    with pytest.raises(ValueError, match="batch dimensions must be nonzero"):
        Ciphertext(
            data=torch.empty(2, 0, 3, 8, dtype=torch.int64),
            level=0,
            scale=float(2**40),
            context_id="ctx",
            prime_ids=(0, 1, 2),
        )
    with pytest.raises(ValueError, match="batch dimensions must be nonzero"):
        Plaintext(
            message=torch.empty(0, 8),
            level=0,
            scale=float(2**40),
        )


def test_slot_embedding_preserves_leading_batch_dimensions():
    messages = torch.randn(3, 2, 4, dtype=torch.float64)
    slots = slot_embedding.make_slot_tensor(messages, num_slots=8, device="cpu")

    assert slots.shape == (3, 2, 8)
    assert torch.equal(slots[..., :4], messages)
    assert torch.count_nonzero(slots[..., 4:]) == 0

    encoded = slot_embedding.inverse_embed_slots(
        slots,
        device="cpu",
    )
    decoded = slot_embedding.embed_coefficients(
        encoded,
    )
    assert encoded.shape == (3, 2, 16)
    assert decoded.shape == (3, 2, 16)


@pytest.fixture(scope="module")
def batch_engine():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    return CkksEngine(Preset.slots8192_scale40_levels7_int64, device="cuda:0")


@pytest.fixture(scope="module")
def cpu_batch_engine():
    return CkksEngine(
        Preset.slots8192_scale40_levels7_int64,
        device="cpu",
        ntt_backend="radix2_indexed",
    )


def _messages(device: torch.device) -> torch.Tensor:
    return torch.stack(
        (
            torch.linspace(-0.002, 0.002, 32, dtype=torch.float64),
            torch.linspace(0.003, -0.003, 32, dtype=torch.float64),
        )
    ).to(device)


def test_sum_ciphertext_batch_matches_cleartext_and_preserves_input(
    cpu_batch_engine: CkksEngine,
) -> None:
    messages = torch.stack(
        tuple(
            torch.full((16,), value, dtype=torch.float64)
            for value in (0.01, -0.02, 0.03)
        )
    )
    batch = cpu_batch_engine.encrypt_message(messages)
    before = batch.data.clone()

    result = cpu_batch_engine.sum_ciphertext_batch(batch)

    assert result.batch_shape == ()
    assert torch.equal(batch.data, before)
    torch.testing.assert_close(
        cpu_batch_engine.decrypt_message(result, is_real=True)[:16],
        messages.sum(dim=0),
        atol=2e-5,
        rtol=0,
    )


def test_sum_ciphertext_batch_reduces_selected_axis(
    cpu_batch_engine: CkksEngine,
) -> None:
    messages = torch.full((2, 3, 16), 0.001, dtype=torch.float64)
    batch = cpu_batch_engine.encrypt_message(messages)

    result = cpu_batch_engine.sum_ciphertext_batch(batch, dim=-1)

    assert result.batch_shape == (2,)
    torch.testing.assert_close(
        cpu_batch_engine.decrypt_message(result, is_real=True)[..., :16],
        messages.sum(dim=1),
        atol=2e-5,
        rtol=0,
    )


def test_sum_ciphertext_batch_rejects_unbatched_or_invalid_axis(
    cpu_batch_engine: CkksEngine,
) -> None:
    unbatched = cpu_batch_engine.encrypt_message(
        torch.zeros(16, dtype=torch.float64)
    )
    with pytest.raises(ValueError, match="requires a batched value"):
        cpu_batch_engine.sum_ciphertext_batch(unbatched)
    batched = Ciphertext.stack_batch((unbatched, unbatched))
    with pytest.raises(IndexError, match="outside shape"):
        cpu_batch_engine.sum_ciphertext_batch(batched, dim=1)


@pytest.mark.gpu
def test_multidimensional_batch_public_facades_preserve_shape(batch_engine):
    base = _messages(batch_engine.device)
    messages = torch.stack((base, -base, 0.5 * base), dim=1)

    lazy = batch_engine.plaintext(messages)
    encoded = batch_engine.encode(messages)
    composed = batch_engine.encrypt(encoded)
    composed_plaintext = batch_engine.decrypt(composed)
    composed_decoded = batch_engine.decode(composed_plaintext, is_real=True)
    direct = batch_engine.encrypt_message(messages)
    doubled = batch_engine.add(direct, direct)
    direct_decoded = batch_engine.decrypt_message(doubled, is_real=True)

    assert lazy.batch_shape == messages.shape[:-1]
    assert encoded.batch_shape == messages.shape[:-1]
    assert composed_plaintext.batch_shape == messages.shape[:-1]
    assert direct.batch_shape == messages.shape[:-1]
    assert direct.data.shape[:3] == (2, 2, 3)
    assert direct_decoded.shape == (2, 3, batch_engine.num_slots)
    torch.testing.assert_close(
        composed_decoded[..., : messages.size(-1)].to(messages.device),
        messages,
        atol=1e-5,
        rtol=0,
    )
    torch.testing.assert_close(
        direct_decoded[..., : messages.size(-1)].to(messages.device),
        2 * messages,
        atol=1e-5,
        rtol=0,
    )


@pytest.mark.gpu
def test_batched_ntt_matches_independent_single_value_calls(batch_engine):
    messages = _messages(batch_engine.device)
    singles = [batch_engine.encrypt_message(message) for message in messages]
    stacked = Ciphertext.stack_batch(singles)

    batched_ntt = batch_engine.coefficient_domain_to_ntt_domain(stacked)
    single_ntt = [
        batch_engine.coefficient_domain_to_ntt_domain(value)
        for value in singles
    ]

    for actual, expected in zip(batched_ntt.unbind_batch(), single_ntt):
        torch.testing.assert_close(actual.data, expected.data, rtol=0, atol=0)

    roundtrip = batch_engine.ntt_domain_to_coefficient_domain(batched_ntt)
    decoded = batch_engine.decrypt_message(roundtrip, is_real=True)
    torch.testing.assert_close(
        decoded[:, : messages.size(-1)].to(messages.device),
        messages,
        atol=1e-5,
        rtol=0,
    )


@pytest.mark.gpu
def test_batched_multiply_relinearize_pipeline(batch_engine):
    messages = _messages(batch_engine.device)
    ciphertext = batch_engine.encrypt_message(messages)

    operand = batch_engine.coefficient_domain_to_ntt_domain(ciphertext)
    triplet = batch_engine.multiply(operand, operand)
    product = batch_engine.rescale_to_next_level(
        batch_engine.relinearize(triplet)
    )
    decoded = batch_engine.decrypt_message(product, is_real=True)

    assert triplet.data.shape[:2] == (3, 2)
    assert product.data.shape[:2] == (2, 2)
    torch.testing.assert_close(
        decoded[:, : messages.size(-1)].to(messages.device),
        messages.square(),
        atol=1e-5,
        rtol=0,
    )


@pytest.mark.gpu
def test_batched_plaintext_operations(batch_engine):
    messages = _messages(batch_engine.device)
    ciphertext = batch_engine.encrypt_message(messages)
    addend = batch_engine.prepare_plaintext_for_addition(
        batch_engine.encode(messages)
    )
    multiplier = batch_engine.prepare_plaintext_for_multiplication(
        batch_engine.encode(torch.ones_like(messages))
    )

    added = batch_engine.add_plaintext(ciphertext, addend)
    multiplied = batch_engine.rescale_to_next_level(
        batch_engine.ntt_domain_to_coefficient_domain(
            batch_engine.multiply_plaintext(
                batch_engine.coefficient_domain_to_ntt_domain(ciphertext),
                multiplier,
            )
        )
    )
    added_message = batch_engine.decrypt_message(added, is_real=True)
    multiplied_message = batch_engine.decrypt_message(multiplied, is_real=True)

    torch.testing.assert_close(
        added_message[:, : messages.size(-1)].to(messages.device),
        2 * messages,
        atol=1e-5,
        rtol=0,
    )
    torch.testing.assert_close(
        multiplied_message[:, : messages.size(-1)].to(messages.device),
        messages,
        atol=1e-5,
        rtol=0,
    )


@pytest.mark.gpu
def test_unbatched_public_plaintext_constants_broadcast_without_copy(
    batch_engine,
):
    base = _messages(batch_engine.device)
    messages = torch.stack((base, -0.5 * base, 1.5 * base), dim=1)
    constant = torch.linspace(
        0.5,
        1.25,
        messages.size(-1),
        dtype=torch.float64,
        device=batch_engine.device,
    )
    ciphertext = batch_engine.encrypt_message(messages)
    addend = batch_engine.prepare_plaintext_for_addition(
        batch_engine.encode(constant)
    )
    multiplier = batch_engine.prepare_plaintext_for_multiplication(
        batch_engine.encode(constant)
    )
    addend_storage = addend.data.untyped_storage().data_ptr()
    multiplier_storage = multiplier.data.untyped_storage().data_ptr()

    added = batch_engine.add_plaintext(ciphertext, addend)
    ciphertext_ntt = batch_engine.coefficient_domain_to_ntt_domain(ciphertext)
    product = batch_engine.multiply_plaintext(ciphertext_ntt, multiplier)
    multiplied = batch_engine.rescale_to_next_level(
        batch_engine.ntt_domain_to_coefficient_domain(product)
    )

    mutable_added = ciphertext.clone()
    assert batch_engine.add_plaintext_(mutable_added, addend) is mutable_added
    mutable_product = ciphertext_ntt.clone()
    assert (
        batch_engine.multiply_plaintext_(mutable_product, multiplier)
        is mutable_product
    )

    assert addend.batch_shape == ()
    assert multiplier.batch_shape == ()
    assert addend.data.untyped_storage().data_ptr() == addend_storage
    assert multiplier.data.untyped_storage().data_ptr() == multiplier_storage
    assert added.batch_shape == messages.shape[:-1]
    assert product.batch_shape == messages.shape[:-1]
    torch.testing.assert_close(added.data, mutable_added.data, rtol=0, atol=0)
    torch.testing.assert_close(
        product.data,
        mutable_product.data,
        rtol=0,
        atol=0,
    )

    added_message = batch_engine.decrypt_message(added, is_real=True)
    multiplied_message = batch_engine.decrypt_message(
        multiplied,
        is_real=True,
    )
    torch.testing.assert_close(
        added_message[..., : messages.size(-1)].to(messages.device),
        messages + constant,
        atol=1e-5,
        rtol=0,
    )
    torch.testing.assert_close(
        multiplied_message[..., : messages.size(-1)].to(messages.device),
        messages * constant,
        atol=1e-5,
        rtol=0,
    )


@pytest.mark.gpu
def test_batched_inplace_ciphertext_arithmetic(batch_engine):
    messages = _messages(batch_engine.device)
    source = batch_engine.encrypt_message(messages)
    other = batch_engine.encrypt_message(-messages)

    mutable = source.clone()
    assert batch_engine.add_(mutable, other) is mutable
    assert batch_engine.subtract_(mutable, other) is mutable
    assert batch_engine.negate_(mutable) is mutable
    decoded = batch_engine.decrypt_message(mutable, is_real=True)
    torch.testing.assert_close(
        decoded[:, : messages.size(-1)].to(messages.device),
        -messages,
        atol=1e-5,
        rtol=0,
    )


@pytest.mark.gpu
def test_batched_qp_encryption_decryption_and_zero_like(batch_engine):
    messages = _messages(batch_engine.device)
    qp_public_key = batch_engine.create_public_key(
        batch_engine.secret_key, modulus_basis="QP"
    )

    ciphertext = batch_engine.encrypt_message(messages, qp_public_key)
    decoded = batch_engine.decrypt_message(ciphertext, is_real=True)
    encrypted_zero = batch_engine.encrypt_zero_like(ciphertext, qp_public_key)
    zero_message = batch_engine.decrypt_message(encrypted_zero, is_real=True)
    qp_zero = batch_engine.prepare_plaintext_for_addition(
        batch_engine.encode(torch.zeros_like(messages)), modulus_basis='QP'
    )
    added = batch_engine.decrypt_message(
        batch_engine.add_plaintext(ciphertext, qp_zero),
        is_real=True,
    )

    assert ciphertext.modulus_basis == "QP"
    assert ciphertext.batch_shape == (2,)
    torch.testing.assert_close(
        decoded[:, : messages.size(-1)].to(messages.device),
        messages,
        atol=1e-5,
        rtol=0,
    )
    torch.testing.assert_close(
        zero_message,
        torch.zeros_like(zero_message),
        atol=1e-5,
        rtol=0,
    )
    torch.testing.assert_close(
        added[:, : messages.size(-1)].to(messages.device),
        messages,
        atol=1e-5,
        rtol=0,
    )


@pytest.mark.gpu
def test_batched_rotation_conjugation_and_explicit_key_switch(batch_engine):
    messages = _messages(batch_engine.device)
    ciphertext = batch_engine.encrypt_message(messages)

    rotation_key = batch_engine.rotation_key(1)
    rotated = batch_engine.rotate_with_key(ciphertext, rotation_key)
    rotated_singles = [
        batch_engine.rotate_with_key(value, rotation_key)
        for value in ciphertext.unbind_batch()
    ]
    torch.testing.assert_close(
        rotated.data,
        Ciphertext.stack_batch(rotated_singles).data,
        rtol=0,
        atol=0,
    )

    conjugation_key = batch_engine.create_conjugation_key(
        batch_engine.secret_key
    )
    conjugated = batch_engine.conjugate(ciphertext, conjugation_key)
    conjugated_message = batch_engine.decrypt_message(conjugated)
    torch.testing.assert_close(
        conjugated_message[:, : messages.size(-1)].to(messages.device),
        messages.to(torch.complex128).conj(),
        atol=1e-5,
        rtol=0,
    )

    destination_key = batch_engine.create_secret_key()
    switch_key = batch_engine.create_key_switch_key(
        batch_engine.secret_key, destination_key
    )
    switched = batch_engine.switch_key(ciphertext, switch_key)
    switched_message = batch_engine.decrypt_message(
        switched, destination_key, is_real=True
    )
    torch.testing.assert_close(
        switched_message[:, : messages.size(-1)].to(messages.device),
        messages,
        atol=1e-5,
        rtol=0,
    )


@pytest.mark.gpu
def test_batch_mismatch_rejection(batch_engine):
    messages = _messages(batch_engine.device)
    ciphertext = batch_engine.encrypt_message(messages)
    with pytest.raises(ValueError, match="tensor shapes differ"):
        batch_engine.add(ciphertext, ciphertext.select_batch(0))

    ntt = batch_engine.coefficient_domain_to_ntt_domain(ciphertext)
    with pytest.raises(ValueError, match="tensor shapes differ"):
        batch_engine.multiply(ntt, ntt.select_batch(0))

    mismatched_messages = torch.stack((messages[0], messages[0], messages[0]))
    plaintext = batch_engine.prepare_plaintext_for_addition(
        batch_engine.encode(mismatched_messages)
    )
    with pytest.raises(ValueError, match="must exactly match"):
        batch_engine.add_plaintext(ciphertext, plaintext)

    singleton_batched_plaintext = batch_engine.prepare_plaintext_for_addition(
        batch_engine.encode(messages[:1])
    )
    assert singleton_batched_plaintext.batch_shape == (1,)
    with pytest.raises(ValueError, match="must exactly match"):
        batch_engine.add_plaintext(ciphertext, singleton_batched_plaintext)
