"""Stable compressed-plaintext layout and evaluator invariants."""

from __future__ import annotations

import gc
from collections.abc import Iterator

import pytest
import torch

import fhelium as fh


def _dense_plaintext(
    data: torch.Tensor,
    *,
    polynomial_domain: fh.PolynomialDomain = "ntt",
) -> fh.Plaintext:
    return fh.Plaintext(
        message=None,
        level=0,
        scale=2.0**40,
        data=data,
        context_id="compressed-test-context",
        representation="rns",
        polynomial_domain=polynomial_domain,
        modulus_basis="Q",
        residue_representation="montgomery",
        prime_ids=tuple(range(data.size(-2))),
    )


@pytest.mark.parametrize("compression_layout", ["cyclic", "contiguous"])
def test_exact_value_round_trips_repetition_layouts(
    compression_layout: fh.CompressedPlaintextLayout,
) -> None:
    compact = torch.tensor([[11, 13], [17, 19]], dtype=torch.int64)
    if compression_layout == "cyclic":
        dense_data = compact.repeat(1, 4)
    else:
        dense_data = compact.repeat_interleave(4, dim=-1)
    dense = _dense_plaintext(dense_data)

    compressed = fh.CompressedPlaintext.from_plaintext(
        dense,
        unique_count=2,
        compression_layout=compression_layout,
    )

    assert compressed.data.shape == (2, 2)
    assert compressed.ring_dimension == 8
    assert compressed.repeat_count == 4
    assert compressed.compression_layout == compression_layout
    assert compressed.compression_format_version == 1
    assert torch.equal(compressed.data, compact)
    assert torch.equal(compressed.to_plaintext().data, dense_data)
    assert compressed.data.untyped_storage().data_ptr() != (
        dense_data.untyped_storage().data_ptr()
    )


def test_compression_rejects_non_repeated_encoded_values() -> None:
    dense = _dense_plaintext(torch.arange(16, dtype=torch.int64).reshape(2, 8))
    for compression_layout in ("cyclic", "contiguous"):
        with pytest.raises(ValueError, match="not exactly representable"):
            fh.CompressedPlaintext.from_plaintext(
                dense,
                unique_count=2,
                compression_layout=compression_layout,
            )


def test_compressed_plaintext_rejects_unknown_physical_layout() -> None:
    with pytest.raises(ValueError, match="Unsupported.*compression_layout"):
        fh.CompressedPlaintext(
            data=torch.tensor([[11, 13], [17, 19]], dtype=torch.int64),
            ring_dimension=8,
            compression_layout="diagonal",  # type: ignore[arg-type]
            level=0,
            scale=2.0**40,
            context_id="compressed-test-context",
            polynomial_domain="ntt",
            modulus_basis="Q",
            residue_representation="montgomery",
            prime_ids=(0, 1),
        )


def test_strided_sparse_layout_preserves_implicit_row_values() -> None:
    compact = torch.tensor([[11, 13], [17, 19]], dtype=torch.int64)
    implicit = torch.tensor([101, 103], dtype=torch.int64)
    dense_data = implicit[:, None].expand(2, 8).clone()
    dense_data[..., ::4] = compact
    dense = _dense_plaintext(dense_data, polynomial_domain="coefficient")

    compressed = fh.CompressedPlaintext.from_plaintext(
        dense,
        unique_count=2,
        compression_layout="strided_sparse",
    )

    assert compressed.compression_layout == "strided_sparse"
    assert compressed.polynomial_domain == "coefficient"
    assert compressed.nbytes == compact.nbytes + implicit.nbytes
    assert torch.equal(compressed.data, compact)
    assert compressed.implicit_data is not None
    assert torch.equal(compressed.implicit_data, implicit)
    assert torch.equal(compressed.to_plaintext().data, dense_data)


def test_compressed_plaintext_rejects_unknown_format_version() -> None:
    with pytest.raises(ValueError, match="compression format version"):
        fh.CompressedPlaintext(
            data=torch.tensor([[11, 13], [17, 19]], dtype=torch.int64),
            ring_dimension=8,
            compression_layout="cyclic",
            compression_format_version=2,
            level=0,
            scale=2.0**40,
            context_id="compressed-test-context",
            polynomial_domain="ntt",
            modulus_basis="Q",
            residue_representation="montgomery",
            prime_ids=(0, 1),
        )


@pytest.fixture(
    scope="module",
    params=(
        pytest.param("cpu", id="cpu"),
        pytest.param("cuda:0", marks=pytest.mark.gpu, id="cuda"),
    ),
)
def engine(request: pytest.FixtureRequest) -> Iterator[fh.CkksEngine]:
    device = str(request.param)
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    instance = fh.CkksEngine(
        fh.Preset.slots8192_scale40_levels7_int64, device=device
    )
    yield instance
    del instance
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


def _with_repeated_encoded_axis(
    plaintext: fh.Plaintext,
    *,
    unique_count: int,
    compression_layout: fh.CompressedPlaintextLayout,
) -> tuple[fh.Plaintext, fh.CompressedPlaintext]:
    assert plaintext.data is not None
    compact = plaintext.data[..., :unique_count].clone()
    if compression_layout == "cyclic":
        dense_data = compact.repeat(
            *([1] * (compact.ndim - 1)),
            plaintext.data.size(-1) // unique_count,
        )
    else:
        dense_data = compact.repeat_interleave(
            plaintext.data.size(-1) // unique_count,
            dim=-1,
        )
    dense = fh.Plaintext(
        message=None,
        level=plaintext.level,
        scale=plaintext.scale,
        data=dense_data,
        context_id=plaintext.context_id,
        representation="rns",
        polynomial_domain=plaintext.polynomial_domain,
        modulus_basis=plaintext.modulus_basis,
        residue_representation=plaintext.residue_representation,
        prime_ids=plaintext.prime_ids,
    )
    compressed = fh.CompressedPlaintext.from_plaintext(
        dense,
        unique_count=unique_count,
        compression_layout=compression_layout,
    )
    return dense, compressed


@pytest.mark.gpu
@pytest.mark.parametrize("compression_layout", ["cyclic", "contiguous"])
def test_multiply_plaintext_physical_paths_match_dense_exactly(
    engine: fh.CkksEngine,
    compression_layout: fh.CompressedPlaintextLayout,
) -> None:
    index = torch.arange(engine.num_slots, dtype=torch.float64)
    message = 0.01 * torch.sin(index * 0.01)
    factor = 0.02 * torch.cos(index * 0.02)
    ciphertext = engine.coefficient_domain_to_ntt_domain(
        engine.encrypt_message(message)
    )
    seed = engine.prepare_plaintext_for_multiplication(engine.encode(factor))
    dense, compressed = _with_repeated_encoded_axis(
        seed,
        unique_count=32,
        compression_layout=compression_layout,
    )

    expected = engine.multiply_plaintext(ciphertext, dense)
    actual = engine.multiply_plaintext(ciphertext, compressed)

    assert torch.equal(actual.data, expected.data)


@pytest.mark.gpu
@pytest.mark.parametrize("compression_layout", ["cyclic", "contiguous"])
def test_add_plaintext_physical_paths_match_dense_exactly(
    engine: fh.CkksEngine,
    compression_layout: fh.CompressedPlaintextLayout,
) -> None:
    index = torch.arange(engine.num_slots, dtype=torch.float64)
    message = 0.01 * torch.sin(index * 0.01)
    addend = 0.02 * torch.cos(index * 0.02)
    ciphertext = engine.encrypt_message(message)
    seed = engine.prepare_plaintext_for_addition(engine.encode(addend))
    dense, compressed = _with_repeated_encoded_axis(
        seed,
        unique_count=32,
        compression_layout=compression_layout,
    )

    expected = engine.add_plaintext(ciphertext, dense)
    actual = engine.add_plaintext(ciphertext, compressed)

    assert torch.equal(actual.data, expected.data)


@pytest.mark.gpu
def test_real_periodic_slots_compress_and_multiply_numerically(
    engine: fh.CkksEngine,
) -> None:
    period = 8
    index = torch.arange(period, dtype=torch.float64)
    unique = 0.03 * torch.cos(index * 0.7) + 0.001 * index
    factor = unique.repeat(engine.num_slots // period)
    dense = engine.prepare_plaintext_for_multiplication(engine.encode(factor))
    compressed = fh.CompressedPlaintext.from_plaintext(
        dense,
        unique_count=2 * period,
        compression_layout="contiguous",
    )

    slot_index = torch.arange(engine.num_slots, dtype=torch.float64)
    message = 0.01 * torch.sin(slot_index * 0.013)
    ciphertext = engine.coefficient_domain_to_ntt_domain(
        engine.encrypt_message(message)
    )
    expected = engine.multiply_plaintext(ciphertext, dense)
    actual = engine.multiply_plaintext(ciphertext, compressed)

    assert torch.equal(actual.data, expected.data)
    decoded = engine.decrypt_message(
        engine.ntt_domain_to_coefficient_domain(actual), is_real=True
    ).cpu()
    max_error = torch.max(torch.abs(decoded - message * factor)).item()
    assert max_error < 4e-5


@pytest.mark.gpu
def test_strided_add_consumes_exact_nonzero_implicit_values(
    engine: fh.CkksEngine,
) -> None:
    """Compact addition must not assume the implicit value is modular zero."""

    period = 8
    unique_count = 2 * period
    addend = (torch.arange(period, dtype=torch.float64) * 0.002 - 0.006).repeat(
        engine.num_slots // period
    )
    encoded = engine.prepare_plaintext_for_addition(engine.encode(addend))
    assert encoded.data is not None
    modified_data = encoded.data.clone()
    support_stride = engine.config.N // unique_count
    support = torch.zeros(
        engine.config.N,
        dtype=torch.bool,
        device=engine.device,
    )
    support[::support_stride] = True
    modified_data[..., ~support] += 1
    dense = fh.Plaintext(
        message=None,
        level=encoded.level,
        scale=encoded.scale,
        data=modified_data,
        context_id=encoded.context_id,
        representation="rns",
        polynomial_domain=encoded.polynomial_domain,
        modulus_basis=encoded.modulus_basis,
        residue_representation=encoded.residue_representation,
        prime_ids=encoded.prime_ids,
    )
    compressed = fh.CompressedPlaintext.from_plaintext(
        dense,
        unique_count=unique_count,
        compression_layout="strided_sparse",
    )

    index = torch.arange(engine.num_slots, dtype=torch.float64)
    ciphertext = engine.encrypt_message(0.01 * torch.sin(index * 0.013))
    moduli = torch.tensor(
        engine.rns_runtime.moduli_for_basis(ciphertext.level),
        dtype=ciphertext.data.dtype,
        device=engine.device,
    )
    ciphertext.c0[..., ~support] += moduli[:, None]

    expected = engine.add_plaintext(ciphertext, dense)
    actual = engine.add_plaintext(ciphertext, compressed)
    assert torch.equal(actual.data, expected.data)


@pytest.mark.gpu
def test_semantic_slot_order_is_not_assumed_to_be_encoded_repetition(
    engine: fh.CkksEngine,
) -> None:
    unique = torch.arange(8, dtype=torch.float64) * 0.003
    contiguous_slots = unique.repeat_interleave(engine.num_slots // 8)
    dense = engine.prepare_plaintext_for_multiplication(
        engine.encode(contiguous_slots)
    )

    for compression_layout in ("cyclic", "contiguous"):
        with pytest.raises(ValueError, match="not exactly representable"):
            fh.CompressedPlaintext.from_plaintext(
                dense,
                unique_count=16,
                compression_layout=compression_layout,
            )
