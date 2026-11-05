"""Stable native FakeTensor behavior, arithmetic, and validation invariants."""

from __future__ import annotations

from fhelium.legacy.engine import CkksEngine

from types import SimpleNamespace

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import fhelium.native  # noqa: F401 - loads native torch operator libraries
from fhelium import Preset
from fhelium.legacy.engine.direct_keyswitch_consumer import (
    CompactTailFusedDirectKeySwitchDigitConsumer,
    StandardDirectKeySwitchDigitConsumer,
)
from fhelium.legacy.engine.ntt.backends.compact_radix2 import (
    CompactRadix2NttBackend,
)
from fhelium.native.wrapper import ckks_ops, rns_ops


@pytest.mark.parametrize(
    ("dtype", "modulus_values"),
    [
        (torch.int32, (65537, 114689, 147457)),
        (
            torch.int64,
            (
                1152921504606748673,
                576460752303423649,
                288230376151711813,
            ),
        ),
    ],
)
def test_cpu_standard_rns_dispatch_preserves_cuda_schema_semantics(
    dtype: torch.dtype,
    modulus_values: tuple[int, ...],
) -> None:
    moduli = torch.tensor(modulus_values, dtype=dtype)
    parameters = torch.zeros((8, len(modulus_values)), dtype=dtype)
    parameters[0] = 2 * moduli
    generator = torch.Generator().manual_seed(20260811)
    upper_bound = min(modulus_values)
    lhs = torch.randint(
        0,
        upper_bound,
        (2, 3, 257),
        dtype=dtype,
        generator=generator,
    )
    singleton_rhs = torch.randint(
        0,
        upper_bound,
        (1, 3, 257),
        dtype=dtype,
        generator=generator,
    )
    row_moduli = moduli.view(1, -1, 1)

    for operation, inplace_operation, cleartext in (
        (
            rns_ops.add_standard,
            rns_ops.add_standard_,
            torch.remainder(lhs + singleton_rhs, row_moduli),
        ),
        (
            rns_ops.sub_standard,
            rns_ops.sub_standard_,
            torch.remainder(lhs - singleton_rhs, row_moduli),
        ),
    ):
        result = operation(lhs, singleton_rhs, parameters)
        assert result.data_ptr() != lhs.data_ptr()
        assert result.data_ptr() != singleton_rhs.data_ptr()
        torch.testing.assert_close(result, cleartext, rtol=0, atol=0)

        mutated = lhs.clone()
        original_storage = mutated.data_ptr()
        inplace_operation(mutated, singleton_rhs, parameters)
        assert mutated.data_ptr() == original_storage
        torch.testing.assert_close(mutated, cleartext, rtol=0, atol=0)


def test_cpu_standard_rns_dispatch_rejects_invalid_operands() -> None:
    lhs = torch.zeros((2, 3, 32), dtype=torch.int64)
    rhs = torch.zeros_like(lhs)
    parameters = torch.zeros((8, 3), dtype=torch.int64)
    parameters[0] = 2 * torch.tensor((65537, 114689, 147457))

    with pytest.raises(RuntimeError, match="operand coefficient counts differ"):
        rns_ops.add_standard(lhs, rhs[..., :16], parameters)
    with pytest.raises(RuntimeError, match="operand dtypes differ"):
        rns_ops.add_standard(lhs, rhs.to(torch.int32), parameters)

    overlapping_lhs = torch.zeros((1, 3, 32), dtype=torch.int64).expand(
        2, -1, -1
    )
    with pytest.raises(RuntimeError, match="more than one element"):
        rns_ops.add_standard_(overlapping_lhs, rhs, parameters)

    with pytest.raises(RuntimeError, match="refer to a single memory location"):
        rns_ops.sub_standard_(lhs, lhs, parameters)


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]
)
@pytest.mark.parametrize(
    ("dtype", "modulus_values"),
    [
        (torch.int32, (268369921, 134176769)),
        (torch.int32, (469762049, 998244353)),
        (torch.int64, (1152921504606748673, 576460752303423649)),
    ],
)
def test_standard_rns_dispatch_preserves_lazy_boundaries(
    dtype: torch.dtype,
    modulus_values: tuple[int, ...],
    device: str,
) -> None:
    moduli = torch.tensor(modulus_values, dtype=dtype, device=device)
    parameters = torch.zeros(
        (8, len(modulus_values)), dtype=dtype, device=device
    )
    parameters[0] = 2 * moduli
    lhs = torch.tensor(
        [(0, q - 1, q, 2 * q - 1, 2 * q - 1) for q in modulus_values],
        dtype=dtype,
        device=device,
    )[None]
    rhs = torch.tensor(
        [(2 * q - 1, q, q - 1, 1, 2 * q - 1) for q in modulus_values],
        dtype=dtype,
        device=device,
    )[None]
    row_moduli = moduli.to(torch.int64).view(1, -1, 1)
    # Widen the reference before adding legal 30-bit lazy representatives.
    sum_ = lhs.to(torch.int64) + rhs.to(torch.int64)
    difference = lhs.to(torch.int64) - rhs.to(torch.int64)
    for operation, value, modulus in (
        (rns_ops.add_standard, sum_, row_moduli),
        (rns_ops.sub_standard, difference, row_moduli),
        (rns_ops.add_lazy, sum_, 2 * row_moduli),
        (rns_ops.sub_lazy, difference, 2 * row_moduli),
    ):
        torch.testing.assert_close(
            operation(lhs, rhs, parameters),
            torch.remainder(value, modulus).to(dtype),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]
)
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_mixed_radix_digits_preserve_large_prime_residues(
    dtype: torch.dtype,
    device: str,
) -> None:
    import math
    import random

    primes = (1073692673, 1073668097, 1073651713, 1073643521)
    modulus = math.prod(primes)
    generator = random.Random(942)
    integers = [0, 1, modulus - 1, modulus // 2]
    integers += [generator.randrange(modulus) for _ in range(252)]
    radix_bits = 32 if dtype == torch.int32 else 62
    radix = 1 << radix_bits
    half_bits = radix_bits // 2
    mask = (1 << half_bits) - 1
    products = [math.prod(primes[: i + 1]) for i in range(len(primes) - 1)]
    normalizers = torch.tensor(
        [
            pow(product, -1, primes[i + 1]) * radix % primes[i + 1]
            for i, product in enumerate(products)
        ],
        dtype=dtype,
        device=device,
    )
    propagation = torch.tensor(
        [
            [
                product * radix % q if target > i + 1 else 0
                for target, q in enumerate(primes)
            ]
            for i, product in enumerate(products)
        ],
        dtype=dtype,
        device=device,
    )
    inverses = [-pow(q, -1, radix) % radix for q in primes]
    reduction = [
        torch.tensor(row, dtype=dtype, device=device)
        for row in (
            [q & mask for q in primes],
            [q >> half_bits for q in primes],
            [inv & mask for inv in inverses],
            [inv >> half_bits for inv in inverses],
        )
    ]
    source = torch.tensor(
        [
            [
                value % q + (q if i % 2 else 0)
                for i, value in enumerate(integers)
            ]
            for q in primes
        ],
        dtype=dtype,
        device=device,
    )[None]
    expected = torch.tensor(
        [
            [value // math.prod(primes[:i]) % q for value in integers]
            for i, q in enumerate(primes)
        ],
        dtype=dtype,
        device=device,
    )[None]
    actual = rns_ops.mixed_radix_decompose(
        source,
        normalizers,
        propagation,
        *reduction,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
def test_shape_changing_native_ops_have_explicit_fake_shapes(
    batch_shape: tuple[int, ...],
) -> None:
    mode = FakeTensorMode()
    with mode:
        mixed_radix_components = torch.empty(*batch_shape, 3, 32, device="cuda")
        basis_extension_coefficients = torch.empty(2, 7, device="cuda")
        rns_parameters = torch.empty(8, 7, device="cuda")
        extended = rns_ops.mixed_radix_basis_extend_to_montgomery(
            mixed_radix_components,
            basis_extension_coefficients,
            rns_parameters,
            7,
        )
        assert extended.shape == (*batch_shape, 7, 32)
        assert extended.device.type == "cuda"

        centered_coefficients = torch.empty(*batch_shape, 32, device="cuda")
        twice_modulus = torch.empty(7, device="cuda")
        lifted = rns_ops.lift_centered_coefficients(
            centered_coefficients,
            twice_modulus,
        )
        assert lifted.shape == (*batch_shape, 7, 32)
        assert lifted.device.type == "cuda"

        lhs = torch.empty(*batch_shape, 7, 32, device="cuda")
        compressed_rhs = torch.empty(*batch_shape, 7, 4, device="cuda")
        implicit_rhs = torch.empty(*batch_shape, 7, device="cuda")
        outputs = (
            rns_ops.montgomery_mul_cyclic_compressed(
                lhs,
                compressed_rhs,
                rns_parameters,
            ),
            rns_ops.montgomery_mul_contiguous_compressed(
                lhs,
                compressed_rhs,
                rns_parameters,
            ),
            ckks_ops.add_cyclic_compressed_plaintext_component(
                lhs,
                compressed_rhs,
                rns_parameters,
            ),
            ckks_ops.add_contiguous_compressed_plaintext_component(
                lhs,
                compressed_rhs,
                rns_parameters,
            ),
            ckks_ops.add_strided_plaintext_component(
                lhs,
                compressed_rhs,
                implicit_rhs,
                rns_parameters,
            ),
        )
        assert all(output.shape == lhs.shape for output in outputs)


@pytest.mark.gpu
def test_keyswitch_moddown_does_not_mutate_public_p_residues() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    engine = CkksEngine(Preset.slots8192_scale40_depth7_int64, device="cuda:0")
    depth = 0
    q_row_count = len(engine.rns_layout.prime_ids(depth))
    p_row_count = engine.config.num_p_primes
    q_residues = torch.zeros(
        (q_row_count, engine.config.N),
        dtype=engine.rns_runtime.dtype,
        device=engine.device,
    )
    p_residues = torch.arange(
        p_row_count,
        dtype=engine.rns_runtime.dtype,
        device=engine.device,
    )[:, None].repeat(1, engine.config.N)
    p_before = p_residues.clone()
    active_parameters = engine.rns_runtime.rns_parameters_for(
        torch.cat((q_residues, p_residues)),
        include_p=True,
    )

    result = ckks_ops.keyswitch_moddown_qp_to_q(
        q_residues,
        p_residues,
        engine.moddown_p_drop_inverses_montgomery_by_depth[depth],
        active_parameters,
        rounding="floor",
    )

    assert result.shape == q_residues.shape
    assert torch.equal(p_residues, p_before)

    with pytest.raises(RuntimeError, match="p_residues to be a CUDA tensor"):
        ckks_ops.keyswitch_moddown_qp_to_q(
            q_residues,
            p_residues.cpu(),
            engine.moddown_p_drop_inverses_montgomery_by_depth[depth],
            active_parameters,
            rounding="floor",
        )
    with pytest.raises(RuntimeError, match="same integral dtype"):
        ckks_ops.keyswitch_moddown_qp_to_q(
            q_residues,
            p_residues,
            engine.moddown_p_drop_inverses_montgomery_by_depth[depth].to(
                torch.int32
            ),
            active_parameters,
            rounding="floor",
        )

    qp_residues = torch.cat((q_residues, p_residues))
    invalid_key_digit = torch.zeros(
        (2, *qp_residues.shape),
        dtype=torch.int32,
        device=engine.device,
    )
    with pytest.raises(RuntimeError, match="same integral dtype"):
        ckks_ops.keyswitch_accumulate_digit_products_(
            qp_residues.clone(),
            qp_residues.clone(),
            qp_residues,
            invalid_key_digit,
            active_parameters,
            0,
        )


def test_ordinary_keyswitch_consumer_remains_decomposed_for_group16() -> None:
    compact_group16 = object.__new__(CompactRadix2NttBackend)
    compact_group16.grouped_radix2_stage_count = 4
    runtime = SimpleNamespace(ntt_backend=compact_group16)

    consumer = StandardDirectKeySwitchDigitConsumer(runtime)  # type: ignore[arg-type]

    assert type(consumer) is StandardDirectKeySwitchDigitConsumer
    assert consumer.rns_runtime is runtime
    fused = CompactTailFusedDirectKeySwitchDigitConsumer(compact_group16)
    assert fused.backend is compact_group16


@pytest.mark.gpu
def test_cpu_and_cuda_complete_native_dispatch_sets_match() -> None:
    common_ops = {
        "fhelium_rns_ops": (
            "add_standard",
            "add_standard_",
            "sub_standard",
            "sub_standard_",
            "montgomery_mul_row_scalars_standard",
            "montgomery_mul",
            "montgomery_mul_cyclic_compressed",
            "montgomery_mul_contiguous_compressed",
            "montgomery_mul_row_scalars_",
            "to_montgomery_",
            "from_montgomery_",
            "add_lazy",
            "add_lazy_with_twice_modulus",
            "sub_lazy",
            "reduce_to_standard_",
            "center_residues_",
            "shift_residues_positive_",
            "lift_centered_coefficients",
            "mixed_radix_decompose",
            "mixed_radix_basis_extend_to_montgomery",
        ),
        "fhelium_ckks_ops": (
            "multiply_two_component_ntt_montgomery",
            "add_prepared_plaintext_component",
            "add_prepared_plaintext_component_",
            "add_cyclic_compressed_plaintext_component",
            "add_cyclic_compressed_plaintext_component_",
            "add_contiguous_compressed_plaintext_component",
            "add_contiguous_compressed_plaintext_component_",
            "add_strided_plaintext_component",
            "add_strided_plaintext_component_",
            "rescale_drop_leading_prime_nearest",
            "rescale_drop_leading_prime_nearest_",
            "rescale_drop_leading_prime_truncate",
            "rescale_drop_leading_prime_truncate_",
            "apply_coefficient_galois_automorphism",
            "apply_ntt_galois_automorphism",
            "keyswitch_moddown_qp_to_q",
            "keyswitch_accumulate_digit_products_",
        ),
        "fhelium_ntt_ops": (
            "forward_ntt_montgomery_indexed_",
            "forward_ntt_to_montgomery_indexed_",
            "forward_ntt_to_montgomery_indexed",
            "inverse_ntt_montgomery_indexed_",
            "inverse_ntt_to_standard_lazy_indexed_",
            "inverse_ntt_to_standard_indexed_",
            "inverse_ntt_to_centered_indexed_",
        ),
    }
    for namespace, operators in common_ops.items():
        for operator in operators:
            qualified = f"{namespace}::{operator}"
            assert torch._C._dispatch_has_kernel_for_dispatch_key(
                qualified, "CPU"
            )
            assert torch._C._dispatch_has_kernel_for_dispatch_key(
                qualified, "CUDA"
            )


@pytest.mark.gpu
def test_native_binary_op_rejects_incompatible_batch_counts() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    engine = CkksEngine(
        Preset.slots8192_scale40_depth7_int64,
        device="cuda:0",
        allow_sk_gen=False,
    )
    limb_count = len(engine.rns_layout.prime_ids(0))
    lhs = torch.zeros(
        (3, limb_count, engine.config.N),
        dtype=engine.rns_runtime.dtype,
        device=engine.device,
    )
    rhs = torch.zeros(
        (2, limb_count, engine.config.N),
        dtype=engine.rns_runtime.dtype,
        device=engine.device,
    )
    parameters = engine.rns_runtime.rns_parameters_for(lhs)

    with pytest.raises(RuntimeError, match="batch counts differ"):
        rns_ops.add_standard(lhs, rhs, parameters)


@pytest.mark.gpu
def test_standard_subtraction_returns_residues_below_modulus() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    engine = CkksEngine(
        Preset.slots8192_scale40_depth7_int64,
        device="cuda:0",
        allow_sk_gen=False,
    )
    moduli = torch.tensor(
        engine.rns_runtime.moduli_for_basis(0),
        dtype=engine.rns_runtime.dtype,
        device=engine.device,
    )[:, None]
    lhs = torch.cat((torch.ones_like(moduli), 2 * moduli - 1), dim=-1)
    rhs = torch.cat(
        (2 * torch.ones_like(moduli), torch.ones_like(moduli)), dim=-1
    )
    parameters = engine.rns_runtime.rns_parameters_for(lhs)

    reduced = rns_ops.sub_standard(lhs, rhs, parameters)
    expected = torch.cat((moduli - 1, moduli - 2), dim=-1)

    assert torch.all(reduced >= 0)
    assert torch.all(reduced < moduli)
    torch.testing.assert_close(reduced, expected, rtol=0, atol=0)


@pytest.mark.gpu
def test_native_compressed_arithmetic_rejects_invalid_storage_and_dtype() -> (
    None
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    engine = CkksEngine(
        Preset.slots8192_scale40_depth7_int64,
        device="cuda:0",
        allow_sk_gen=False,
    )
    limb_count = len(engine.rns_layout.prime_ids(0))
    valid_ring = torch.zeros(
        (limb_count, engine.config.N),
        dtype=engine.rns_runtime.dtype,
        device=engine.device,
    )
    parameters = engine.rns_runtime.rns_parameters_for(valid_ring)
    invalid_ring = torch.zeros(
        (limb_count, 12),
        dtype=engine.rns_runtime.dtype,
        device=engine.device,
    )
    compressed = torch.zeros(
        (limb_count, 4),
        dtype=engine.rns_runtime.dtype,
        device=engine.device,
    )

    with pytest.raises(RuntimeError, match="ring dimension must be"):
        rns_ops.montgomery_mul_cyclic_compressed(
            invalid_ring,
            compressed,
            parameters,
        )

    with pytest.raises(RuntimeError, match="operand dtypes differ"):
        rns_ops.montgomery_mul_contiguous_compressed(
            valid_ring,
            compressed.to(torch.int32),
            parameters,
        )


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]
)
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_mixed_radix_output_buffers_and_table_extents(
    device: str,
    dtype: torch.dtype,
) -> None:
    """Garner/extension preserve strided output storage and reject unsafe tables."""
    import math

    from fhelium.backend.rns.context import RnsContext
    from fhelium.backend.rns.tables import centered_basis_extension_tables
    from fhelium.config import CkksConfig

    config = CkksConfig.parse(Preset.slots32768_scale25_depth24_int32)
    context = RnsContext(config, dtype=dtype, device=device)
    target_ids = (0, 17, 18)
    for width in (1, 4, 17):
        ids = tuple(range(width))
        tables = centered_basis_extension_tables(
            context, context, ids, target_ids
        )
        moduli = tuple(config.moduli[i] for i in ids)
        product = math.prod(moduli)
        integers = [0, 1, product // 2, product - 1, *range(28)]
        source = (
            torch.tensor(
                [[value % q for value in integers] for q in moduli],
                dtype=dtype,
                device=device,
            )
            .unsqueeze(0)
            .repeat(2, 1, 1)
        )
        original = source.clone()
        arguments = (
            source,
            tables['normalizers'],
            tables['propagation'],
            tables['source_lo'],
            tables['source_hi'],
            tables['source_neg_lo'],
            tables['source_neg_hi'],
        )
        digits = torch.empty(
            (2, width, 64),
            dtype=dtype,
            device=device,
        )[..., ::2]
        storage = digits.untyped_storage().data_ptr()
        rns_ops.mixed_radix_decompose_out_(*arguments, digits)
        expected = (
            torch.tensor(
                [
                    [value // math.prod(moduli[:i]) % q for value in integers]
                    for i, q in enumerate(moduli)
                ],
                dtype=dtype,
                device=device,
            )
            .unsqueeze(0)
            .repeat(2, 1, 1)
        )
        torch.testing.assert_close(digits, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            digits,
            rns_ops.mixed_radix_decompose(*arguments),
            rtol=0,
            atol=0,
        )
        assert digits.untyped_storage().data_ptr() == storage
        assert digits.stride(-1) == 2

        extended = torch.empty(
            (2, len(target_ids), 64),
            dtype=dtype,
            device=device,
        )[..., ::2]
        parameters = tables['target_parameters']
        extension_arguments = (
            digits,
            tables['extension'],
            parameters,
            len(target_ids),
        )
        storage = extended.untyped_storage().data_ptr()
        rns_ops.mixed_radix_basis_extend_to_montgomery_out_(
            *extension_arguments,
            extended,
        )
        torch.testing.assert_close(
            extended,
            rns_ops.mixed_radix_basis_extend_to_montgomery(
                *extension_arguments
            ),
            rtol=0,
            atol=0,
        )
        assert extended.untyped_storage().data_ptr() == storage

        from fhelium.backend.rns.modup import modup_digit

        bindings = {
            "qp_parameters": parameters,
            "digit0_normalizers": tables["normalizers"],
            "digit0_propagation": tables["propagation"],
            "digit0_extension": tables["extension"],
            **{
                f"digit0_reduction{i}": value
                for i, value in enumerate(arguments[3:])
            },
        }
        span = (0, width, 0)
        reference = extended.clone()
        torch.testing.assert_close(
            modup_digit(source, bindings, 0, span), reference, rtol=0, atol=0
        )
        output = torch.empty_like(extended)
        assert modup_digit(source, bindings, 0, span, out=output) is output
        torch.testing.assert_close(output, reference, rtol=0, atol=0)
        if width > 1:
            modup_digit(source, bindings, 0, span, out=extended, scratch=digits)
            torch.testing.assert_close(digits, expected, rtol=0, atol=0)
            torch.testing.assert_close(extended, reference, rtol=0, atol=0)
        if source.is_cuda and not torch.version.hip:
            decomposition = list(arguments[1:]) if width > 1 else []
            rns_ops.mixed_radix_modup_tiled_out_(
                source, decomposition, tables["extension"], parameters, extended
            )
            torch.testing.assert_close(extended, reference, rtol=0, atol=0)
            assert extended.untyped_storage().data_ptr() == storage
            if width == 4:
                with pytest.raises(RuntimeError, match="disjoint"):
                    rns_ops.mixed_radix_modup_tiled_out_(
                        source,
                        decomposition,
                        tables["extension"],
                        parameters,
                        source[:, : len(target_ids), :],
                    )
        rns_ops.from_montgomery_(extended, parameters)
        rns_ops.reduce_to_standard_(extended, parameters)
        oracle = (
            torch.tensor(
                [
                    [value % config.moduli[i] for value in integers]
                    for i in target_ids
                ],
                dtype=dtype,
                device=device,
            )
            .unsqueeze(0)
            .repeat(2, 1, 1)
        )
        torch.testing.assert_close(extended, oracle, rtol=0, atol=0)
        torch.testing.assert_close(source, original, rtol=0, atol=0)

        if width == 4:
            for index in range(3, 7):
                invalid = list(arguments)
                invalid[index] = invalid[index][:0]
                with pytest.raises(RuntimeError):
                    rns_ops.mixed_radix_decompose(*invalid)
                with pytest.raises(RuntimeError):
                    rns_ops.mixed_radix_decompose_out_(*invalid, digits)
            with pytest.raises(RuntimeError):
                rns_ops.mixed_radix_decompose_out_(*arguments, source)
            overlapping = torch.empty(
                (1, width, 32),
                dtype=dtype,
                device=device,
            ).expand(2, width, 32)
            with pytest.raises(RuntimeError):
                rns_ops.mixed_radix_decompose_out_(*arguments, overlapping)
            for operation, extra in (
                (rns_ops.mixed_radix_basis_extend_to_montgomery, ()),
                (
                    rns_ops.mixed_radix_basis_extend_to_montgomery_out_,
                    (extended,),
                ),
            ):
                with pytest.raises(RuntimeError):
                    operation(
                        digits,
                        tables['extension'],
                        parameters[:7],
                        len(target_ids),
                        *extra,
                    )
