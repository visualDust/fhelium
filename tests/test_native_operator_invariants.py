"""Stable native FakeTensor behavior, arithmetic, and validation invariants."""

from __future__ import annotations

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import fhelium.native  # noqa: F401 - loads native torch operator libraries
from fhelium import CkksEngine, Preset
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
def test_cpu_canonical_rns_dispatch_preserves_cuda_schema_semantics(
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
            rns_ops.add_canonical,
            rns_ops.add_canonical_,
            torch.remainder(lhs + singleton_rhs, row_moduli),
        ),
        (
            rns_ops.sub_canonical,
            rns_ops.sub_canonical_,
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


def test_cpu_canonical_rns_dispatch_rejects_invalid_operands() -> None:
    lhs = torch.zeros((2, 3, 32), dtype=torch.int64)
    rhs = torch.zeros_like(lhs)
    parameters = torch.zeros((8, 3), dtype=torch.int64)
    parameters[0] = 2 * torch.tensor((65537, 114689, 147457))

    with pytest.raises(RuntimeError, match="operand coefficient counts differ"):
        rns_ops.add_canonical(lhs, rhs[..., :16], parameters)
    with pytest.raises(RuntimeError, match="operand dtypes differ"):
        rns_ops.add_canonical(lhs, rhs.to(torch.int32), parameters)

    overlapping_lhs = torch.zeros((1, 3, 32), dtype=torch.int64).expand(
        2, -1, -1
    )
    with pytest.raises(RuntimeError, match="more than one element"):
        rns_ops.add_canonical_(overlapping_lhs, rhs, parameters)

    with pytest.raises(RuntimeError, match="refer to a single memory location"):
        rns_ops.sub_canonical_(lhs, lhs, parameters)


@pytest.mark.parametrize(
    ("dtype", "modulus_values"),
    [
        (torch.int32, (268369921, 134176769)),
        (torch.int64, (1152921504606748673, 576460752303423649)),
    ],
)
def test_cpu_canonical_rns_dispatch_preserves_exact_lazy_boundaries(
    dtype: torch.dtype,
    modulus_values: tuple[int, ...],
) -> None:
    moduli = torch.tensor(modulus_values, dtype=dtype)
    parameters = torch.zeros((8, len(modulus_values)), dtype=dtype)
    parameters[0] = 2 * moduli
    lhs = torch.stack(
        [
            torch.tensor((0, modulus - 1, modulus, 2 * modulus - 1))
            for modulus in modulus_values
        ]
    ).to(dtype=dtype)[None]
    rhs = torch.stack(
        [
            torch.tensor((2 * modulus - 1, modulus, modulus - 1, 1))
            for modulus in modulus_values
        ]
    ).to(dtype=dtype)[None]
    row_moduli = moduli.view(1, -1, 1)

    torch.testing.assert_close(
        rns_ops.add_canonical(lhs, rhs, parameters),
        torch.remainder(lhs + rhs, row_moduli),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        rns_ops.sub_canonical(lhs, rhs, parameters),
        torch.remainder(lhs - rhs, row_moduli),
        rtol=0,
        atol=0,
    )


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

    engine = CkksEngine(Preset.slots8192_scale40_levels7_int64, device="cuda:0")
    level = 0
    q_row_count = len(engine.rns_layout.prime_ids(level))
    p_row_count = engine.config.num_p_primes
    q_residues = torch.zeros(
        (q_row_count, engine.config.N),
        dtype=engine.config.torch_dtype,
        device=engine.device,
    )
    p_residues = torch.arange(
        p_row_count,
        dtype=engine.config.torch_dtype,
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
        engine.moddown_p_drop_inverses_montgomery_by_level[level],
        active_parameters,
    )

    assert result.shape == q_residues.shape
    assert torch.equal(p_residues, p_before)

    with pytest.raises(RuntimeError, match="p_residues to be a CUDA tensor"):
        ckks_ops.keyswitch_moddown_qp_to_q(
            q_residues,
            p_residues.cpu(),
            engine.moddown_p_drop_inverses_montgomery_by_level[level],
            active_parameters,
        )
    with pytest.raises(RuntimeError, match="same integral dtype"):
        ckks_ops.keyswitch_moddown_qp_to_q(
            q_residues,
            p_residues,
            engine.moddown_p_drop_inverses_montgomery_by_level[level].to(
                torch.int32
            ),
            active_parameters,
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


@pytest.mark.gpu
def test_cpu_and_cuda_complete_native_dispatch_sets_match() -> None:
    common_ops = {
        "fhelium_rns_ops": (
            "add_canonical",
            "add_canonical_",
            "sub_canonical",
            "sub_canonical_",
            "montgomery_mul_row_scalars_canonical",
            "montgomery_mul",
            "montgomery_mul_cyclic_compressed",
            "montgomery_mul_contiguous_compressed",
            "montgomery_mul_row_scalars_",
            "to_montgomery_",
            "from_montgomery_",
            "add_lazy",
            "add_lazy_with_twice_modulus",
            "sub_lazy",
            "canonicalize_residues_",
            "center_residues_",
            "shift_residues_positive_",
            "lift_centered_coefficients",
            "mixed_radix_decompose",
            "mixed_radix_basis_extend_to_montgomery",
        ),
        "fhelium_ckks_ops": (
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
        Preset.slots8192_scale40_levels7_int64,
        device="cuda:0",
        allow_sk_gen=False,
    )
    limb_count = len(engine.rns_layout.prime_ids(0))
    lhs = torch.zeros(
        (3, limb_count, engine.config.N),
        dtype=engine.config.torch_dtype,
        device=engine.device,
    )
    rhs = torch.zeros(
        (2, limb_count, engine.config.N),
        dtype=engine.config.torch_dtype,
        device=engine.device,
    )
    parameters = engine.rns_runtime.rns_parameters_for(lhs)

    with pytest.raises(RuntimeError, match="batch counts differ"):
        rns_ops.add_canonical(lhs, rhs, parameters)


@pytest.mark.gpu
def test_canonical_subtraction_has_canonical_residue_range() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    engine = CkksEngine(
        Preset.slots8192_scale40_levels7_int64,
        device="cuda:0",
        allow_sk_gen=False,
    )
    moduli = torch.tensor(
        engine.rns_runtime.moduli_for_basis(0),
        dtype=engine.config.torch_dtype,
        device=engine.device,
    )[:, None]
    lhs = torch.cat((torch.ones_like(moduli), 2 * moduli - 1), dim=-1)
    rhs = torch.cat(
        (2 * torch.ones_like(moduli), torch.ones_like(moduli)), dim=-1
    )
    parameters = engine.rns_runtime.rns_parameters_for(lhs)

    canonical = rns_ops.sub_canonical(lhs, rhs, parameters)
    expected = torch.cat((moduli - 1, moduli - 2), dim=-1)

    assert torch.all(canonical >= 0)
    assert torch.all(canonical < moduli)
    torch.testing.assert_close(canonical, expected, rtol=0, atol=0)


@pytest.mark.gpu
def test_native_compressed_arithmetic_rejects_invalid_storage_and_dtype() -> (
    None
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    engine = CkksEngine(
        Preset.slots8192_scale40_levels7_int64,
        device="cuda:0",
        allow_sk_gen=False,
    )
    limb_count = len(engine.rns_layout.prime_ids(0))
    valid_ring = torch.zeros(
        (limb_count, engine.config.N),
        dtype=engine.config.torch_dtype,
        device=engine.device,
    )
    parameters = engine.rns_runtime.rns_parameters_for(valid_ring)
    invalid_ring = torch.zeros(
        (limb_count, 12),
        dtype=engine.config.torch_dtype,
        device=engine.device,
    )
    compressed = torch.zeros(
        (limb_count, 4),
        dtype=engine.config.torch_dtype,
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
