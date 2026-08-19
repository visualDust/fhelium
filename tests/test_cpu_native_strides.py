"""CPU native kernels preserve logical strides of auxiliary tensors."""

from __future__ import annotations

import pytest
import torch

from fhelium import CkksConfig, CkksEngine, Preset
from fhelium.native.wrapper import ckks_ops, ntt_ops, rns_ops


def _gap_last_axis(tensor: torch.Tensor) -> torch.Tensor:
    """Return an equal view whose final logical axis has stride two."""

    shape = (*tensor.shape[:-1], tensor.size(-1) * 2)
    storage = torch.empty(shape, dtype=tensor.dtype, device=tensor.device)
    view = storage[..., ::2]
    view.copy_(tensor)
    assert torch.equal(view, tensor)
    assert view.stride(-1) == 2
    return view


def _transpose_layout(tensor: torch.Tensor) -> torch.Tensor:
    """Return an equal rank-two non-overlapping-dense transposed layout."""

    assert tensor.ndim == 2
    view = tensor.mT.contiguous().mT
    assert torch.equal(view, tensor)
    assert not view.is_contiguous()
    return view


def _cpu_engine(*, p_count: int = 2) -> CkksEngine:
    config = CkksConfig(
        logN=12,
        num_scale_primes=1,
        num_p_primes=p_count,
        enforce_security_budget=False,
    )
    return CkksEngine(config, device="cpu", allow_sk_gen=False)


def test_cpu_rns_and_ckks_vector_and_parameter_strides() -> None:
    engine = _cpu_engine()
    parameters = engine.rns_runtime.basis_parameters(
        0, include_p=True
    ).native_parameters[:8, :3]
    strided_parameters = _transpose_layout(parameters)
    moduli = engine.config.moduli[:3]
    generator = torch.Generator().manual_seed(20260814)
    residues = torch.stack(
        [
            torch.randint(
                0,
                modulus,
                (2, 64),
                dtype=engine.config.torch_dtype,
                generator=generator,
            )
            for modulus in moduli
        ],
        dim=1,
    )
    row_scalars = torch.tensor([17, 19, 23], dtype=engine.config.torch_dtype)

    expected = rns_ops.montgomery_mul_row_scalars_canonical(
        residues, row_scalars, parameters
    )
    actual = rns_ops.montgomery_mul_row_scalars_canonical(
        residues,
        _gap_last_axis(row_scalars),
        strided_parameters,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    rhs = torch.flip(residues, dims=(-1,))
    expected = rns_ops.add_lazy_with_twice_modulus(residues, rhs, parameters[0])
    actual = rns_ops.add_lazy_with_twice_modulus(
        residues, rhs, _gap_last_axis(parameters[0])
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    source_indices = torch.arange(63, -1, -1, dtype=torch.int32)
    source_sign = torch.where(
        torch.arange(64) % 3 == 0,
        torch.tensor(-1, dtype=torch.int8),
        torch.tensor(1, dtype=torch.int8),
    )
    expected = ckks_ops.apply_coefficient_galois_automorphism(
        residues,
        source_indices,
        source_sign,
        parameters[0],
    )
    actual = ckks_ops.apply_coefficient_galois_automorphism(
        residues,
        _gap_last_axis(source_indices),
        _gap_last_axis(source_sign),
        _gap_last_axis(parameters[0]),
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    expected = ckks_ops.apply_ntt_galois_automorphism(residues, source_indices)
    actual = ckks_ops.apply_ntt_galois_automorphism(
        residues, _gap_last_axis(source_indices)
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    inverse = torch.tensor([29, 31, 37], dtype=engine.config.torch_dtype)
    dropped = torch.remainder(residues[:, 0], moduli[0])
    expected = ckks_ops.rescale_drop_leading_prime_truncate(
        residues, inverse, dropped, parameters
    )
    actual = ckks_ops.rescale_drop_leading_prime_truncate(
        residues,
        _gap_last_axis(inverse),
        _gap_last_axis(dropped),
        strided_parameters,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_cpu_mixed_radix_auxiliary_strides() -> None:
    engine = _cpu_engine()
    parameters = engine.rns_runtime.basis_parameters(
        0, include_p=True
    ).native_parameters[:8, :3]
    generator = torch.Generator().manual_seed(20260815)
    source = torch.stack(
        [
            torch.randint(
                0,
                modulus,
                (2, 32),
                dtype=engine.config.torch_dtype,
                generator=generator,
            )
            for modulus in engine.config.moduli[:3]
        ],
        dim=1,
    )
    normalizers = torch.tensor([41, 43], dtype=engine.config.torch_dtype)
    propagation = torch.tensor(
        [[0, 47, 53], [0, 0, 59]], dtype=engine.config.torch_dtype
    )
    reduction_vectors = tuple(parameters[row] for row in (1, 2, 3, 4))

    expected = rns_ops.mixed_radix_decompose(
        source,
        normalizers,
        propagation,
        *reduction_vectors,
    )
    actual = rns_ops.mixed_radix_decompose(
        source,
        _gap_last_axis(normalizers),
        _transpose_layout(propagation),
        *(_gap_last_axis(vector) for vector in reduction_vectors),
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    extension_coefficients = torch.tensor(
        [[61, 67, 71], [73, 79, 83]], dtype=engine.config.torch_dtype
    )
    expected_extended = rns_ops.mixed_radix_basis_extend_to_montgomery(
        expected,
        extension_coefficients,
        parameters,
        3,
    )
    actual_extended = rns_ops.mixed_radix_basis_extend_to_montgomery(
        expected,
        _transpose_layout(extension_coefficients),
        _transpose_layout(parameters),
        3,
    )
    torch.testing.assert_close(
        actual_extended, expected_extended, rtol=0, atol=0
    )


def test_cpu_indexed_ntt_auxiliary_strides() -> None:
    engine = _cpu_engine(p_count=1)
    backend = engine.rns_runtime.ntt_backend
    generator = torch.Generator().manual_seed(20260816)
    residues = torch.stack(
        [
            torch.randint(
                0,
                modulus,
                (engine.config.N,),
                dtype=engine.config.torch_dtype,
                generator=generator,
            )
            for modulus in engine.config.moduli
        ]
    )

    expected = ntt_ops.forward_ntt_to_montgomery_indexed(
        residues,
        backend.forward_even_indices,
        backend.forward_odd_indices,
        backend.forward_twiddles,
        backend.rns_params,
    )
    actual = ntt_ops.forward_ntt_to_montgomery_indexed(
        residues,
        _gap_last_axis(backend.forward_even_indices),
        _gap_last_axis(backend.forward_odd_indices),
        _gap_last_axis(backend.forward_twiddles),
        _transpose_layout(backend.rns_params),
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.gpu
def test_transpose_strides_preserve_cpu_cuda_native_semantics() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    engine = _cpu_engine(p_count=1)
    backend = engine.rns_runtime.ntt_backend
    generator = torch.Generator().manual_seed(20260818)
    residues = torch.stack(
        [
            torch.randint(
                0,
                modulus,
                (engine.config.N,),
                dtype=engine.config.torch_dtype,
                generator=generator,
            )
            for modulus in engine.config.moduli
        ]
    )
    even = _transpose_layout(backend.forward_even_indices)
    odd = _transpose_layout(backend.forward_odd_indices)
    parameters = _transpose_layout(backend.rns_params)
    cuda_even = even.cuda()
    cuda_odd = odd.cuda()
    cuda_parameters = parameters.cuda()
    assert cuda_even.stride() == even.stride()
    assert cuda_odd.stride() == odd.stride()
    assert cuda_parameters.stride() == parameters.stride()

    cpu_result = ntt_ops.forward_ntt_to_montgomery_indexed(
        residues,
        even,
        odd,
        backend.forward_twiddles,
        parameters,
    )
    cuda_result = ntt_ops.forward_ntt_to_montgomery_indexed(
        residues.cuda(),
        cuda_even,
        cuda_odd,
        backend.forward_twiddles.cuda(),
        cuda_parameters,
    ).cpu()
    moduli = torch.tensor(engine.config.moduli)[:, None]
    torch.testing.assert_close(
        torch.remainder(cuda_result, moduli),
        torch.remainder(cpu_result, moduli),
        rtol=0,
        atol=0,
    )


@pytest.mark.gpu
def test_cpu_created_values_remain_valid_after_cuda_transfer() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    cpu_engine = CkksEngine(
        Preset.slots8192_scale40_levels7_int64,
        device="cpu",
        rng_seed=911,
        rng_nonce=4,
    )
    secret_key = cpu_engine.create_secret_key()
    public_key = cpu_engine.create_public_key(secret_key)
    message = torch.linspace(-0.01, 0.01, 64, dtype=torch.float64).to(
        torch.complex128
    )
    ciphertext = cpu_engine.encrypt_message(message, public_key)
    cpu_plaintext = cpu_engine.decrypt(ciphertext, secret_key)

    cuda_engine = CkksEngine(
        Preset.slots8192_scale40_levels7_int64,
        device="cuda:0",
        allow_sk_gen=False,
    )
    cuda_ciphertext = ciphertext.to("cuda:0")
    cuda_secret_key = secret_key.to("cuda:0")
    cuda_plaintext = cuda_engine.decrypt(cuda_ciphertext, cuda_secret_key).to(
        "cpu"
    )
    torch.testing.assert_close(
        cuda_plaintext.data, cpu_plaintext.data, rtol=0, atol=0
    )

    cpu_ntt = cpu_engine.coefficient_domain_to_ntt_domain(ciphertext)
    cuda_coefficients = cuda_engine.ntt_domain_to_coefficient_domain(
        cpu_ntt.to("cuda:0")
    )
    moduli = torch.tensor(cpu_engine.config.q_moduli)[:, None]
    for component in range(ciphertext.data.size(0)):
        torch.testing.assert_close(
            torch.remainder(
                cuda_coefficients.data[component].cpu()
                - ciphertext.data[component],
                moduli,
            ),
            torch.zeros_like(ciphertext.data[component]),
            rtol=0,
            atol=0,
        )


def _moddown_reference(
    q_residues: torch.Tensor,
    p_residues: torch.Tensor,
    inverse_montgomery: torch.Tensor,
    moduli: tuple[int, ...],
    *,
    montgomery_radix: int,
) -> torch.Tensor:
    """Evaluate sequential P ModDown with Python integers."""

    q_count = q_residues.size(-2)
    p_count = p_residues.size(-2)
    coefficients = q_residues.size(-1)
    result = torch.empty_like(q_residues)
    for batch in range(q_residues.size(0)):
        for coefficient in range(coefficients):
            p_chain = [
                int(p_residues[batch, row, coefficient])
                for row in range(p_count)
            ]
            for row in range(p_count - 2, -1, -1):
                modulus = moduli[q_count + row]
                value = p_chain[row]
                radix_inverse = pow(montgomery_radix, -1, modulus)
                for lower in range(p_count - 1, row, -1):
                    inverse = (
                        int(
                            inverse_montgomery[
                                p_count - lower - 1, q_count + row
                            ]
                        )
                        * radix_inverse
                        % modulus
                    )
                    value = (value - p_chain[lower]) * inverse % modulus
                p_chain[row] = value

            for row in range(q_count):
                modulus = moduli[row]
                radix_inverse = pow(montgomery_radix, -1, modulus)
                value = int(q_residues[batch, row, coefficient])
                for p_row in range(p_count - 1, -1, -1):
                    inverse = (
                        int(inverse_montgomery[p_count - p_row - 1, row])
                        * radix_inverse
                        % modulus
                    )
                    value = (value - p_chain[p_row]) * inverse % modulus
                result[batch, row, coefficient] = value
    return result


def test_cpu_keyswitch_moddown_supports_more_than_eight_p_rows_and_strides() -> (
    None
):
    engine = _cpu_engine(p_count=9)
    q_moduli = engine.config.q_moduli
    p_moduli = engine.config.p_moduli
    generator = torch.Generator().manual_seed(20260817)
    q_residues = torch.stack(
        [
            torch.randint(
                0,
                modulus,
                (7,),
                dtype=engine.config.torch_dtype,
                generator=generator,
            )
            for modulus in q_moduli
        ]
    ).unsqueeze(0)
    p_residues = torch.stack(
        [
            torch.randint(
                0,
                modulus,
                (7,),
                dtype=engine.config.torch_dtype,
                generator=generator,
            )
            for modulus in p_moduli
        ]
    ).unsqueeze(0)
    inverse = engine.moddown_p_drop_inverses_montgomery_by_level[0]
    parameters = engine.rns_runtime.basis_parameters(
        0, include_p=True
    ).native_parameters

    expected = _moddown_reference(
        q_residues,
        p_residues,
        inverse,
        tuple(engine.config.moduli),
        montgomery_radix=engine.montgomery_parameters.R,
    )
    actual = ckks_ops.keyswitch_moddown_qp_to_q(
        q_residues,
        p_residues,
        _transpose_layout(inverse),
        _transpose_layout(parameters),
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "configuration",
    [
        CkksConfig(
            logN=12,
            num_scale_primes=3,
            num_p_primes=1,
            enforce_security_budget=False,
        ),
        Preset.slots16384_scale30_levels21_int64,
        CkksConfig(
            buffer_bit_length=30,
            scale_bits=25,
            logN=12,
            num_scale_primes=3,
            num_p_primes=2,
            enforce_security_budget=False,
        ),
        CkksConfig(
            logN=12,
            num_scale_primes=3,
            num_p_primes=4,
            enforce_security_budget=False,
        ),
        CkksConfig(
            logN=12,
            num_scale_primes=3,
            num_p_primes=6,
            enforce_security_budget=False,
        ),
    ],
    ids=[
        "custom-p1",
        "production-p2",
        "int32-p2",
        "custom-p4",
        "custom-p6",
    ],
)
def test_cuda_multistep_moddown_matches_integer_reference(
    configuration: Preset | CkksConfig,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    engine = CkksEngine(configuration, device="cpu", allow_sk_gen=False)
    key_switch_level_count = len(
        engine.moddown_p_drop_inverses_montgomery_by_level
    )
    levels = sorted(
        {0, key_switch_level_count // 2, key_switch_level_count - 1}
    )
    p_count = engine.config.num_p_primes
    for level in levels:
        generator = torch.Generator().manual_seed(20260819 + level)
        basis = engine.rns_runtime.basis_parameters(level, include_p=True)
        q_moduli = basis.moduli[:-p_count]
        p_moduli = basis.moduli[-p_count:]
        q_residues = torch.stack(
            [
                torch.randint(
                    0,
                    modulus,
                    (2, 11),
                    dtype=engine.config.torch_dtype,
                    generator=generator,
                )
                for modulus in q_moduli
            ],
            dim=1,
        )
        p_residues = torch.stack(
            [
                torch.randint(
                    0,
                    modulus,
                    (2, 11),
                    dtype=engine.config.torch_dtype,
                    generator=generator,
                )
                for modulus in p_moduli
            ],
            dim=1,
        )
        inverse = engine.moddown_p_drop_inverses_montgomery_by_level[level]
        parameters = basis.native_parameters
        expected = _moddown_reference(
            q_residues,
            p_residues,
            inverse,
            tuple(basis.moduli),
            montgomery_radix=engine.montgomery_parameters.R,
        )
        q_cuda = q_residues.cuda()
        p_cuda = p_residues.cuda()
        inverse_cuda = inverse.cuda()
        parameters_cuda = parameters.cuda()
        q_snapshot = q_cuda.clone()
        p_snapshot = p_cuda.clone()
        inverse_snapshot = inverse_cuda.clone()
        parameter_snapshot = parameters_cuda.clone()

        actual_cuda = ckks_ops.keyswitch_moddown_qp_to_q(
            q_cuda,
            p_cuda,
            inverse_cuda,
            parameters_cuda,
        )
        actual = actual_cuda.cpu()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual_cuda.untyped_storage().data_ptr() not in {
            q_cuda.untyped_storage().data_ptr(),
            p_cuda.untyped_storage().data_ptr(),
        }
        torch.testing.assert_close(q_cuda, q_snapshot, rtol=0, atol=0)
        torch.testing.assert_close(p_cuda, p_snapshot, rtol=0, atol=0)
        torch.testing.assert_close(
            inverse_cuda, inverse_snapshot, rtol=0, atol=0
        )
        torch.testing.assert_close(
            parameters_cuda, parameter_snapshot, rtol=0, atol=0
        )
