"""Behavior checks for CKKS operations that retain NTT-domain values."""

from __future__ import annotations

import pytest
import torch

from fhelium import CkksConfig, Preset
from fhelium.eager import Engine
from fhelium.values import Ciphertext


def _assert_congruent(
    engine: Engine,
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    *,
    depth: int,
) -> None:
    moduli = torch.tensor(
        engine.config.q_moduli[depth:],
        dtype=lhs.dtype,
        device=lhs.device,
    ).view(*([1] * (lhs.ndim - 2)), -1, 1)
    assert torch.count_nonzero(torch.remainder(lhs - rhs, moduli)) == 0


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_ntt_domain_ckks_operations_match_coefficient_results(
    device: str,
) -> None:
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.manual_seed(20260905)
    with torch.device(device):
        engine = Engine(
            Preset.slots8192_scale40_depth7_int64,
            allow_automatic_key_generation=False,
        )
        secret = engine.create_secret_key()
        public = engine.create_public_key(secret)
        relinearization = engine.create_relinearization_key(secret)
        destination = engine.create_secret_key()
        switching = engine.create_key_switch_key(secret, destination)
        conjugation = engine.create_conjugation_key(secret)
        rotation = engine.create_rotation_key(1, secret)
        message = 0.01 * torch.sin(
            torch.arange(engine.num_slots, dtype=torch.float64) * 0.01
        )

        encrypted_ntt = engine.encrypt_message(
            message,
            public,
            output_domain="ntt",
        )
        assert encrypted_ntt.polynomial_domain == "ntt"
        assert encrypted_ntt.residue_representation == "montgomery"
        torch.testing.assert_close(
            engine.decrypt_message(encrypted_ntt, secret, is_real=True),
            message.to(device),
            rtol=0.0,
            atol=5e-6,
        )

        source = engine.encrypt_message(message, public)
        source_ntt = engine.coefficient_domain_to_ntt_domain(source)
        product = engine.multiply(source_ntt, source_ntt)
        coefficient_relinearized = engine.relinearize(product, relinearization)
        ntt_relinearized = engine.relinearize(
            product,
            relinearization,
            output_domain="ntt",
        )
        _assert_congruent(
            engine,
            coefficient_relinearized.data,
            engine.ntt_domain_to_coefficient_domain(ntt_relinearized).data,
            depth=product.depth,
        )

        coefficient_rescaled = engine.rescale_to_next_depth(
            coefficient_relinearized
        )
        ntt_rescaled = engine.rescale_to_next_depth(ntt_relinearized)
        _assert_congruent(
            engine,
            coefficient_rescaled.data,
            engine.ntt_domain_to_coefficient_domain(ntt_rescaled).data,
            depth=coefficient_rescaled.depth,
        )

        coefficient_switched = engine.switch_key(source, switching)
        ntt_switched = engine.switch_key(source, switching, output_domain="ntt")
        _assert_congruent(
            engine,
            coefficient_switched.data,
            engine.ntt_domain_to_coefficient_domain(ntt_switched).data,
            depth=source.depth,
        )

        coefficient_conjugated = engine.conjugate(source, conjugation)
        ntt_conjugated = engine.conjugate(
            source, conjugation, output_domain="ntt"
        )
        _assert_congruent(
            engine,
            coefficient_conjugated.data,
            engine.ntt_domain_to_coefficient_domain(ntt_conjugated).data,
            depth=source.depth,
        )

        coefficient_rotated = engine.rotate_with_key(source, rotation)
        ntt_rotated = engine.rotate_with_key(
            source_ntt, rotation, output_domain="ntt"
        )
        _assert_congruent(
            engine,
            coefficient_rotated.data,
            engine.ntt_domain_to_coefficient_domain(ntt_rotated).data,
            depth=source.depth,
        )

        addend = engine.encode(
            torch.full_like(message, 0.002), scale=source.scale
        )
        coefficient_plaintext = engine.prepare_plaintext_for_addition(addend)
        ntt_plaintext = engine.prepare_plaintext_for_addition(
            addend,
            polynomial_domain="ntt",
        )
        coefficient_sum = engine.add_plaintext(source, coefficient_plaintext)
        ntt_sum = engine.add_plaintext(source_ntt, ntt_plaintext)
        _assert_congruent(
            engine,
            coefficient_sum.data,
            engine.ntt_domain_to_coefficient_domain(ntt_sum).data,
            depth=source.depth,
        )


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize(
    "preset",
    [
        Preset.slots8192_scale40_depth7_int64,
        Preset.slots8192_scale25_depth14_int32,
    ],
)
def test_plaintext_product_sum_matches_individual_operations(
    device: str,
    preset: Preset,
) -> None:
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.manual_seed(20260905)
    with torch.device(device):
        engine = Engine(
            preset,
            allow_automatic_key_generation=False,
        )
        secret = engine.create_secret_key()
        public = engine.create_public_key(secret)
        slots = 0.01 * torch.sin(
            torch.arange(engine.num_slots, dtype=torch.float64) * 0.01
        )
        message = torch.stack((slots, -slots))
        source = engine.coefficient_domain_to_ntt_domain(
            engine.encrypt_message(message, public)
        )
        ciphertexts = [source.clone() for _ in range(4)]
        plaintexts = [
            engine.prepare_plaintext_for_multiplication(
                engine.encode(
                    torch.full_like(message, (index + 1) / 7),
                    depth=source.depth,
                    scale=engine.config.default_scale,
                )
            )
            for index in range(4)
        ]
        source_tensors = [value.data.clone() for value in ciphertexts]
        plaintext_tensors = []
        for value in plaintexts:
            assert value.data is not None
            plaintext_tensors.append(value.data.clone())

        expected = engine.multiply_plaintext(ciphertexts[0], plaintexts[0])
        for ciphertext, plaintext in zip(
            ciphertexts[1:], plaintexts[1:], strict=True
        ):
            expected = engine.add(
                expected,
                engine.multiply_plaintext(ciphertext, plaintext),
            )
        actual = engine.sum_plaintext_products(ciphertexts, plaintexts)
        grouped = engine.sum_plaintext_product_groups(
            ciphertexts,
            (plaintexts, tuple(reversed(plaintexts))),
        )
        grouped_values = grouped.unbind_batch()
        reversed_expected = engine.sum_plaintext_products(
            ciphertexts,
            tuple(reversed(plaintexts)),
        )
        rotation_key = engine.create_rotation_key(1, secret)
        coefficient_source = engine.ntt_domain_to_coefficient_domain(source)
        (rotated_source,) = engine.rotate_many_with_keys(
            coefficient_source,
            (rotation_key,),
            use_hoisting=True,
            output_domain="ntt",
        )
        rotated_expected = engine.sum_plaintext_product_groups(
            [source, rotated_source],
            (plaintexts[:2], tuple(reversed(plaintexts[:2]))),
        )
        rotated_grouped = engine.sum_rotated_plaintext_product_groups(
            coefficient_source,
            (None, rotation_key),
            (plaintexts[:2], tuple(reversed(plaintexts[:2]))),
        )

        _assert_congruent(
            engine,
            actual.data,
            expected.data,
            depth=source.depth,
        )
        assert actual.depth == source.depth
        assert actual.scale == source.scale * plaintexts[0].scale
        assert actual.polynomial_domain == "ntt"
        assert actual.residue_representation == "montgomery"
        assert grouped.batch_shape[0] == 2
        _assert_congruent(
            engine,
            grouped_values[0].data,
            actual.data,
            depth=source.depth,
        )
        _assert_congruent(
            engine,
            grouped_values[1].data,
            reversed_expected.data,
            depth=source.depth,
        )
        assert grouped.scale == actual.scale
        assert grouped.polynomial_domain == "ntt"
        _assert_congruent(
            engine,
            rotated_grouped.data,
            rotated_expected.data,
            depth=source.depth,
        )
        for value, original in zip(
            (*ciphertexts, *plaintexts),
            (*source_tensors, *plaintext_tensors),
            strict=True,
        ):
            assert value.data is not None
            assert torch.equal(value.data, original)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_group_rescale_matches_integer_quotients(device: str) -> None:
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    base = CkksConfig.parse(Preset.slots8192_scale25_depth14_int32)
    q = base.q_moduli[:6]
    config = CkksConfig(
        default_scale=2.0**50,
        q_depth_groups=(q[:2], q[2:3], q[3:]),
        p_moduli=base.p_moduli,
        logN=13,
    )
    engine = Engine(config)
    divisor = config.rescale_divisor(0)
    x = (torch.arange(config.N, dtype=torch.int64, device=device) - config.N // 2) * 2**40
    coefficients = torch.stack((x, -x - 1))
    for basis in ("Q", "QP"):
        moduli = q + (config.p_moduli if basis == "QP" else ())
        modulus_tensor = torch.tensor(moduli, dtype=torch.int64, device=device).view(1, -1, 1)
        data = torch.remainder(coefficients.unsqueeze(-2), modulus_tensor).to(engine.dtype)
        source = Ciphertext(data=data, depth=0, scale=2.0**100, prime_ids=tuple(range(len(moduli))), modulus_basis=basis)
        for rounding in ("floor", "nearest"):
            quotient = torch.div(coefficients + (divisor // 2 if rounding == "nearest" else 0), divisor, rounding_mode="floor")
            expected = torch.remainder(quotient.unsqueeze(-2), modulus_tensor[:, 2:, :]).to(engine.dtype)
            for value in (source, engine.coefficient_domain_to_ntt_domain(source)):
                result = engine.rescale_to_next_depth(value, rounding=rounding)
                coefficient_result = result if result.polynomial_domain == "coefficient" else engine.ntt_domain_to_coefficient_domain(result)
                assert torch.equal(coefficient_result.data, expected)
                assert result.depth == 1
                assert result.scale == source.scale / divisor
