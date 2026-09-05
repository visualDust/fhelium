"""Behavior checks for CKKS operations that retain NTT-domain values."""

from __future__ import annotations

import pytest
import torch

from fhelium import Preset
from fhelium.eager import Engine


def _assert_congruent(
    engine: Engine,
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    *,
    level: int,
) -> None:
    moduli = torch.tensor(
        engine.config.q_moduli[level:],
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
            Preset.slots8192_scale40_levels7_int64,
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
            level=product.level,
        )

        coefficient_rescaled = engine.rescale_to_next_level(
            coefficient_relinearized
        )
        ntt_rescaled = engine.rescale_to_next_level(ntt_relinearized)
        _assert_congruent(
            engine,
            coefficient_rescaled.data,
            engine.ntt_domain_to_coefficient_domain(ntt_rescaled).data,
            level=coefficient_rescaled.level,
        )

        coefficient_switched = engine.switch_key(source, switching)
        ntt_switched = engine.switch_key(source, switching, output_domain="ntt")
        _assert_congruent(
            engine,
            coefficient_switched.data,
            engine.ntt_domain_to_coefficient_domain(ntt_switched).data,
            level=source.level,
        )

        coefficient_conjugated = engine.conjugate(source, conjugation)
        ntt_conjugated = engine.conjugate(
            source, conjugation, output_domain="ntt"
        )
        _assert_congruent(
            engine,
            coefficient_conjugated.data,
            engine.ntt_domain_to_coefficient_domain(ntt_conjugated).data,
            level=source.level,
        )

        coefficient_rotated = engine.rotate_with_key(source, rotation)
        ntt_rotated = engine.rotate_with_key(
            source_ntt, rotation, output_domain="ntt"
        )
        _assert_congruent(
            engine,
            coefficient_rotated.data,
            engine.ntt_domain_to_coefficient_domain(ntt_rotated).data,
            level=source.level,
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
            level=source.level,
        )
