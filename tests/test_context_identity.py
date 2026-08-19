"""Distinct public ciphertext context-safety regression."""

import pytest
import torch

from fhelium import CkksEngine, Preset


@pytest.mark.gpu
def test_galois_generator_is_part_of_context_identity() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    generator3 = CkksEngine(
        Preset.slots8192_scale40_levels7_int64, device="cuda:0"
    )
    generator5 = CkksEngine(
        Preset.slots8192_scale40_levels7_int64,
        device="cuda:0",
        galois_generator=5,
    )

    assert generator3.config.dumps() == generator5.config.dumps()
    assert generator3.context.context_id != generator5.context.context_id
    with pytest.raises(ValueError, match="context"):
        generator3.encrypt_message(
            torch.zeros(16, dtype=torch.float64),
            generator5.public_key,
        )
