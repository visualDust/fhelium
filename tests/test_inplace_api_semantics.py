import pytest
import torch

import fhelium.native  # noqa: F401 - loads native torch operator libraries
from fhelium import CkksEngine, Preset
from fhelium.native.wrapper import ckks_ops, ntt_ops, rns_ops


@pytest.mark.parametrize(
    ("namespace", "op_name", "wrapper"),
    [
        (
            "fhelium_ckks_ops",
            "add_prepared_plaintext_component_",
            ckks_ops.add_prepared_plaintext_component_,
        ),
        ("fhelium_rns_ops", "add_canonical_", rns_ops.add_canonical_),
        (
            "fhelium_ntt_ops",
            "forward_ntt_to_montgomery_indexed_",
            ntt_ops.forward_ntt_to_montgomery_indexed_,
        ),
    ],
)
def test_representative_native_inplace_schemas_declare_aliasing(
    namespace: str,
    op_name: str,
    wrapper,
) -> None:
    torch_namespace = getattr(torch.ops, namespace)
    schema = next(iter(getattr(torch_namespace, op_name)._schemas.values()))
    functional_name = op_name.removesuffix("_")
    functional_schema = next(
        iter(getattr(torch_namespace, functional_name)._schemas.values())
    )

    assert wrapper.__name__ == op_name
    assert "(a!)" in str(schema)
    assert "(a!)" not in str(functional_schema)


@pytest.mark.gpu
def test_ciphertext_add_inplace_reuses_storage_without_mutating_inputs() -> (
    None
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    engine = CkksEngine(Preset.slots8192_scale40_levels7_int64, device="cuda:0")
    left = engine.encrypt_message(torch.randn(32, dtype=torch.float64) * 0.01)
    right = engine.encrypt_message(torch.randn(32, dtype=torch.float64) * 0.01)
    left_before = left.data.clone()
    right_before = right.data.clone()

    expected = engine.add(left, right)
    assert torch.equal(left.data, left_before)
    assert torch.equal(right.data, right_before)

    mutable = left.clone()
    storage = mutable.data.untyped_storage().data_ptr()
    assert engine.add_(mutable, right) is mutable
    assert mutable.data.untyped_storage().data_ptr() == storage
    assert torch.equal(mutable.data, expected.data)
    assert torch.equal(right.data, right_before)
