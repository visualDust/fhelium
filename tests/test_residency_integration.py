from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import pytest
import torch

from fhelium import Ciphertext, CkksEngine, Plaintext, Preset, RotationKey
from fhelium.artifacts import ArtifactRef, ArtifactStore
from fhelium.errors import ResidencyLifetimeClosedError
from fhelium.residency import (
    PAGEABLE_HOST,
    MoveResident,
    Recoverability,
    ReplicaMode,
    ResidencyHandle,
    ResidencyLocation,
    ResidencyManager,
    ResidencyPlan,
    ResidencyValueSpec,
    cuda_location,
)

_KEYSWITCH_ATOL = 2e-5


def _value_snapshot(
    manager: ResidencyManager,
    handle: ResidencyHandle[Any],
) -> Any:
    return next(
        value for value in manager.snapshot().values if value.handle == handle
    )


def _locations(
    manager: ResidencyManager,
    handle: ResidencyHandle[Any],
) -> tuple[ResidencyLocation, ...]:
    return tuple(
        materialization.location
        for materialization in _value_snapshot(
            manager,
            handle,
        ).materializations
    )


class _ArtifactPlaintextSource:
    def __init__(
        self,
        store: ArtifactStore,
        reference: ArtifactRef[Plaintext],
    ) -> None:
        self._store = store
        self._reference = reference
        self.load_count = 0

    def load(self) -> Plaintext:
        self.load_count += 1
        return self._store.get(
            self._reference,
            device="cpu",
            expected_type=Plaintext,
        )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_ckks_values_follow_plan_and_lease_through_real_operation() -> None:
    engine = CkksEngine(
        Preset.slots8192_scale40_levels7_int64,
        device="cuda:0",
    )
    manager = ResidencyManager()
    device_location = cuda_location(engine.device)
    try:
        index = torch.arange(engine.num_slots, dtype=torch.float64)
        message = 0.012 * torch.sin(index * 0.013)
        addend_message = torch.full_like(message, 0.125)

        ciphertext_value = engine.encrypt_message(message).cpu()
        plaintext_value = engine.prepare_plaintext_for_addition(
            engine.encode(addend_message, level=ciphertext_value.level)
        ).cpu()
        rotation_key_value = engine.create_rotation_key(
            1,
            engine.secret_key,
        ).cpu()

        ciphertext = manager.adopt(ciphertext_value)
        plaintext = manager.adopt(plaintext_value)
        rotation_key = manager.adopt(rotation_key_value)
        del ciphertext_value, plaintext_value, rotation_key_value

        handles = (ciphertext, plaintext, rotation_key)
        assert tuple(_locations(manager, handle) for handle in handles) == (
            (PAGEABLE_HOST,),
            (PAGEABLE_HOST,),
            (PAGEABLE_HOST,),
        )

        enter = tuple(
            MoveResident(
                handle,
                device_location,
                from_location=PAGEABLE_HOST,
            )
            for handle in handles
        )
        exit_actions = tuple(
            MoveResident(
                handle,
                PAGEABLE_HOST,
                from_location=device_location,
            )
            for handle in handles
        )
        plan = ResidencyPlan(
            "ckks-rotation-window",
            enter=enter,
            exit=exit_actions,
        )
        explanation = manager.explain(plan)
        assert explanation.feasible
        assert tuple(item.action for item in explanation.actions) == (
            *enter,
            *exit_actions,
        )

        scope = manager.scope(plan)
        with scope:
            assert (
                tuple(_locations(manager, handle) for handle in handles)
                == ((device_location,),) * 3
            )

            stream = torch.cuda.current_stream(engine.device)
            lease = manager.acquire(
                handles,
                at=device_location,
                consumer_stream=stream,
            )
            borrowed = lease.values
            borrowed_ciphertext = None
            borrowed_plaintext = None
            borrowed_rotation_key = None
            try:
                borrowed_ciphertext = borrowed[ciphertext]
                borrowed_plaintext = borrowed[plaintext]
                borrowed_rotation_key = borrowed[rotation_key]
                assert isinstance(borrowed_ciphertext, Ciphertext)
                assert isinstance(borrowed_plaintext, Plaintext)
                assert isinstance(borrowed_rotation_key, RotationKey)
                assert all(
                    _value_snapshot(
                        manager,
                        handle,
                    )
                    .materializations[0]
                    .use_count
                    == 1
                    for handle in handles
                )

                added = engine.add_plaintext(
                    borrowed_ciphertext,
                    borrowed_plaintext,
                )
                rotated = engine.rotate_with_key(
                    added,
                    borrowed_rotation_key,
                )
                decoded = torch.as_tensor(
                    engine.decrypt_message(rotated)
                ).resolve_conj()
                expected = torch.roll(
                    message + addend_message,
                    shifts=1,
                ).to(decoded.device)
                max_error = float(torch.max(torch.abs(decoded - expected)))
                assert max_error < _KEYSWITCH_ATOL
            finally:
                del (
                    borrowed_ciphertext,
                    borrowed_plaintext,
                    borrowed_rotation_key,
                )
                lease.release(wait=True)

            assert all(
                _value_snapshot(manager, handle).materializations[0].use_count
                == 0
                for handle in handles
            )
            with pytest.raises(ResidencyLifetimeClosedError):
                borrowed[ciphertext]

        assert scope.report is not None
        assert tuple(
            transition.action for transition in scope.report.transitions
        ) == (*enter, *exit_actions)
        assert tuple(_locations(manager, handle) for handle in handles) == (
            (PAGEABLE_HOST,),
            (PAGEABLE_HOST,),
            (PAGEABLE_HOST,),
        )
    finally:
        manager.close()
        del engine
        gc.collect()
        torch.cuda.empty_cache()


def test_artifact_source_reconstructs_exact_value_and_obeys_lifetimes(
    tmp_path: Path,
) -> None:
    store_root = tmp_path / "artifacts"
    original = Plaintext(
        message=None,
        level=2,
        scale=2.0**30,
        data=torch.arange(16, dtype=torch.int64),
        context_id="residency-integration-context",
        representation="integer_coefficients",
        polynomial_domain="coefficient",
    )
    spec = ResidencyValueSpec(
        value_type=Plaintext,
        logical_nbytes=original.nbytes,
        storage_nbytes=original.storage_nbytes,
        replica_mode=ReplicaMode.REPLICABLE,
        recoverability=Recoverability.RECONSTRUCTIBLE,
    )
    assert original.data is not None
    expected_data = original.data.clone()
    reference = ArtifactStore(store_root).put(
        "integration/exact-plaintext",
        original,
    )
    del original

    source = _ArtifactPlaintextSource(ArtifactStore(store_root), reference)
    manager = ResidencyManager({PAGEABLE_HOST: spec.storage_nbytes})
    try:
        handle = manager.register_source(spec, source)
        registered = _value_snapshot(manager, handle)
        assert registered.has_source
        assert registered.source_location == PAGEABLE_HOST
        assert registered.materializations == ()

        manager.ensure(handle, PAGEABLE_HOST)
        assert source.load_count == 1
        lease = manager.acquire((handle,), at=PAGEABLE_HOST)
        borrowed = lease.values
        restored = None
        try:
            restored = borrowed[handle]
            assert isinstance(restored, Plaintext)
            assert restored.level == 2
            assert restored.scale == 2.0**30
            assert restored.context_id == "residency-integration-context"
            assert restored.representation == "integer_coefficients"
            assert restored.polynomial_domain == "coefficient"
            assert restored.modulus_basis is None
            assert restored.residue_representation is None
            assert restored.prime_ids == ()
            assert restored.data is not None
            torch.testing.assert_close(
                restored.data,
                expected_data,
                rtol=0,
                atol=0,
            )
            materialization = _value_snapshot(
                manager,
                handle,
            ).materializations[0]
            assert materialization.location == PAGEABLE_HOST
            assert materialization.logical_nbytes == spec.logical_nbytes
            assert materialization.storage_nbytes == spec.storage_nbytes
            assert materialization.charged_nbytes == spec.storage_nbytes
            assert materialization.use_count == 1
        finally:
            del restored
            lease.release()

        with pytest.raises(ResidencyLifetimeClosedError):
            borrowed[handle]
        assert (
            _value_snapshot(manager, handle).materializations[0].use_count == 0
        )

        manager.drop(handle, PAGEABLE_HOST)
        dropped = _value_snapshot(manager, handle)
        assert dropped.has_source
        assert dropped.materializations == ()
        manager.ensure(handle, PAGEABLE_HOST)
        assert source.load_count == 2
    finally:
        manager.close()
