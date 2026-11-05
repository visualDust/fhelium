"""Evaluation-key inventories retain objects and select compatible physical layouts."""

import pytest
import torch

from fhelium.values import (
    ConjugationKey,
    EvaluationKeySet,
    KeySwitchKey,
    RelinearizationKey,
    RotationKey,
)


def key(
    role=RotationKey, step=1, *, compact=False, device='cpu', dtype=torch.int32
):
    prime_ids = tuple(range(2 if compact else 0, 8))
    digits = ((2, 3), (4, 5)) if compact else ((0, 1), (2, 3), (4, 5))
    return role(
        data=torch.empty(
            len(digits), 2, len(prime_ids), 32, dtype=dtype, device=device
        ),
        prime_ids=prime_ids,
        digit_prime_ids=digits,
        **({'rotation_step': step} if role is RotationKey else {}),
    )


def facts():
    return dict(
        q_prime_ids=(2, 3, 4, 5),
        p_prime_ids=(6, 7),
        ring_dimension=32,
        dtype=torch.int32,
        device=torch.device('cpu'),
    )


def test_inventory_retains_object_identity_and_role_views():
    rotation, relin, conjugation = (
        key(),
        key(RelinearizationKey),
        key(ConjugationKey),
    )
    inventory = EvaluationKeySet((rotation, relin))
    assert inventory.add(rotation).update((conjugation, rotation)) is inventory
    assert len(inventory) == 3
    assert tuple(inventory) == (rotation, relin, conjugation)
    assert inventory.rotation_keys == (rotation,)
    assert inventory.relinearization_keys == (relin,)
    assert inventory.conjugation_keys == (conjugation,)
    clone = rotation.clone()
    inventory.add(clone)
    assert len(inventory) == 4 and inventory.rotation_keys[-1] is clone
    inventory.remove(clone)
    assert inventory.rotation_keys == (rotation,)
    with pytest.raises(KeyError):
        inventory.remove(clone)
    selected = inventory.select_rotations((1,), **facts())
    inventory.clear()
    assert len(inventory) == 0 and inventory.rotation_keys == ()
    assert selected[1] is rotation
    with pytest.raises(
        TypeError, match='rotation, relinearization and conjugation'
    ):
        inventory.add(key(KeySwitchKey))


def test_rotation_selection_requires_one_complete_layout_and_actual_coverage():
    full1, full2 = key(step=1), key(step=2)
    compact1, compact2 = key(step=1, compact=True), key(step=2, compact=True)
    inventory = EvaluationKeySet((full1, compact1, full2))
    selected = inventory.select_rotations((1, 2), **facts())
    assert selected[1] is full1 and selected[2] is full2
    inventory.update((compact2, key(step=1, compact=True, device='meta')))
    selected = inventory.select_rotations((1, 2), **facts())
    assert selected[1] is compact1 and selected[2] is compact2
    constrained = inventory.select_rotations(
        (1, 2),
        key_prime_ids=full1.prime_ids,
        digit_prime_ids=full1.digit_prime_ids,
        **facts(),
    )
    assert constrained[1] is full1 and constrained[2] is full2
    for name, value in (
        ('p_prime_ids', (7,)),
        ('ring_dimension', 16),
        ('dtype', torch.int64),
    ):
        conditions = facts()
        conditions[name] = value
        assert inventory.select_rotations((1, 2), **conditions) is None
    compact_only = EvaluationKeySet((compact1, compact2))
    conditions = facts()
    conditions['q_prime_ids'] = tuple(range(6))
    assert compact_only.select_rotations((1, 2), **conditions) is None
    assert len(EvaluationKeySet().select_rotations((0, 16), **facts())) == 0


def test_composed_rotation_selection_and_unknown_candidates():
    left, right = key(step=-1), key(step=4)
    inventory = EvaluationKeySet((left, right))
    assert inventory.select_rotations((3,), **facts()) is None
    selected = inventory.select_rotations((3,), allow_composed=True, **facts())
    assert set(selected) == {-1, 4}
    direct = key(step=3)
    inventory.add(direct)
    assert tuple(
        inventory.select_rotations(
            (3,), allow_composed=True, **facts()
        ).values()
    ) == (direct,)
    assert EvaluationKeySet((direct,)).select_rotations((3,))[3] is direct
    inventory.add(key(step=3, compact=True))
    assert inventory.select_rotations((3,)) is None
    assert (
        EvaluationKeySet((direct, direct.clone())).select_rotations((3,))
        is None
    )
    mixed = EvaluationKeySet((left, key(step=4, compact=True)))
    assert mixed.select_rotations((3,), allow_composed=True, **facts()) is None


@pytest.mark.parametrize(
    'role,selector',
    (
        (RelinearizationKey, 'select_relinearization'),
        (ConjugationKey, 'select_conjugation'),
    ),
)
def test_single_role_selection_returns_references_without_generation(
    role, selector
):
    full, compact = key(role), key(role, compact=True)
    inventory = EvaluationKeySet((full, compact))
    assert getattr(inventory, selector)(**facts()) is compact
    assert getattr(inventory, selector)() is None
    conditions = facts()
    conditions['q_prime_ids'] = tuple(range(6))
    assert getattr(inventory, selector)(**conditions) is full


def test_composed_selection_counts_repeated_primitive_invocations():
    composed = [
        RotationKey(
            data=torch.empty(2, 2, 8, 32, dtype=torch.int32),
            prime_ids=tuple(range(8)),
            digit_prime_ids=((0, 1, 2), (3, 4, 5)),
            rotation_step=step,
        )
        for step in (-1, 1, 4)
    ]
    direct = (key(step=3), key(step=5))
    inventory = EvaluationKeySet((*composed, *direct))
    conditions = facts()
    conditions['q_prime_ids'] = tuple(range(6))
    selected = inventory.select_rotations(
        (3, 5), allow_composed=True, **conditions
    )
    # Composing both offsets uses four switches, including the step-4 key twice.
    assert set(selected) == {3, 5}
    assert selected[3] is direct[0] and selected[5] is direct[1]
