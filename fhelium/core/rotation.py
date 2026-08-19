"""Rotation-step decomposition helpers."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping


def decompose_power_of_two_rotation(
    rotation_step: int, num_slots: int
) -> list[int]:
    """Decompose a signed rotation into positive power-of-two steps."""

    if num_slots <= 0 or num_slots & (num_slots - 1):
        raise ValueError("num_slots must be a positive power of two")
    normalized = rotation_step % num_slots
    return [
        1 << exponent
        for exponent in range(num_slots.bit_length() - 1)
        if normalized & (1 << exponent)
    ]


def decompose_signed_power_of_two_rotation(
    rotation_step: int,
    num_slots: int,
) -> list[int]:
    """Decompose a cyclic rotation into a minimal signed power-of-two path.

    The result is the non-adjacent form of the canonical rotation in
    ``[-num_slots/2, num_slots/2)``.  Unlike
    :func:`decompose_power_of_two_rotation`, negative steps are retained, so a
    key planner can trade a smaller key inventory against composed rotations
    without turning a short negative rotation into a long positive path.
    """

    if num_slots <= 0 or num_slots & (num_slots - 1):
        raise ValueError("num_slots must be a positive power of two")
    remaining = (
        int(rotation_step) + num_slots // 2
    ) % num_slots - num_slots // 2
    power = 1
    decomposition: list[int] = []
    while remaining:
        if remaining & 1:
            digit = 2 - (remaining % 4)
            step = digit * power
            decomposition.append(
                (step + num_slots // 2) % num_slots - num_slots // 2
            )
            remaining -= digit
        remaining //= 2
        power *= 2
    return decomposition


def decompose_rotation_step(
    rotation_step: int,
    num_slots: int,
    rotation_keys: Mapping[int, object],
) -> list[int]:
    """Find the shortest cyclic decomposition using installed keys only."""

    if num_slots <= 0 or num_slots & (num_slots - 1):
        raise ValueError("num_slots must be a positive power of two")
    target = int(rotation_step) % num_slots
    if target == 0:
        return []

    step_by_residue: dict[int, int] = {}
    for raw_step in sorted(rotation_keys):
        step = int(raw_step)
        residue = step % num_slots
        if residue:
            step_by_residue.setdefault(residue, step)
    if not step_by_residue:
        raise KeyError(
            f"No installed rotation-key path can realize step {rotation_step}"
        )

    predecessor: dict[int, tuple[int, int] | None] = {0: None}
    queue: deque[int] = deque([0])
    while queue and target not in predecessor:
        accumulated = queue.popleft()
        for residue, step in step_by_residue.items():
            candidate = (accumulated + residue) % num_slots
            if candidate in predecessor:
                continue
            predecessor[candidate] = (accumulated, step)
            queue.append(candidate)

    if target not in predecessor:
        available = tuple(step_by_residue.values())
        raise KeyError(
            "No installed rotation-key path can realize step "
            f"{rotation_step}; installed steps are {available}"
        )

    path: list[int] = []
    cursor = target
    while cursor:
        edge = predecessor[cursor]
        assert edge is not None
        cursor, step = edge
        path.append(step)
    path.reverse()
    return path
