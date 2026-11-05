"""Derive Bootstrap key consumers and add missing primitive evaluation keys."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, cast

import torch

from fhelium.backend.rns.decomposition import _partition_q_digits
from fhelium.utils.rotation import decompose_signed_power_of_two_rotation
from fhelium.values import EvaluationKeySet, RotationKey, SecretKey

from .linear.evaluators import DiagonalBSGSEvaluator, DirectDiagonalEvaluator

if TYPE_CHECKING:
    from .full_slot import FullSlotBootstrap


# Generation depth, actual stored prime rows, and actual stored digit supports.
_Layout = tuple[int, tuple[int, ...], tuple[tuple[int, ...], ...]]
_Addition = tuple[int, _Layout]
_Consumer = tuple[int | None, tuple[int, ...], str, tuple[int, ...]]


def _offsets(steps: Iterable[int], ring_dimension: int) -> tuple[int, ...]:
    normalized = {
        RotationKey.normalize_step(step, ring_dimension=ring_dimension)
        for step in steps
    }
    return tuple(sorted(normalized - {0}))


def _stage_rotation_uses(
    evaluator: Any, transform: Any, input_depth: int
) -> tuple[tuple[int | None, tuple[int, ...], str], ...]:
    """Describe the depth, logical offsets and sharing of each rotation consumer.

    Hoisted babies share decomposition only when their selected direct keys have one stored layout. Aggregate giants require a common layout for their final key switches; composed prefixes remain independent. Unknown custom consumers retain full-basis coverage and one common layout for every primitive.
    """
    ring_dimension = 2 * transform.slots
    if type(evaluator) is DirectDiagonalEvaluator:
        return (
            (
                input_depth,
                _offsets(
                    evaluator.required_rotation_offsets(transform),
                    ring_dimension,
                ),
                "independent",
            ),
        )
    if type(evaluator) is DiagonalBSGSEvaluator:
        groups = evaluator._partition(transform)
        babies = _offsets(
            (baby for terms in groups.values() for baby, _, _ in terms),
            ring_dimension,
        )
        giants = _offsets(groups, ring_dimension)
        uses = (
            (
                input_depth,
                babies,
                "hoisted" if evaluator.hoist_baby_rotations else "independent",
            ),
        )
        if giants:
            aggregate = evaluator.giant_step_reduction == "aggregate"
            uses += (
                (
                    input_depth if aggregate else input_depth + 1,
                    giants,
                    "aggregate" if aggregate else "independent",
                ),
            )
        return uses
    return (
        (
            None,
            _offsets(
                evaluator.required_rotation_offsets(transform), ring_dimension
            ),
            "unknown",
        ),
    )


def _select_rotation_paths(
    engine,
    inventory: EvaluationKeySet,
    depth: int | None,
    offsets: Iterable[int],
    kind: str,
    device: torch.device,
    direct_steps: Iterable[int] = (),
) -> dict[int, tuple[RotationKey, ...]] | None:
    """Select ordinary primitive key references for each nonzero logical step.

    Independent paths may change stored layout between primitive rotations. Hoisting prefers one common direct group and otherwise uses independent paths. Aggregate final keys share a stored layout while prefixes may differ. Unknown consumers use full Q coverage and one layout for the complete paths. Steps in ``direct_steps`` cannot be satisfied by composition.
    """
    requested = _offsets(offsets, engine.config.N)
    if not requested:
        return {}
    forced = set(_offsets(direct_steps, engine.config.N))
    geometry = _geometry(engine, None if kind == "unknown" else depth, device)
    cache: dict[tuple[object, int], RotationKey | None] = {}

    def primitive(step, layout=None):
        token = (layout, step)
        if token not in cache:
            constraints = (
                {}
                if layout is None
                else {"key_prime_ids": layout[0], "digit_prime_ids": layout[1]}
            )
            selected = inventory.select_rotations(
                (step,), **geometry, **constraints
            )
            cache[token] = None if selected is None else selected[step]
        return cache[token]

    if kind == "hoisted":
        shared = inventory.select_rotations(requested, **geometry)
        if shared is not None:
            return {step: (shared[step],) for step in requested}

    def paths(layout=None):
        result: dict[int, tuple[RotationKey, ...]] = {}
        for step in requested:
            direct = primitive(step, layout)
            if direct is not None:
                result[step] = (direct,)
                continue
            if step in forced:
                return None
            decomposition = decompose_signed_power_of_two_rotation(
                step, engine.num_slots
            )
            keys = tuple(
                primitive(
                    part,
                    layout
                    if kind == "unknown" or i == len(decomposition) - 1
                    else None,
                )
                for i, part in enumerate(decomposition)
            )
            if any(key is None for key in keys):
                return None
            result[step] = cast(tuple[RotationKey, ...], keys)
        return result

    if kind not in {"aggregate", "unknown"}:
        return paths()
    layouts = dict.fromkeys(
        (key.prime_ids, key.digit_prime_ids) for key in inventory.rotation_keys
    )
    choices = [
        selected
        for layout in layouts
        if (selected := paths(layout)) is not None
    ]
    if not choices:
        return None
    q_ids = set(geometry["q_prime_ids"])

    def rank(selected):
        calls = tuple(key for path in selected.values() for key in path)
        unique = {id(key): key for key in calls}
        return (
            len(calls),
            sum(
                sum(
                    any(i in q_ids for i in rows)
                    for rows in key.digit_prime_ids
                )
                for key in calls
            ),
            sum(key.nbytes for key in unique.values()),
        )

    return min(choices, key=rank)


def _consumers(
    bootstrap: FullSlotBootstrap, strategy: str
) -> tuple[tuple[_Consumer, ...], int, int]:
    consumers: list[_Consumer] = []
    depth = bootstrap.modulus_raise_target_depth
    reduction_start = reduction_stop = depth
    for stages, evaluator, forward in (
        (bootstrap.coeffs_to_slots, bootstrap.coeffs_to_slots_evaluator, True),
        (bootstrap.slots_to_coeffs, bootstrap.slots_to_coeffs_evaluator, False),
    ):
        for transform in stages:
            direct_babies: tuple[int, ...] = ()
            baby_offsets = getattr(evaluator, "baby_rotation_offsets", None)
            if strategy == "signed_power_of_two_with_baby_steps" and callable(
                baby_offsets
            ):
                direct_babies = _offsets(
                    cast(Iterable[int], baby_offsets(transform)),
                    bootstrap.engine.config.N,
                )
            for index, (consumer_depth, offsets, kind) in enumerate(
                _stage_rotation_uses(evaluator, transform, depth)
            ):
                forced = (
                    offsets
                    if strategy == "all_required"
                    else (direct_babies if index == 0 else ())
                )
                consumers.append((consumer_depth, offsets, kind, forced))
            depth += evaluator.required_depths(transform)
        if forward:
            depth += 1
            reduction_start = depth
            depth += bootstrap.modular_reduction.required_depths
            reduction_stop = depth
    return tuple(consumers), reduction_start, reduction_stop


def _geometry(
    engine, depth: int | None, device: torch.device
) -> dict[str, Any]:
    config = engine.config
    return {
        "q_prime_ids": tuple(
            range(
                config.q_row_start(0 if depth is None else depth),
                config.num_q_primes,
            )
        ),
        "p_prime_ids": None,
        "ring_dimension": config.N,
        "dtype": engine.dtype,
        "device": device,
    }


def _layout(
    engine, depth: int, depth_zero_digits: tuple[tuple[int, ...], ...]
) -> _Layout:
    """Describe the configured full layout or the factory's active greedy layout."""
    config = engine.config
    start = config.q_row_start(depth)
    q_ids = tuple(range(start, config.num_q_primes))
    digits = (
        depth_zero_digits
        if depth == 0
        else _partition_q_digits(
            q_ids, config.q_moduli[start:], config.p_moduli
        )
    )
    return depth, tuple(range(start, config.total_num_primes)), digits


def _rotation_additions(
    bootstrap: FullSlotBootstrap,
    inventory: EvaluationKeySet,
    consumers: Sequence[_Consumer],
    strategy: str,
    device: torch.device,
) -> tuple[_Addition, ...]:
    """Choose missing key specifications using the consumers' actual sharing.

    Independent primitives choose layouts separately. Requested direct hoists that need new material retain one stored layout; already complete mixed paths are reused without generation. Aggregate consumers couple only their final keys, and unknown custom consumers couple every primitive at full depth. Candidate cost is the union of newly generated payloads across all active consumers. Equal-storage choices prefer preserving shared direct hoists. Full, active and common-layout starts seed deterministic coordinate improvement without a global-optimality claim.
    """
    engine = bootstrap.engine
    consumers = tuple(
        (
            None if kind == "unknown" else depth,
            _offsets(steps, engine.config.N),
            kind,
            _offsets(
                steps if strategy == "all_required" else direct, engine.config.N
            ),
        )
        for depth, steps, kind, direct in consumers
    )
    satisfied = tuple(
        _select_rotation_paths(
            engine, inventory, depth, steps, kind, device, direct
        )
        is not None
        for depth, steps, kind, direct in consumers
    )
    if all(satisfied):
        return ()

    depth_zero_digits = tuple(
        tuple(rows)
        for rows in engine.rns_context(
            device
        ).hybrid_decomposition.depth_zero_digits
    )
    full = _layout(engine, 0, depth_zero_digits)
    layouts = {
        full,
        *(
            _layout(engine, depth, depth_zero_digits)
            for depth, _, _, _ in consumers
            if depth is not None
        ),
    }
    basis_depth = {
        tuple(
            range(engine.config.q_row_start(depth), engine.config.num_q_primes)
        ): depth
        for depth in range(engine.max_depth + 1)
    }
    for key in inventory.rotation_keys:
        stored_q = tuple(row for digit in key.digit_prime_ids for row in digit)
        depth = basis_depth.get(stored_q)
        if (
            depth is not None
            and EvaluationKeySet((key,)).select_rotations(
                (key.rotation_step,), **_geometry(engine, depth, device)
            )
            is not None
        ):
            layouts.add((depth, key.prime_ids, key.digit_prime_ids))
    ordered_layouts = tuple(sorted(layouts))
    physical = dict.fromkeys(
        (key.prime_ids, key.digit_prime_ids) for key in inventory.rotation_keys
    )
    physical.update({layout[1:]: None for layout in ordered_layouts})
    reproducible = {layout[1:]: layout for layout in ordered_layouts}
    available_cache: dict[tuple[object, object, int], bool] = {}

    def available(depth, step, layout=None):
        token = (depth, layout, step)
        if token not in available_cache:
            constraints = (
                {}
                if layout is None
                else {"key_prime_ids": layout[0], "digit_prime_ids": layout[1]}
            )
            available_cache[token] = (
                inventory.select_rotations(
                    (step,), **_geometry(engine, depth, device), **constraints
                )
                is not None
            )
        return available_cache[token]

    # An independent need is (consumption depth, primitive step). Group choices
    # activate only the prefixes needed by that choice of final stored layout.
    independent: set[tuple[int | None, int]] = set()
    groups = []
    group_depths = []
    for (depth, steps, kind, forced_steps), ready in zip(
        consumers, satisfied, strict=True
    ):
        if ready:
            continue
        forced = set(forced_steps)
        shared_direct = kind == "hoisted" and set(steps).issubset(forced)
        if kind == "independent" or (kind == "hoisted" and not shared_direct):
            for step in steps:
                if available(depth, step):
                    continue
                path = (
                    (step,)
                    if step in forced
                    else tuple(
                        decompose_signed_power_of_two_rotation(
                            step, engine.num_slots
                        )
                    )
                )
                independent.update(
                    (depth, part) for part in path if not available(depth, part)
                )
            continue
        choices = []
        for stored in physical:
            layout = reproducible.get(stored)
            if layout is not None and layout[0] > (
                0 if depth is None else depth
            ):
                continue
            added: set[_Addition] = set()
            prefixes: set[tuple[int | None, int]] = set()
            valid = True
            for step in steps:
                if available(depth, step, stored):
                    continue
                path = (
                    (step,)
                    if step in forced
                    else tuple(
                        decompose_signed_power_of_two_rotation(
                            step, engine.num_slots
                        )
                    )
                )
                coupled = path if kind == "unknown" else path[-1:]
                for part in coupled:
                    if not available(depth, part, stored):
                        if layout is None:
                            valid = False
                            break
                        added.add((part, layout))
                if not valid:
                    break
                if kind == "aggregate":
                    prefixes.update(
                        (depth, part)
                        for part in path[:-1]
                        if not available(depth, part)
                    )
            if valid:
                choices.append((layout, frozenset(added), frozenset(prefixes)))
        groups.append(tuple(choices))
        group_depths.append(depth)

    needs = independent | {
        need
        for choices in groups
        for _, _, required in choices
        for need in required
    }
    ordered_needs = tuple(
        sorted(
            needs,
            key=lambda item: (-1 if item[0] is None else item[0], item[1]),
        )
    )
    need_indices = {need: index for index, need in enumerate(ordered_needs)}
    options = [
        *groups,
        *(
            tuple(
                (layout, frozenset(((step, layout),)), frozenset())
                for layout in ordered_layouts
                if layout[0] <= (0 if depth is None else depth)
            )
            for depth, step in ordered_needs
        ),
    ]
    depths = (*group_depths, *(depth for depth, _ in ordered_needs))
    bytes_per_element = torch.iinfo(engine.dtype).bits // 8
    hoists = []
    for depth, steps, kind, _ in consumers:
        if kind == "hoisted" and steps:
            existing = {
                stored: {
                    step for step in steps if available(depth, step, stored)
                }
                for stored in physical
            }
            hoists.append((depth, set(steps), existing))

    def additions(assignment):
        active = set(independent)
        added = set()
        for choices, index in zip(
            groups, assignment[: len(groups)], strict=True
        ):
            _, material, required = choices[index]
            added.update(material)
            active.update(required)
        for need in active:
            index = len(groups) + need_indices[need]
            added.update(options[index][assignment[index]][1])
        return frozenset(added)

    def score(assignment):
        unique = additions(assignment)
        payload = sum(
            2
            * engine.config.N
            * len(layout[1])
            * len(layout[2])
            * bytes_per_element
            for _, layout in unique
        )
        unshared = 0
        for depth, steps, existing in hoists:
            combined = {
                stored: set(present) for stored, present in existing.items()
            }
            for step, layout in unique:
                if layout[0] <= (0 if depth is None else depth):
                    combined.setdefault(layout[1:], set()).add(step)
            unshared += not any(
                steps <= present for present in combined.values()
            )
        return payload, unshared, len(unique), tuple(sorted(unique))

    full_start = tuple(
        next(i for i, (layout, _, _) in enumerate(choices) if layout == full)
        for choices in options
    )
    active_start = tuple(
        next(
            (
                i
                for i, (layout, _, _) in enumerate(choices)
                if layout
                == _layout(
                    engine, 0 if depth is None else depth, depth_zero_digits
                )
            ),
            0,
        )
        for depth, choices in zip(depths, options, strict=True)
    )
    starts = {full_start, active_start}
    for common in ordered_layouts:
        starts.add(
            tuple(
                next(
                    (
                        i
                        for i, (layout, _, _) in enumerate(choices)
                        if layout == common
                    ),
                    fallback,
                )
                for choices, fallback in zip(options, full_start, strict=True)
            )
        )
    best = full_start
    for start in sorted(starts):
        current = start
        changed = True
        while changed:
            changed = False
            for index, choices in enumerate(options):
                trial = min(
                    (
                        current[:index] + (choice,) + current[index + 1 :]
                        for choice in range(len(choices))
                    ),
                    key=score,
                )
                if score(trial) < score(current):
                    current, changed = trial, True
        if score(current) < score(best):
            best = current
    return tuple(sorted(additions(best)))


def _declared_relinearization_depths(
    bootstrap: FullSlotBootstrap, *, input_depth: int
) -> tuple[int, ...] | None:
    """Read declared relinearization consumers; custom undeclared schedules stay unknown."""
    declare = getattr(
        bootstrap.modular_reduction, "relinearization_depths", None
    )
    return (
        cast(tuple[int, ...] | None, declare(input_depth=input_depth))
        if callable(declare)
        else None
    )


def _create_evaluation_keys(
    bootstrap: FullSlotBootstrap,
    secret_key: SecretKey,
    *,
    into: EvaluationKeySet | None = None,
    rotation_key_strategy: str = "signed_power_of_two",
    device: torch.device | str | None = None,
) -> EvaluationKeySet:
    """Add missing evaluator keys, preserving every existing object reference.

    The secret relation and modulus context of retained entries belong to the caller. This factory consumes the supplied secret only during primitive key generation; the returned ordinary inventory retains no secret or provider.
    """
    if rotation_key_strategy not in {
        "all_required",
        "signed_power_of_two",
        "signed_power_of_two_with_baby_steps",
    }:
        raise ValueError("Unknown rotation_key_strategy")
    inventory = EvaluationKeySet() if into is None else into
    selected_device = (
        secret_key.device if device is None else torch.device(device)
    )
    if selected_device.type == "cuda" and selected_device.index is None:
        selected_device = torch.device("cuda", torch.cuda.current_device())
    elif selected_device.type == "cpu":
        selected_device = torch.device("cpu")
    consumers, reduction_start, reduction_stop = _consumers(
        bootstrap, rotation_key_strategy
    )
    additions = _rotation_additions(
        bootstrap, inventory, consumers, rotation_key_strategy, selected_device
    )
    for step, (source_depth, prime_ids, digits) in additions:
        inventory.add(
            bootstrap.engine.create_rotation_key(
                step,
                secret_key,
                source_depth=source_depth,
                digit_prime_ids=digits,
                p_prime_ids=prime_ids[sum(map(len, digits)) :],
                device=selected_device,
            )
        )

    declared_depths = _declared_relinearization_depths(
        bootstrap, input_depth=reduction_start
    )
    relin_depths: tuple[int | None, ...] = (
        (None,) if declared_depths is None else declared_depths
    )
    if bootstrap.requires_relinearization and any(
        inventory.select_relinearization(
            **_geometry(bootstrap.engine, depth, selected_device)
        )
        is None
        for depth in relin_depths
    ):
        inventory.add(
            bootstrap.engine.create_relinearization_key(
                secret_key,
                source_depth=0,
                device=selected_device,
            )
        )
    conjugation_depths = (
        (reduction_start, bootstrap.output_depth)
        if bootstrap.message_domain == "real"
        else (reduction_start,)
    )
    if any(
        inventory.select_conjugation(
            **_geometry(bootstrap.engine, depth, selected_device)
        )
        is None
        for depth in conjugation_depths
    ):
        inventory.add(
            bootstrap.engine.create_conjugation_key(
                secret_key,
                source_depth=0,
                device=selected_device,
            )
        )
    return inventory
