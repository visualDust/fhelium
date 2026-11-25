"""Shape constraints shared by Triton region selection and concrete binding."""

from __future__ import annotations

from collections.abc import Sequence
from math import prod

from xdsl.dialects.builtin import ArrayAttr, IntegerAttr

from fhelium.ir._operation_structure import (
    OperationStructure,
    operation_structure,
)
from fhelium.ir.dialects import ckks, rns

Shape = tuple[int, ...] | None

GENERATED_LAYOUTS = frozenset(
    ("same", "select", "pack", "key_product", "component_product", "transform")
)
"""Result layouts whose access the Triton code generator emits. ``permute`` and ``reduce_rows`` operations are region boundaries."""


def generated_structure(operation) -> OperationStructure | None:
    """Return the registered structure of ``operation`` when its layout is in :data:`GENERATED_LAYOUTS`."""
    structure = operation_structure(operation)
    if structure is None or structure.layout not in GENERATED_LAYOUTS:
        return None
    return structure


def polynomial_shape(value_type: object) -> ArrayAttr[IntegerAttr] | None:
    """Recover polynomial storage extents from represented value state.

    Ciphertexts and RNS bundles store ``[component?, *batch, prime, coefficient]``.
    Prime identities determine the row extent; component_count (or components)
    determines whether storage has a component axis. Missing batch or ring
    facts leave the shape unresolved.
    """
    if not isinstance(value_type, (ckks.CiphertextType, rns.RnsBundleType)):
        return None
    state = value_type.state.data
    batch, primes = state.get("batch_shape"), state.get("prime_ids")
    ring = state.get("ring_dimension")
    components = state.get("component_count", state.get("components"))
    if not (
        isinstance(batch, ArrayAttr)
        and isinstance(primes, ArrayAttr)
        and isinstance(ring, IntegerAttr)
    ):
        return None
    if isinstance(value_type, ckks.CiphertextType) and not isinstance(
        components, IntegerAttr
    ):
        return None
    return ArrayAttr(
        (
            *((components,) if isinstance(components, IntegerAttr) else ()),
            *batch,
            IntegerAttr(len(primes), 64),
            ring,
        )
    )


def result_layout(
    structure: OperationStructure,
    shapes: Sequence[Shape],
    component_axes: Sequence[bool],
) -> tuple[Shape, bool]:
    """Check known extents and infer an output layout; unknown shapes stay unknown.

    ``shapes`` and ``component_axes`` describe the operands with ``data`` or ``compact`` roles, in operand order.
    """
    reduction = structure.reduction
    if reduction is not None and reduction.stacked is not None:
        # The leading term axis of a stacked operand is reduced away.
        position = structure.operands("data", "compact").index(reduction.stacked)
        stacked = shapes[position]
        shapes = list(shapes)
        shapes[position] = None if stacked is None else stacked[1:]
    first = shapes[0]
    axis = component_axes[0]
    for shape in shapes:
        if shape is not None and len(shape) < 2:
            raise ValueError(
                "Triton RNS values require prime and polynomial axes"
            )
    if structure.layout == "select":
        component = structure.component
        if not axis:
            raise ValueError(
                "Fusion component extraction requires a component axis"
            )
        if (
            component is None
            or component < 0
            or first is not None
            and component >= first[0]
        ):
            raise ValueError(
                "Fusion component extraction is outside its source extent"
            )
        return (None if first is None else first[1:]), False
    if structure.layout == "pack":
        if any(component_axes):
            raise ValueError("Fusion packing requires polynomial operands")
        known = [shape for shape in shapes if shape is not None]
        if known and any(shape != known[0] for shape in known):
            raise ValueError("Fusion packing requires equal polynomial shapes")
        return (None if first is None else (len(shapes), *first)), True
    if structure.layout == "key_product":
        if axis:
            raise ValueError(
                "A key-switch digit requires one polynomial bundle"
            )
        return (None if first is None else (2, *first)), True
    if structure.layout == "component_product":
        if not all(component_axes) or any(
            shape is not None and shape[0] != 2 for shape in shapes
        ):
            raise ValueError("A tensor product requires two-component operands")
        known = [shape for shape in shapes if shape is not None]
        if known and any(shape != known[0] for shape in known):
            raise ValueError("A tensor product requires equal operand shapes")
        return (None if first is None else (3, *first[1:])), True
    compact = "compact" in structure.roles
    plaintext = structure.plaintext
    for shape, rhs_axis in zip(shapes[1:], component_axes[1:], strict=True):
        if plaintext and rhs_axis:
            raise ValueError("Prepared plaintext cannot carry a component axis")
        if first is None or shape is None:
            continue
        if (
            shape[-2] != first[-2]
            or (not compact and shape[-1] != first[-1])
            or (compact and (shape[-1] <= 0 or first[-1] % shape[-1]))
        ):
            raise ValueError(
                "Fusion arithmetic requires matching prime and polynomial extents"
            )
        lhs_batch = first[1:-2] if axis else first[:-2]
        rhs_batch = shape[1:-2] if rhs_axis else shape[:-2]
        if prod(rhs_batch) not in (1, prod(lhs_batch)):
            raise ValueError(
                "Fusion arithmetic requires equal batches or one broadcast polynomial"
            )
        lhs_components = first[0] if axis else 1
        rhs_components = shape[0] if rhs_axis else 1
        if not plaintext and rhs_components not in (1, lhs_components):
            raise ValueError("Fusion arithmetic component extents differ")
    return first, axis


def check_ntt_extent(n: int | None) -> None:
    if n is not None and (n < 256 or n & (n - 1)):
        raise ValueError(
            "Compact NTT fusion requires a power-of-two extent >= 256"
        )


def check_ntt_layouts(layouts: Sequence[tuple[Shape, bool]]) -> None:
    """Require common known transform/output batches and a supported polynomial extent."""
    known = [(shape, axis) for shape, axis in layouts if shape is not None]
    if not known:
        return
    first, first_axis = known[0]
    assert first is not None
    batch = first[1:-2] if first_axis else first[:-2]
    for shape, axis in known:
        assert shape is not None
        if (
            shape[-2:] != first[-2:]
            or (shape[1:-2] if axis else shape[:-2]) != batch
        ):
            raise ValueError(
                "One NTT fusion region requires a common batch shape and prime extent"
            )
        check_ntt_extent(shape[-1])
