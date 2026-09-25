"""Shape constraints shared by Triton region selection and concrete binding."""

from __future__ import annotations

from collections.abc import Sequence
from math import prod

from xdsl.ir import Operation

from fhelium.config.ntt import CompactRadix2Policy, resolve_ntt_backend_policy
from fhelium.ir.dialects import ckks, rns

Shape = tuple[int, ...] | None


def result_layout(
    operation_type: type[Operation],
    shapes: Sequence[Shape],
    component_axes: Sequence[bool],
    *,
    component: int | None = None,
) -> tuple[Shape, bool]:
    """Check known extents and infer an output layout; unknown shapes stay unknown."""
    first = shapes[0]
    axis = component_axes[0]
    for shape in shapes:
        if shape is not None and len(shape) < 2:
            raise ValueError(
                "Triton RNS values require prime and polynomial axes"
            )
    if operation_type is rns.ExtractComponentOp:
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
    if operation_type in (rns.PackTwoComponentsOp, rns.PackThreeComponentsOp):
        if any(component_axes):
            raise ValueError("Fusion packing requires polynomial operands")
        known = [shape for shape in shapes if shape is not None]
        if known and any(shape != known[0] for shape in known):
            raise ValueError("Fusion packing requires equal polynomial shapes")
        return (None if first is None else (len(shapes), *first)), True
    if operation_type is rns.KeySwitchDigitProductOp:
        if axis:
            raise ValueError(
                "A key-switch digit requires one polynomial bundle"
            )
        return (None if first is None else (2, *first)), True
    compact = operation_type in (
        ckks.AddCompressedPlaintextOp,
        ckks.MultiplyCompressedPlaintextOp,
    )
    plaintext = compact or operation_type in (
        rns.MultiplyPlaintextOp,
        rns.AddPlaintextOp,
    )
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


def compact_policy(policy: object) -> bool:
    """Recognize compact-radix-2 policy objects or declared Backend names."""
    if isinstance(policy, str):
        try:
            policy = resolve_ntt_backend_policy(policy)
        except ValueError:
            return False
    return isinstance(policy, CompactRadix2Policy)
