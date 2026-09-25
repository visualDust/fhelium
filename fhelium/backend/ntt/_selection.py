"""Select supported native NTT schedules from represented execution facts."""

from __future__ import annotations

from fhelium.config.ntt import (
    DEFAULT_CPU_NTT_BACKEND,
    DEFAULT_NTT_BACKEND,
    NTT_BACKEND_POLICIES,
    CompactFixedRadixPolicy,
    CompactRadix2Policy,
    IndexedRadix2Policy,
    NttBackendPolicy,
    resolve_ntt_backend_policy,
)


def algorithm_name(policy: NttBackendPolicy) -> str:
    """Identify the algorithm independently of its stage grouping."""
    if isinstance(policy, IndexedRadix2Policy):
        return "radix2_indexed"
    if isinstance(policy, CompactRadix2Policy):
        return "radix2_compact"
    return "fixed_radix"


def table_layout(policy: NttBackendPolicy) -> str:
    """Identify interchangeable native table operands, without comparing data."""
    if isinstance(policy, CompactFixedRadixPolicy):
        return f"fixed_radix{policy.radix}"
    return algorithm_name(policy)


def table_count(policy: NttBackendPolicy) -> int:
    """Count table operands after the common RNS parameter Tensor."""
    if isinstance(policy, IndexedRadix2Policy):
        return 3
    if isinstance(policy, CompactRadix2Policy):
        return 1
    return 2


def select_policy(
    *,
    selected: str | None = None,
    algorithm: str | None = None,
    group_width: int | None = None,
    radix: int | None = None,
    tables: str | None = None,
    device_type: str | None = None,
    log_n: int | None = None,
) -> NttBackendPolicy | None:
    """Apply caller constraints and a stable default, without timing candidates.

    An unspecified device alone does not select CPU or CUDA. A represented
    algorithm or table layout can determine a schedule before placement; the
    selected executor still requires its supported device at execution.
    """
    choices = (
        (resolve_ntt_backend_policy(selected),)
        if selected is not None
        else tuple(NTT_BACKEND_POLICIES.values())
    )
    compatible = []
    for policy in choices:
        if algorithm is not None and algorithm_name(policy) != algorithm:
            continue
        if tables is not None and table_layout(policy) != tables:
            continue
        if group_width is not None and (
            not isinstance(policy, CompactRadix2Policy)
            or policy.group_width != group_width
        ):
            continue
        if radix is not None and (
            not isinstance(policy, CompactFixedRadixPolicy)
            or policy.radix != radix
        ):
            continue
        if device_type == "cpu" and not isinstance(policy, IndexedRadix2Policy):
            continue
        if device_type not in (None, "cpu", "cuda"):
            continue
        if (
            log_n is not None
            and isinstance(policy, CompactFixedRadixPolicy)
            and log_n % policy.radix_bits
        ):
            continue
        compatible.append(policy)
    if not compatible:
        raise ValueError(
            "No native NTT schedule satisfies the supplied constraints"
        )
    if selected is not None or len(compatible) == 1:
        return compatible[0]
    if (
        device_type is None
        and algorithm is None
        and tables is None
        and group_width is None
        and radix is None
    ):
        return None
    default = (
        DEFAULT_CPU_NTT_BACKEND if device_type == "cpu" else DEFAULT_NTT_BACKEND
    )
    for policy in compatible:
        if policy.name == default:
            return policy
    # A partially selected fixed-radix family still needs the transform size
    # before choosing a radix that divides its number of radix-2 stages.
    if log_n is None and all(
        isinstance(p, CompactFixedRadixPolicy) for p in compatible
    ):
        return None
    return compatible[0]


def select_operation_policy(operation, *, log_n):
    """Apply NTT constraints recorded on a concrete operation's input and attributes."""
    from xdsl.dialects.builtin import IntegerAttr, StringAttr

    state = getattr(
        getattr(operation.operands[0].type, "state", None), "data", {}
    )

    def text(name):
        value = operation.attributes.get(name)
        return (
            value.data
            if isinstance(value, StringAttr) and value.data != "unknown"
            else None
        )

    def integer(name):
        value = operation.attributes.get(name)
        return int(value.value.data) if isinstance(value, IntegerAttr) else None

    device = state.get("device")
    return select_policy(
        selected=text("ntt_backend"),
        algorithm=text("ntt_algorithm"),
        group_width=integer("ntt_group_width"),
        radix=integer("ntt_radix"),
        tables=text("ntt_table_layout"),
        device_type=device.data.split(":", 1)[0]
        if isinstance(device, StringAttr) and device.data != "unknown"
        else None,
        log_n=log_n,
    )
