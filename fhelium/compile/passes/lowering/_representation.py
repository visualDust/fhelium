"""Lower CKKS representation and depth transitions into logical operations."""

from __future__ import annotations

from typing import Literal, cast

from fractions import Fraction

from xdsl.dialects.builtin import (
    ArrayAttr,
    FloatAttr,
    IntegerAttr,
    StringAttr,
)
from xdsl.ir import Attribute, Operation

from fhelium.backend.rns.tables import RESCALE_TABLE_NAMES
from fhelium.config import CkksConfig
from fhelium.ir.ckks_state import depth_prime_ids
from fhelium.config.ntt import (
    CompactRadix2Policy,
    IndexedRadix2Policy,
    resolve_ntt_backend_policy,
)

from ....ir.dialects import ckks, core, ntt, rns
from ..._materials import (
    declare_material,
    material_symbol,
    parameter_description,
    parameter_request_identity,
)
from ._arithmetic import _parameter_descriptions, _parameters
from ._core import (
    CkksLoweringDefinition,
    LoweredCkksOperation,
)
from ._types import (
    _cast_to_ckks,
    _cast_to_rns,
    _rns_type,
)


def _ntt_table_count(policy_name: str) -> int:
    """Count numerical operands after selection, without guessing a table layout."""
    if policy_name == "unknown":
        return 1
    policy = resolve_ntt_backend_policy(policy_name)
    return (
        4
        if isinstance(policy, IndexedRadix2Policy)
        else 2
        if isinstance(policy, CompactRadix2Policy)
        else 3
    )


def _represented_string(
    operation: Operation,
    value_type: object,
    name: str,
    *,
    value: str,
) -> str:
    state = getattr(getattr(value_type, "state", None), "data", {})
    attribute = state.get(name)
    if not isinstance(attribute, StringAttr) or attribute.data == "unknown":
        raise ValueError(
            f"{operation.name} requires concrete {value} {name!r} state"
        )
    return attribute.data


def _lower_ntt(
    operation: Operation,
    config: CkksConfig | None,
    compilation,
) -> LoweredCkksOperation:
    if not isinstance(operation, (ckks.ToNttOp, ckks.FromNttOp)):
        raise TypeError("NTT lowering received another operation")
    input_cast, value = _cast_to_rns(operation.value)
    parameters = tuple(operation.parameters)
    parameter_ops: tuple[Operation, ...] = ()
    descriptions: dict[str, dict[str, object]] = {}
    if not parameters:
        state = cast(
            ckks.CiphertextType | ckks.PlaintextType, operation.value.type
        ).state.data
        depth_attr = state.get("depth")
        depth = (
            str(depth_attr.value.data)
            if isinstance(depth_attr, IntegerAttr)
            else "unknown"
        )
        basis_attr = state.get("basis")
        basis = (
            basis_attr.data if isinstance(basis_attr, StringAttr) else "unknown"
        )
        direction = (
            "inverse" if isinstance(operation, ckks.FromNttOp) else "forward"
        )
        policy = operation.attributes.get("ntt_backend")
        policy_name = (
            policy.data if isinstance(policy, StringAttr) else "unknown"
        )
        count = _ntt_table_count(policy_name)
        rns_operations, parameter = _parameters(operation, config, compilation)
        parameter_ops = rns_operations
        table_values = []
        physical = {
            name: state[name] for name in ("dtype", "device") if name in state
        }
        descriptions.update(
            _parameter_descriptions(rns_operations, operation, config)
        )
        prime_ids = state.get("prime_ids")
        ids = (
            [int(item.value.data) for item in prime_ids]
            if isinstance(prime_ids, ArrayAttr)
            and all(isinstance(item, IntegerAttr) for item in prime_ids)
            else []
        )
        for index in range(1, count):
            fields = dict(
                prime_ids=ids,
                direction=direction,
                index=index,
                ntt_backend=policy_name,
            )
            description = (
                {"kind": "ntt_table", **fields}
                if config is None
                else parameter_description(config, "ntt_table", **fields)
            )
            created, table = declare_material(
                compilation,
                operation,
                f"ntt/{basis}/{depth}/{direction}/{index}",
                core.MessageType().with_state(physical),
                description,
                identity=(
                    parameter_request_identity(description, physical)
                    if config is not None
                    else None
                ),
            )
            parameter_ops = (*parameter_ops, *created)
            table_values.append(table)
            for reference in created:
                descriptions[cast(StringAttr, reference.symbol).data] = (
                    description
                )
        parameters = (parameter, *table_values)

    value_kind = (
        "ciphertext"
        if isinstance(operation.value.type, ckks.CiphertextType)
        else "plaintext"
    )
    input_domain = _represented_string(
        operation,
        operation.value.type,
        "polynomial_domain",
        value=f"{value_kind} input",
    )
    input_residues = _represented_string(
        operation,
        operation.value.type,
        "residue_representation",
        value=f"{value_kind} input",
    )
    output_domain = _represented_string(
        operation,
        operation.result.type,
        "polynomial_domain",
        value="result",
    )
    output_residues = _represented_string(
        operation,
        operation.result.type,
        "residue_representation",
        value="result",
    )
    if isinstance(operation, ckks.ToNttOp):
        if input_domain != "coefficient" or input_residues not in {
            "standard",
            "montgomery",
        }:
            raise ValueError(
                f"{operation.name} requires coefficient-domain standard or "
                "Montgomery input"
            )
        if (output_domain, output_residues) != ("ntt", "montgomery"):
            raise ValueError(
                f"{operation.name} requires NTT/Montgomery result state"
            )
        operation_type: type[Operation] = (
            ntt.CoefficientStandardToNttMontgomeryOp
            if input_residues == "standard"
            else ntt.CoefficientMontgomeryToNttMontgomeryOp
        )
    else:
        if (input_domain, input_residues) != ("ntt", "montgomery"):
            raise ValueError(
                f"{operation.name} requires NTT/Montgomery input state"
            )
        if output_domain != "coefficient" or output_residues not in {
            "standard",
            "montgomery",
        }:
            raise ValueError(
                f"{operation.name} requires coefficient-domain standard or "
                "Montgomery result"
            )
        operation_type = (
            ntt.NttMontgomeryToCoefficientStandardOp
            if output_residues == "standard"
            else ntt.NttMontgomeryToCoefficientMontgomeryOp
        )
    logical = operation_type.create(
        operands=(value, *parameters),
        result_types=(_rns_type(operation.result.type),),
        attributes={
            name: value
            for name, value in operation.attributes.items()
            if name.startswith("ntt_") or name == "ckks_config"
        },
    )
    result_cast, result = _cast_to_ckks(
        logical.results[0], operation.result.type
    )
    return LoweredCkksOperation(
        (input_cast, *parameter_ops, logical, result_cast),
        result,
        descriptions,
    )


def _rescale_materials(
    operation: Operation,
    config: CkksConfig,
    path: str,
    source_ids: tuple[int, ...],
    target_ids: tuple[int, ...],
    rounding: str,
    domain: str,
    attributes: dict[str, Attribute],
) -> tuple[tuple[core.MaterialRefOp, ...], dict[str, dict[str, object]]]:
    """Reference the tables of one rescale from ``source_ids`` to ``target_ids``.

    ``attributes`` receives the ``parameter_names`` of the referenced tables."""

    names = [
        *RESCALE_TABLE_NAMES,
        *(
            ("target_keep_positions", "restore")
            if not set(target_ids) <= set(source_ids)
            else ()
        ),
    ]
    policy = attributes.get("ntt_backend")
    policy_name = policy.data if isinstance(policy, StringAttr) else "unknown"
    if domain == "ntt":
        count = _ntt_table_count(policy_name)
        names.extend(
            f"{part}_{direction}_{i}"
            for part, direction in (("dropped", "inverse"), ("target", "forward"))
            for i in range(count)
        )
        names.append("product_inverse")
    refs = tuple(
        core.MaterialRefOp(
            core.MessageType(),
            symbol=material_symbol(operation, f"{path}/{name}"),
        )
        for name in names
    )
    descriptions = {
        cast(StringAttr, reference.symbol).data: parameter_description(
            config,
            "rescale_table",
            source_prime_ids=list(source_ids),
            target_prime_ids=list(target_ids),
            rounding=rounding,
            name=name,
            input_domain=domain,
            ntt_backend=policy_name,
        )
        for name, reference in zip(names, refs, strict=True)
    }
    attributes["parameter_names"] = ArrayAttr(StringAttr(name) for name in names)
    return refs, descriptions


def _lower_rescale(
    operation: Operation,
    config: CkksConfig | None,
    compilation,
) -> LoweredCkksOperation:
    if not isinstance(operation, ckks.RescaleOp):
        raise TypeError("rescale lowering received another operation")
    if config is None:
        raise ValueError(
            "Rescale lowering requires CKKS depth-group parameters"
        )
    input_cast, value = _cast_to_rns(operation.value)
    rounding = operation.rounding or StringAttr("nearest")
    source_rns_type = _rns_type(operation.value.type)
    depth_attr = source_rns_type.state.data.get("depth")
    if not isinstance(depth_attr, IntegerAttr):
        raise ValueError("rescale lowering requires concrete ciphertext depth")
    depth = int(depth_attr.value.data)
    basis_attr = source_rns_type.state.data.get("basis")
    basis = basis_attr.data if isinstance(basis_attr, StringAttr) else "Q"
    source_ids = depth_prime_ids(config, depth, basis)
    target_ids = depth_prime_ids(config, depth + 1, basis)
    parameters = tuple(operation.parameters)
    parameter_ops: tuple[Operation, ...] = ()
    descriptions: dict[str, dict[str, object]] = {}
    attributes = {
        name: value
        for name, value in operation.attributes.items()
        if name in {"parameter_names", "ntt_backend"}
    }
    if not parameters:
        parameter_ops, descriptions = _rescale_materials(
            operation,
            config,
            f"rescale/{basis}/{depth}/{rounding.data}",
            source_ids,
            target_ids,
            rounding.data,
            operation.input_domain.data,
            attributes,
        )
        parameters = tuple(ref.value for ref in parameter_ops)
    logical = rns.RescaleOp(
        value,
        parameters,
        _rns_type(operation.result.type),
        attributes=attributes,
        rounding=rounding,
        polynomial_domain=cast(
            Literal["coefficient", "ntt"], operation.input_domain.data
        ),
    )
    result_cast, result = _cast_to_ckks(logical.result, operation.result.type)
    return LoweredCkksOperation(
        (*parameter_ops, input_cast, logical, result_cast),
        result,
        descriptions,
    )


def _lower_residue_conversion(
    operation: Operation,
    config: CkksConfig | None,
    compilation,
) -> LoweredCkksOperation:
    if not isinstance(
        operation,
        (ckks.ToMontgomeryResiduesOp, ckks.ToStandardResiduesOp),
    ):
        raise TypeError(
            "Residue conversion lowering received another operation"
        )
    input_cast, value = _cast_to_rns(operation.value)
    parameter_ops, parameters = _parameters(operation, config, compilation)
    operation_type: type[Operation] = (
        rns.StandardToMontgomeryOp
        if isinstance(operation, ckks.ToMontgomeryResiduesOp)
        else rns.MontgomeryToStandardOp
    )
    logical = operation_type.create(
        operands=(value, parameters),
        result_types=(_rns_type(operation.result.type),),
    )
    result_cast, result = _cast_to_ckks(
        logical.results[0], operation.result.type
    )
    return LoweredCkksOperation(
        (input_cast, *parameter_ops, logical, result_cast),
        result,
        _parameter_descriptions(parameter_ops, operation, config),
    )


def _lower_mod_switch(
    operation: Operation,
    config: CkksConfig | None,
    compilation,
) -> LoweredCkksOperation:
    if not isinstance(operation, ckks.ModSwitchOp):
        raise TypeError("Modulus-switch lowering received another operation")
    state = operation.value.type.state.data
    depth_attr = state.get("depth")
    target_depth = int(operation.target_depth.value.data)
    if (
        config is not None
        and isinstance(depth_attr, IntegerAttr)
        and not config.is_subbasis(int(depth_attr.value.data), target_depth)
    ):
        return _lower_level_reduce(operation, config, compilation)
    input_cast, value = _cast_to_rns(operation.value)
    logical = rns.RestrictDepthOp(
        value,
        _rns_type(operation.result.type),
        target_depth=operation.target_depth,
    )
    result_cast, result = _cast_to_ckks(
        logical.results[0], operation.result.type
    )
    return LoweredCkksOperation(
        (input_cast, logical, result_cast),
        result,
    )


def _lower_level_reduce(
    operation: ckks.ModSwitchOp,
    config: CkksConfig,
    compilation,
) -> LoweredCkksOperation:
    r"""Lower a move to a non-subset basis as $c\cdot x$ followed by one rescale.

    With $\rho=D/R$ the ratio from the source basis to the target basis,
    $c=\operatorname{round}(\Delta_t\rho/\Delta)$ for the input actual scale
    $\Delta$, matching ``Engine.mod_switch_to_depth``.
    """

    state = operation.value.type.state.data
    source_depth = int(cast(IntegerAttr, state["depth"]).value.data)
    target_depth = int(operation.target_depth.value.data)
    scale_attr = state.get("scale")
    if not isinstance(scale_attr, FloatAttr):
        raise ValueError(
            "Moving to a non-subset basis requires a known input scale"
        )
    multiplier = round(
        Fraction(config.scale_at(target_depth))
        * config.transition_ratio(source_depth, target_depth)
        / Fraction(float(scale_attr.value.data))
    )
    scaled = ckks.MultiplyIntegerScalarOp(
        operation.value, scalar=IntegerAttr(multiplier, 64)
    )
    basis_attr = state.get("basis")
    basis = basis_attr.data if isinstance(basis_attr, StringAttr) else "Q"
    source_ids = depth_prime_ids(config, source_depth, basis)
    target_ids = depth_prime_ids(config, target_depth, basis)
    domain_attr = state.get("polynomial_domain")
    domain = (
        domain_attr.data if isinstance(domain_attr, StringAttr) else "coefficient"
    )
    attributes: dict[str, Attribute] = {
        name: value
        for name, value in operation.attributes.items()
        if name == "ntt_backend"
    }
    refs, descriptions = _rescale_materials(
        operation,
        config,
        f"level-reduce/{basis}/{source_depth}/{target_depth}",
        source_ids,
        target_ids,
        "nearest",
        domain,
        attributes,
    )
    input_cast, value = _cast_to_rns(scaled.result)
    logical = rns.RescaleOp(
        value,
        tuple(ref.value for ref in refs),
        _rns_type(operation.result.type),
        attributes=attributes,
        rounding="nearest",
        polynomial_domain=cast(Literal["coefficient", "ntt"], domain),
    )
    result_cast, result = _cast_to_ckks(logical.result, operation.result.type)
    return LoweredCkksOperation(
        (scaled, *refs, input_cast, logical, result_cast),
        result,
        descriptions,
    )


def _lower_reinterpret_scale(
    operation: Operation,
    config: CkksConfig | None,
    compilation,
) -> LoweredCkksOperation:
    del config
    if not isinstance(operation, ckks.ReinterpretScaleOp):
        raise TypeError(
            "Scale reinterpretation lowering received another operation"
        )
    input_cast, value = _cast_to_rns(operation.value)
    logical = rns.ReinterpretScaleOp(
        value,
        _rns_type(operation.result.type),
        scale=operation.scale,
    )
    result_cast, result = _cast_to_ckks(
        logical.results[0], operation.result.type
    )
    return LoweredCkksOperation((input_cast, logical, result_cast), result)


REPRESENTATION_LOWERINGS = (
    CkksLoweringDefinition(
        "rns-ntt-transition",
        ckks.ToNttOp,
        _lower_ntt,
        is_default=True,
    ),
    CkksLoweringDefinition(
        "rns-ntt-transition",
        ckks.FromNttOp,
        _lower_ntt,
        is_default=True,
    ),
    CkksLoweringDefinition(
        "rns-drop-leading-prime",
        ckks.RescaleOp,
        _lower_rescale,
        is_default=True,
    ),
    CkksLoweringDefinition(
        "rns-residue-conversion",
        ckks.ToMontgomeryResiduesOp,
        _lower_residue_conversion,
        is_default=True,
    ),
    CkksLoweringDefinition(
        "rns-residue-conversion",
        ckks.ToStandardResiduesOp,
        _lower_residue_conversion,
        is_default=True,
    ),
    CkksLoweringDefinition(
        "rns-restrict-depth",
        ckks.ModSwitchOp,
        _lower_mod_switch,
        is_default=True,
    ),
    CkksLoweringDefinition(
        "rns-reinterpret-scale",
        ckks.ReinterpretScaleOp,
        _lower_reinterpret_scale,
        is_default=True,
    ),
)
