"""Lower CKKS representation and depth transitions into logical operations."""

from __future__ import annotations

from typing import Literal, cast

from xdsl.dialects.builtin import (
    ArrayAttr,
    IntegerAttr,
    StringAttr,
)
from xdsl.ir import Operation

from fhelium.config import CkksConfig
from fhelium.config.ntt import (
    CompactRadix2Policy,
    IndexedRadix2Policy,
    resolve_ntt_backend_policy,
)

from ....ir.dialects import ckks, core, ntt, rns
from ..._materials import material_symbol, parameter_description
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
        refs = tuple(
            core.MaterialRefOp(
                core.MessageType(),
                symbol=material_symbol(
                    operation, f"ntt/{basis}/{depth}/{direction}/{i}"
                ),
            )
            for i in range(1, count)
        )
        parameter_ops = (*rns_operations, *refs)
        parameters = (parameter, *(ref.value for ref in refs))
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
        for index, reference in enumerate(refs, 1):
            fields = dict(
                prime_ids=ids,
                direction=direction,
                index=index,
                ntt_backend=policy_name,
            )
            descriptions[cast(StringAttr, reference.symbol).data] = (
                {"kind": "ntt_table", **fields}
                if config is None
                else parameter_description(config, "ntt_table", **fields)
            )

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
    drop_count = len(config.q_depth_groups[depth])
    parameters = tuple(operation.parameters)
    parameter_ops: tuple[Operation, ...] = ()
    descriptions: dict[str, dict[str, object]] = {}
    attributes = {
        name: value
        for name, value in operation.attributes.items()
        if name in {"parameter_names", "half_primes", "ntt_backend"}
    }
    if not parameters:
        names = [
            name
            for i in range(drop_count)
            for name in (f"step{i}_parameters", f"step{i}_inverse")
        ]
        if operation.input_domain.data == "ntt":
            policy = attributes.get("ntt_backend", StringAttr("unknown"))
            assert isinstance(policy, StringAttr)
            count = _ntt_table_count(policy.data)
            names.extend(
                f"{basis}_{direction}_{i}"
                for basis, direction in (
                    ("dropped", "inverse"),
                    ("remaining", "forward"),
                )
                for i in range(count)
            )
            names.append("prefix_inverse")
        basis_attr = source_rns_type.state.data.get("basis")
        basis = (
            basis_attr.data if isinstance(basis_attr, StringAttr) else "unknown"
        )
        refs = tuple(
            core.MaterialRefOp(
                core.MessageType(),
                symbol=material_symbol(
                    operation, f"rescale/{basis}/{depth}/{name}"
                ),
            )
            for name in names
        )
        parameter_ops = refs
        parameters = tuple(ref.value for ref in refs)
        for name, reference in zip(names, refs, strict=True):
            policy = attributes.get("ntt_backend")
            descriptions[cast(StringAttr, reference.symbol).data] = (
                parameter_description(
                    config,
                    "rescale_table",
                    depth=depth,
                    basis=basis,
                    name=name,
                    input_domain=operation.input_domain.data,
                    ntt_backend=policy.data
                    if isinstance(policy, StringAttr)
                    else "unknown",
                )
            )

        attributes["parameter_names"] = ArrayAttr(
            StringAttr(name) for name in names
        )
        attributes["half_primes"] = ArrayAttr(
            IntegerAttr(prime // 2, 64)
            for prime in config.q_depth_groups[depth]
        )
    logical = rns.RescaleDropLeadingPrimesOp(
        value,
        parameters,
        _rns_type(operation.result.type),
        drop_count=drop_count,
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
