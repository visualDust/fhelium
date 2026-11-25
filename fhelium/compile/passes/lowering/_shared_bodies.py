"""Lower structurally equal key switches once into shared Program functions."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import cast

from xdsl.dialects.builtin import UnrealizedConversionCastOp
from xdsl.dialects.func import CallOp, FuncOp, ReturnOp
from xdsl.ir import Attribute, Block, Operation, Region, SSAValue
from xdsl.rewriter import Rewriter

from fhelium.backend.ckks.rotation._galois import rotation_galois_element
from fhelium.config import CkksConfig
from fhelium.ir.dialects import ckks, core

from ._core import CkksLoweringRegistry
from ._driver import apply_lowering
from .._operation_transforms import cast_before
from ._keyswitch import galois_tables
from ._types import _cast_to_ckks, _rns_type

KEY_SWITCH_OPERATIONS = (
    ckks.RotateOp,
    ckks.ConjugateOp,
    ckks.RelinearizeOp,
    ckks.SwitchKeyOp,
)
"""Operations whose lowered body depends on the operation only through its signature and Galois element."""


def _signature(operation: Operation) -> tuple[object, ...]:
    """Return the operation facts that determine its lowered body, without the Galois element."""
    return (
        type(operation),
        tuple(
            sorted(
                (name, value)
                for name, value in operation.attributes.items()
                if name != "rotation_step"
            )
        ),
        tuple(sorted(operation.properties.items())),
        tuple(value.type for value in operation.operands),
        tuple(result.type for result in operation.results),
    )


def _galois_element(operation: Operation, config: CkksConfig) -> int | None:
    if isinstance(operation, ckks.RotateOp):
        return rotation_galois_element(
            config.N,
            int(operation.rotation_step.value.data),
            config.galois_generator,
        )
    if isinstance(operation, ckks.ConjugateOp):
        return 2 * config.N - 1
    return None


def _type_edge(operation: Operation, result_type: Attribute) -> bool:
    return (
        type(operation) is UnrealizedConversionCastOp
        and len(operation.inputs) == len(operation.outputs) == 1
        and not operation.attributes
        and operation.outputs[0].type == result_type
    )


def _bundle_signature(block: Block) -> None:
    """Type CKKS arguments and results of a lowered body as RNS bundles.

    Lowering reads every CKKS argument through a type cast and returns casts of bundle results; the function then carries the bundle types directly.
    """
    for argument in tuple(block.args):
        if not isinstance(argument.type, (ckks.CiphertextType, ckks.PlaintextType)):
            continue
        bundle = _rns_type(argument.type)
        users = tuple(use.operation for use in argument.uses)
        if not users or not all(_type_edge(user, bundle) for user in users):
            continue
        replacement = block.insert_arg(bundle, argument.index)
        for user in users:
            user.results[0].replace_all_uses_with(replacement)
            Rewriter.erase_op(user)
        block.erase_arg(argument)
    terminator = block.last_op
    assert isinstance(terminator, ReturnOp)
    values: list[SSAValue] = []
    for value in terminator.operands:
        owner = value.owner
        if (
            isinstance(owner, Operation)
            and isinstance(value.type, (ckks.CiphertextType, ckks.PlaintextType))
            and _type_edge(owner, value.type)
            and owner.inputs[0].type == _rns_type(value.type)
        ):
            value = owner.inputs[0]
        values.append(value)
    terminator.operands = values
    for operation in tuple(block.ops):
        if type(operation) is UnrealizedConversionCastOp and not any(
            result.uses for result in operation.results
        ):
            Rewriter.erase_op(operation)


def _replace_with_call(
    member: Operation, name: str, arguments: list[SSAValue], function: FuncOp
) -> None:
    inputs = function.function_type.inputs.data
    arguments = [
        value
        if value.type == expected
        else cast_before(member, value, expected, name_hint="call_argument")
        for value, expected in zip(arguments, inputs, strict=True)
    ]
    call = CallOp(name, arguments, tuple(function.function_type.outputs.data))
    operations: list[Operation] = [call]
    results: list[SSAValue | None] = []
    for old, new in zip(member.results, call.results, strict=True):
        if old.type == new.type:
            results.append(new)
            continue
        for use in tuple(old.uses):
            if _type_edge(use.operation, new.type):
                use.operation.results[0].replace_all_uses_with(new)
                Rewriter.erase_op(use.operation)
        if old.uses:
            cast, value = _cast_to_ckks(new, old.type)
            operations.append(cast)
            results.append(value)
        else:
            results.append(None)
    Rewriter.replace_op(member, operations, new_results=results)


def lower_shared_key_switches(
    compilation,
    operations: Sequence[tuple[Operation, CkksConfig]],
    registry: CkksLoweringRegistry,
) -> int:
    r"""Lower accepted key switches, sharing one function body per signature.

    Key switches with equal operation type, attributes other than ``rotation_step``, and operand and result types lower to the same expression except for the coefficient automorphism $\sigma_g$. The first operation of each group with at least two members is lowered inside a private ``func.func`` whose arguments are the operation operands followed by the gather and sign tables of $\sigma_g$. Every member becomes a ``func.call`` that passes its own Galois tables, declared at the call site. Single members are lowered in place. Returns the number of inserted operations.
    """
    groups: dict[tuple[object, ...], list[tuple[Operation, CkksConfig]]] = (
        defaultdict(list)
    )
    for operation, config in operations:
        groups[_signature(operation)].append((operation, config))
    module_block = compilation.program.module.body.block
    names = {
        function.sym_name.data for function in compilation.program.functions
    }
    inserted = 0
    for members in groups.values():
        if len(members) == 1:
            operation, config = members[0]
            inserted += len(
                apply_lowering(
                    operation, config, compilation, registry=registry
                ).operations
            )
            continue
        operation, config = members[0]
        index = len(names)
        while f"key_switch_{index}" in names:
            index += 1
        name = f"key_switch_{index}"
        names.add(name)
        block = Block(arg_types=[value.type for value in operation.operands])
        body = operation.clone(
            value_mapper=dict(zip(operation.operands, block.args, strict=True))
        )
        block.add_ops((body, ReturnOp(*body.results)))
        function = FuncOp(
            name,
            (
                tuple(value.type for value in operation.operands),
                tuple(result.type for result in operation.results),
            ),
            Region(block),
            visibility="private",
        )
        module_block.add_op(function)
        apply_lowering(body, config, compilation, registry=registry)
        tables: dict[str, core.MaterialRefOp] = {}
        for reference in tuple(block.ops):
            if not isinstance(reference, core.MaterialRefOp) or (
                reference.symbol is None
            ):
                continue
            description = compilation.program.material_description(
                reference.symbol.data
            )
            if description is not None and (
                description.get("kind") == "galois_table"
            ):
                tables[cast(str, description["name"])] = reference
        table_names = tuple(sorted(tables))
        for table in table_names:
            reference = tables[table]
            argument = block.insert_arg(reference.value.type, len(block.args))
            reference.value.replace_all_uses_with(argument)
            Rewriter.erase_op(reference)
        _bundle_signature(block)
        function.update_function_type()
        inserted += len(block.ops)
        # Members are in program order, so the tables declared before the first
        # member of a block and Galois element dominate its later members.
        declared: dict[tuple[Block, int, Attribute | None], dict] = {}
        for member, member_config in members:
            arguments = list(member.operands)
            if table_names:
                state = getattr(member.operands[0].type, "state", None)
                device = None if state is None else state.data.get("device")
                parent = member.parent_block()
                assert parent is not None
                element = cast(int, _galois_element(member, member_config))
                values = declared.get((parent, element, device))
                if values is None:
                    created, (indices, sign) = galois_tables(
                        compilation, member, member_config, element, device
                    )
                    values = {"source_indices": indices, "source_sign": sign}
                    declared[(parent, element, device)] = values
                    parent.insert_ops_before(created, member)
                    inserted += len(created)
                arguments.extend(values[table] for table in table_names)
            _replace_with_call(member, name, arguments, function)
            inserted += 1
    return inserted


__all__ = ["KEY_SWITCH_OPERATIONS", "lower_shared_key_switches"]
