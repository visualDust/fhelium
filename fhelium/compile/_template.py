"""Instantiate captured mathematical Programs without executing Python source.

A template keeps the concrete captured Program and a JSON sidecar of state dependencies, material origins, and metadata-read guards. Instance Programs carry ordinary value types and Tensor material bindings.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from xdsl.dialects.builtin import (
    DenseArrayBase,
    DictionaryAttr,
    StringAttr,
    i32,
)
from xdsl.rewriter import Rewriter
from xdsl.ir import Attribute

from fhelium.ir import Program
from fhelium.ir.dialects import ckks, core
from fhelium.ir.dialects._common import OpenStateType
from ._compilation import Compilation
from ._workspace import CompileWorkspace
from ._template_rules import (
    SEMANTIC_RULES,
    rule_name,
    leaf_compatible,
    read_metadata,
    result_types,
    state_values,
    supported,
)
from .frontend._eager_values import state_attributes, value_type
from .frontend._template_provenance import (
    METADATA,
    json_value,
    operation_records,
)


def _values(program, records):
    from ._template_layout import template_layout

    return template_layout(program)


def _current_type(sample, value):
    if isinstance(value, OpenStateType):
        return value
    if isinstance(value, Mapping):
        from fhelium.ir.dialects._common import state_attribute

        return type(sample)().with_state(
            {
                name: item
                if isinstance(item, Attribute)
                else state_attribute(item)
                for name, item in value.items()
            }
        )
    return value_type(value)


def _attribute_facts(operation, record):
    options = record["options"]
    dependent = {}
    if options.get("depth_offset") is not None:
        dependent["target_depth"] = "depth_offset"
    if options.get("scalar_scale_from_input"):
        dependent["scalar_scale"] = "input_scale"
    if options.get("view") == "append_unit_axis":
        dependent["fhelium.call.arguments"] = "append_unit_axis"
    return {
        name: {
            "kind": "Derived",
            "rule": dependent[name],
            "operands": record["operands"],
            "options": options,
        }
        if name in dependent
        else {"kind": "Known", "value": str(attribute)}
        for name, attribute in operation.attributes.items()
        if name not in record["generated_attributes"]
    }


def _facts(program, records):
    """Describe field facts using one origin and intrinsic table per operation.

    A field's integer ``origin`` indexes ``operations``; a string origin names an ``inputs`` entry. Operation origins refer to their intrinsic table by operation index. Requirements likewise reference the operation containing their sources and intrinsic options.
    """
    _, operations, values = _values(program, records)
    operation_facts = []
    requirements = []
    for index, (operation, record) in enumerate(
        zip(operations, records, strict=True)
    ):
        if isinstance(operation, core.MaterialRefOp):
            origin = (
                {
                    "kind": "Derived",
                    "rule": "preparation_result",
                    "symbol": operation.symbol.data,
                }
                if record["options"].get("preparation_output")
                else {
                    "kind": "Derived",
                    "rule": "numerical_material",
                    "symbol": operation.symbol.data,
                }
                if record.get("generated_material")
                else {"kind": "InputField", "leaf": operation.symbol.data}
            )
        elif isinstance(operation, core.ConstantOp):
            origin = {"kind": "Known"}
        else:
            origin = {
                "kind": "Derived",
                "rule": rule_name(operation, record),
                "operands": record["operands"],
                "options": record["options"],
                "intrinsic_attributes": {"operation": index},
            }
        entry = {
            "origin": origin,
            "intrinsics": _attribute_facts(operation, record),
        }
        if record["generated_operands"]:
            entry["requirement"] = {
                "rule": rule_name(operation, record),
                "sources": [
                    value
                    for i, value in enumerate(record["operands"])
                    if i not in record["generated_operands"]
                ],
            }
            requirements.append({"operation": index})
        operation_facts.append(entry)
    inputs = {}
    facts = {}
    for identity, value in values.items():
        fields = getattr(value.type, "state", DictionaryAttr({})).data
        if identity.startswith("arg:"):
            reference = identity
            origin = {"kind": "InputField", "leaf": identity}
            inputs[identity] = origin
        else:
            reference = int(identity.split(":")[1])
            origin = operation_facts[reference]["origin"]
        facts[identity] = {
            name: {"kind": "Unknown"}
            if attribute == StringAttr("unknown")
            else {"kind": "Known", "origin": reference, "value": str(attribute)}
            if origin["kind"] == "Known"
            else {"kind": origin["kind"], "origin": reference, "field": name}
            for name, attribute in fields.items()
        }
        for name in ("shape", "strides"):
            facts[identity].setdefault(name, {"kind": "Unknown"})
    return {
        "inputs": inputs,
        "operations": operation_facts,
        "values": facts,
        "requirements": requirements,
    }


@dataclass(frozen=True)
class StructuralTemplate:
    """Store one capture's mathematical structure and serializable dependencies."""

    program: Program
    metadata: dict[str, object]

    def instantiate(
        self,
        arguments: Mapping[str, object],
        *,
        material_bindings: Mapping[str, torch.Tensor] | None = None,
        leaf_states: Mapping[str, object] | None = None,
        workspace: CompileWorkspace | None = None,
    ) -> Compilation | None:
        """Build a fresh concrete instance, or return None on a structural guard miss.

        External CKKS leaves use current envelopes or types in ``leaf_states``. Raw key Tensors may be rebound against the template's fixed declared key layout. Numerical tables are declared by ordinary lowering and implementation requirements using this instance's states and current workspace providers.
        """
        if (
            workspace is not None
            and workspace.get("ckks_repeat") is False
            and any(
                isinstance(operation, ckks.RepeatOp)
                for operation in self.program.walk()
            )
        ):
            from ._repeat_template import expanded_template

            return expanded_template(self).instantiate(
                arguments,
                material_bindings=material_bindings,
                leaf_states=leaf_states,
                workspace=workspace,
            )
        metadata = self.metadata
        for name, expected in metadata.get("static_arguments", {}).items():
            if (
                name not in arguments
                or type(arguments[name]).__name__
                != metadata["static_argument_types"][name]
                or json_value(arguments[name]) != expected
            ):
                return None
        program = self.program.clone()
        records = metadata["operations"]
        function, operations, values = _values(program, records)
        bindings = dict(material_bindings or {})
        leaves = leaf_states or {}
        work = (
            CompileWorkspace()
            if workspace is None
            else CompileWorkspace(workspace)
        )
        guards = {}
        for guard in metadata["guards"]:
            guards.setdefault(guard["value"], []).append(guard)

        def guard_matches(identity):
            return all(
                json_value(read_metadata(values[identity], guard["field"]))
                == guard["expected"]
                for guard in guards.get(identity, ())
            )

        def compatible(identity, sample, typ):
            owner_id = metadata.get("implicit_owners", {}).get(identity)
            if owner_id is None:
                return leaf_compatible(sample, typ)
            owner = state_values(values[owner_id].type)
            state = state_values(typ)
            return (
                type(sample) is type(typ)
                and state.get("shape")
                == (*owner["batch_shape"], len(owner["prime_ids"]))
                and all(
                    sample.state.data.get(field) == typ.state.data.get(field)
                    for field in ("dtype", "device")
                )
            )

        input_fields = json.loads(
            program.module.attributes["fhelium.input_fields"].data
        )
        names = [
            name.data
            for name in program.module.attributes["fhelium.input_names"]
        ]
        for name, argument in zip(
            names, tuple(function.body.block.args), strict=True
        ):
            path = input_fields.get(name, [name])
            if path[0] not in arguments:
                return None
            current = arguments[path[0]]
            for field in path[1:]:
                current = getattr(current, field)
            typ = _current_type(argument.type, current)
            identity = f"arg:{name}"
            if not compatible(identity, argument.type, typ):
                return None
            values[identity] = Rewriter.replace_value_with_new_type(
                argument, typ
            )
            if not guard_matches(identity):
                return None
        external_symbols = set()
        for operation, record in zip(operations, records, strict=True):
            if not isinstance(operation, core.MaterialRefOp) or record.get(
                "generated_material"
            ):
                continue
            symbol = operation.symbol.data
            external_symbols.add(symbol)
            sample = operation.value.type
            if record["options"].get("preparation_output"):
                from ._binding_time import PREPARATION

                plan = work.get(PREPARATION)
                if plan is None or symbol not in plan.bindings:
                    return None
                typ = sample.with_state(state_attributes(plan.bindings[symbol]))
            elif symbol in leaves:
                typ = _current_type(sample, leaves[symbol])
            elif symbol in bindings:
                if isinstance(
                    sample,
                    (
                        ckks.CiphertextType,
                        ckks.PlaintextType,
                        ckks.CompressedPlaintextType,
                    ),
                ):
                    return None
                typ = sample.with_state(state_attributes(bindings[symbol]))
            elif isinstance(sample, ckks.EvaluationKeyType):
                # An unbound declared key remains available for ordinary linking.
                typ = sample
            else:
                return None
            identity = record["results"][0]
            if not compatible(identity, sample, typ):
                return None
            values[identity] = Rewriter.replace_value_with_new_type(
                operation.value, typ
            )
            if not guard_matches(identity):
                return None
        # Strip only operands and attributes marked as generated at capture production sites.
        for operation, record in reversed(
            tuple(zip(operations, records, strict=True))
        ):
            removed = set(record["generated_operands"])
            if removed and isinstance(operation, ckks.RepeatOp):
                from ._repeat_template import trim_repeat_parameters

                trim_repeat_parameters(operation, removed)
                removed = set()
            if removed:
                operation.operands = tuple(
                    value
                    for i, value in enumerate(operation.operands)
                    if i not in removed
                )
                segments = operation.properties.get("operandSegmentSizes")
                if isinstance(segments, DenseArrayBase):
                    start = 0
                    sizes = []
                    for size in segments.iter_values():
                        sizes.append(
                            size
                            - sum(
                                i in removed for i in range(start, start + size)
                            )
                        )
                        start += size
                    operation.properties["operandSegmentSizes"] = (
                        DenseArrayBase.from_list(i32, sizes)
                    )
            for name in record["generated_attributes"]:
                operation.attributes.pop(name, None)
        for operation, record in zip(operations, records, strict=True):
            if record.get("generated_material"):
                Rewriter.erase_op(operation)
        records_by_operation = dict(zip(operations, records, strict=True))
        identities = {value: identity for identity, value in values.items()}
        from ._repeat import (
            sync_repeat_inputs,
            sync_repeat_results,
            sync_iteration,
        )

        class UnrollRequired(Exception):
            pass

        def refresh_values(old_values, new_values):
            for old, new in zip(old_values, new_values, strict=True):
                identity = identities.get(old)
                if identity is not None:
                    values[identity] = new
                    identities[new] = identity

        def instantiate_block(block):
            for operation in tuple(block.ops):
                record = records_by_operation[operation]
                if isinstance(operation, ckks.RepeatOp):
                    old_args = tuple(operation.body.block.args)
                    if not sync_repeat_inputs(operation):
                        raise UnrollRequired
                    refresh_values(old_args, tuple(operation.body.block.args))
                    if not instantiate_block(operation.body.block):
                        return False
                    old_results = tuple(operation.results)
                    if not sync_repeat_results(operation):
                        raise UnrollRequired
                    refresh_values(old_results, tuple(operation.results))
                elif isinstance(operation, ckks.RepeatIterationOp):
                    old_results = tuple(operation.results)
                    sync_iteration(operation)
                    refresh_values(old_results, tuple(operation.results))
                elif not isinstance(operation, ckks.RepeatYieldOp):
                    for identity, value, typ in zip(
                        record["results"],
                        tuple(operation.results),
                        result_types(operation, record, work),
                        strict=True,
                    ):
                        values[identity] = Rewriter.replace_value_with_new_type(
                            value, typ
                        )
                        identities[values[identity]] = identity
                for identity in record["results"]:
                    if not guard_matches(identity):
                        return False
            return True

        try:
            if not instantiate_block(function.body.block):
                return None
        except UnrollRequired:
            from ._repeat_template import expanded_template

            return expanded_template(self).instantiate(
                arguments,
                material_bindings=bindings,
                leaf_states=leaves,
                workspace=work,
            )
        descriptor = json.loads(
            program.module.attributes["fhelium.output_structure"].data
        )

        def update_descriptor(node):
            if isinstance(node, dict):
                for output in metadata.get("output_values", ()):
                    if node == output["descriptor"]:
                        value = values[output["value"]]
                        node["fields"] = {
                            name: json_value(read_metadata(value, name))
                            for name in node["fields"]
                        }
                        break
                for child in node.values():
                    update_descriptor(child)
            elif isinstance(node, list):
                for child in node:
                    update_descriptor(child)

        update_descriptor(descriptor)
        program.module.attributes["fhelium.output_structure"] = StringAttr(
            json.dumps(descriptor)
        )
        descriptions = {
            symbol: description
            for symbol, description in program.material_descriptions.items()
            if symbol in external_symbols
        }
        program.module.attributes.pop("fhelium.material_descriptions", None)
        for symbol, description in descriptions.items():
            program.set_material_description(symbol, description)
        function.update_function_type()
        work.pop(METADATA, None)
        work["structural_template_instance"] = True
        work["structural_template_stage"] = metadata["stage"]
        compilation = Compilation(
            program,
            work,
            material_bindings={
                symbol: tensor
                for symbol, tensor in bindings.items()
                if symbol in external_symbols
            },
        )
        from ._origins import restore_origins

        restore_origins(program, work, operations, records)
        # Whole implementations declare current key-switch/compressed operands here;
        # logical lowerings declare their own tables later in the selected pipeline.
        from fhelium.backend.implementation import (
            OperationImplementationRegistry,
        )

        if OperationImplementationRegistry in work:
            from ._material_requirements import resolve_material_requirements

            resolve_material_requirements(compilation, tuple(program.walk()))
        program.module.verify()
        return compilation


def build_template(source: Compilation) -> StructuralTemplate | None:
    """Freeze an eligible untransformed capture with all sample facts retained."""
    metadata = source.workspace.get(METADATA)
    if (
        not isinstance(metadata, dict)
        or source.reports
        or len(source.program.functions) != 1
    ):
        return None
    records = operation_records(metadata)
    from ._template_layout import template_layout

    _, operations, _ = template_layout(source.program)
    if len(records) != len(operations):
        return None
    if any(
        operation.name != record["name"] or not supported(operation, record)
        for operation, record in zip(operations, records, strict=True)
    ):
        return None
    metadata = copy.deepcopy(metadata)
    metadata["facts"] = _facts(source.program, records)
    metadata["stage"] = (
        "captured-semantic"
        if any(type(operation) in SEMANTIC_RULES for operation in operations)
        else "captured-ckks"
    )
    return StructuralTemplate(source.program.clone(), metadata)


__all__ = ["StructuralTemplate", "build_template"]
