"""Maintain capture dependency records across repeat folding and expansion."""

from __future__ import annotations

from xdsl.dialects.builtin import IntegerAttr

from fhelium.ir.dialects import ckks, core
from ._origins import restore_origins
from ._repeat import expand_repeats, fold_repeats, repeat_bindings
from ._template_layout import template_layout
from .frontend._template_provenance import METADATA, operation_records


def _rewrite_metadata(
    compilation,
    previous,
    old_operations,
    old_values,
    replacements,
    operation_sources,
):
    by_operation = dict(
        zip(old_operations, operation_records(previous), strict=True)
    )
    _, operations, values = template_layout(compilation.program)
    identities = {value: identity for identity, value in values.items()}
    generated = {
        operation.value
        for operation, record in by_operation.items()
        if isinstance(operation, core.MaterialRefOp)
        and record.get("generated_material")
    }
    for operation in operations:
        if isinstance(operation, ckks.RepeatOp):
            for argument, value in zip(
                tuple(operation.body.block.args)[1:],
                repeat_bindings(operation),
                strict=True,
            ):
                if value in generated:
                    generated.add(argument)
    records = []
    for operation in operations:
        source = operation_sources.get(operation, operation)
        original = by_operation.get(source)
        if original is None:
            record = {
                "name": operation.name,
                "generated_attributes": [],
                "options": {},
            }
        else:
            record = dict(original)
        record["operands"] = [identities[value] for value in operation.operands]
        record["results"] = [identities[value] for value in operation.results]
        record["generated_operands"] = (
            list(original["generated_operands"])
            if original is not None
            else [
                i
                for i, value in enumerate(operation.operands)
                if value in generated
            ]
        )
        records.append(record)

    def translated(identity):
        value = old_values[identity]
        while value in replacements:
            value = replacements[value]
        return identities[value]

    metadata = {**previous, "operations": records}
    metadata.pop("facts", None)
    metadata["guards"] = [
        {**guard, "value": translated(guard["value"])}
        for guard in previous["guards"]
    ]
    metadata["output_values"] = [
        {**item, "value": translated(item["value"])}
        for item in previous.get("output_values", ())
    ]
    metadata["implicit_owners"] = {
        translated(value): translated(owner)
        for value, owner in previous.get("implicit_owners", {}).items()
    }
    compilation.workspace[METADATA] = metadata
    restore_origins(
        compilation.program, compilation.workspace, operations, records
    )
    return metadata


def fold_capture(compilation):
    previous = compilation.workspace.get(METADATA)
    if (
        not isinstance(previous, dict)
        or compilation.workspace.get("ckks_repeat") is False
    ):
        return
    _, operations, values = template_layout(compilation.program)
    observed = {
        values[item["value"]]
        for item in (*previous["guards"], *previous.get("output_values", ()))
    }
    observed.update(
        values[identity] for identity in previous.get("implicit_owners", {})
    )
    replacements = fold_repeats(compilation, observed=observed)
    from ._repeat_rotation import fold_rotation_tails

    replacements.update(fold_rotation_tails(compilation, observed=observed))
    if replacements:
        _rewrite_metadata(
            compilation, previous, operations, values, replacements, {}
        )


def expanded_template(template):
    from ._compilation import Compilation
    from ._template import StructuralTemplate
    from ._workspace import CompileWorkspace

    program = template.program.clone()
    compilation = Compilation(program, CompileWorkspace())
    _, operations, values = template_layout(program)
    sources, replacements = {}, {}
    expand_repeats(
        compilation, operation_sources=sources, replacements=replacements
    )
    metadata = _rewrite_metadata(
        compilation,
        template.metadata,
        operations,
        values,
        replacements,
        sources,
    )
    return StructuralTemplate(program, metadata)


def trim_repeat_parameters(loop, removed):
    """Remove generated parameter ports after their body consumers are trimmed."""
    shared, width, count = (
        loop.shared_count.value.data,
        loop.table_width.value.data,
        loop.count.value.data,
    )
    removed_shared = {index for index in removed if index < shared}
    removed_columns = {
        index
        for index in range(width)
        if all(
            shared + round_ * width + index in removed
            for round_ in range(count)
        )
    }
    arguments = [1 + index for index in removed_shared] + [
        1 + shared + index for index in removed_columns
    ]
    for index in sorted(arguments, reverse=True):
        loop.body.block.erase_arg(loop.body.block.args[index])
    loop.operands = tuple(
        value
        for index, value in enumerate(loop.operands)
        if index not in removed
    )
    loop.attributes["shared_count"] = IntegerAttr(
        shared - len(removed_shared), 64
    )
    loop.attributes["table_width"] = IntegerAttr(
        width - len(removed_columns), 64
    )
