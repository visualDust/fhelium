"""Identify captured values in a Program, including repeat body arguments."""


def template_layout(program):
    function = program.functions[0]
    names = [
        name.data for name in program.module.attributes["fhelium.input_names"]
    ]
    values = {
        f"arg:{name}": value
        for name, value in zip(names, function.body.block.args, strict=True)
    }
    operations = tuple(function.body.block.walk())
    for index, operation in enumerate(operations):
        values.update(
            (f"op:{index}:{i}", value)
            for i, value in enumerate(operation.results)
        )
        for region_index, region in enumerate(operation.regions):
            for block_index, block in enumerate(region.blocks):
                values.update(
                    (
                        f"op:{index}:block:{region_index}:{block_index}:arg:{i}",
                        value,
                    )
                    for i, value in enumerate(block.args)
                )
    return function, operations, values
