"""Record capture-only origins for structural specialization.

The recorder lives in CompileWorkspace. Ordinary Eager values and operations do not participate in this protocol.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from hashlib import sha256
from types import CodeType

from xdsl.ir import Attribute, Operation, SSAValue

RECORDING = "structural_template_recording"
METADATA = "structural_template_metadata"


@dataclass
class TemplateProvenance:
    """Collect origins while a Python frontend emits its concrete sample."""

    record_origins: bool = True
    internal: int = 0
    generated_values: set[SSAValue] = field(default_factory=set)
    generated_attributes: list[Attribute] = field(default_factory=list)
    options: dict[Operation, dict[str, object]] = field(default_factory=dict)
    reads: list[tuple[SSAValue, str, object]] = field(default_factory=list)
    output_values: list[tuple[dict, SSAValue]] = field(default_factory=list)
    implicit_owners: dict[SSAValue, SSAValue] = field(default_factory=dict)
    origins: dict[Operation, dict[str, object]] = field(default_factory=dict)
    code_sites: dict[CodeType, dict[str, object]] = field(default_factory=dict)
    emit_serial: int = 0

    def register_code(self, code, module):
        if not self.record_origins or code in self.code_sites:
            return
        self.code_sites[code] = {
            "code": sha256(code.co_code).hexdigest(),
            "module": module,
            "qualname": code.co_qualname,
            "firstline": code.co_firstlineno,
        }
        for constant in code.co_consts:
            if isinstance(constant, CodeType):
                self.register_code(constant, module)

    def emission(self, frame, adapter):
        """Read a direct user site and at most its direct captured caller."""
        if not self.record_origins:
            return None
        code = self.code_sites.get(frame.f_code)
        if code is None:
            return None
        self.emit_serial += 1
        site = {**code, "offset": frame.f_lasti}
        caller = frame.f_back
        parent = None if caller is None else self.code_sites.get(caller.f_code)
        return {
            "source": "capture",
            "site": site,
            "call_path": (
                [{**parent, "offset": caller.f_lasti}]
                if parent is not None
                else []
            )
            + [site],
            "occurrence": f"capture:{self.emit_serial}",
            "adapter": adapter,
        }

    def record_emission(self, operation, origin):
        if origin is None:
            return
        sequence = 0
        while operation is not None:
            self.origins.setdefault(
                operation,
                {**origin, "sequence": sequence, "role": operation.name},
            )
            sequence += 1
            operation = operation.next_op

    @contextmanager
    def adapter(self):
        self.internal += 1
        try:
            yield
        finally:
            self.internal -= 1

    def read(self, value: SSAValue, name: str, result: object):
        if self.internal or isinstance(result, SSAValue):
            return
        if name in {"data", "implicit_data", "_resident_tensors"}:
            return
        self.reads.append((value, name, json_value(result)))

    def attributes(self, attributes):
        self.generated_attributes.extend(attributes.values())
        return attributes

    def finish(self, program, input_names):
        from .._template_layout import template_layout

        _, operations, named_values = template_layout(program)
        # Identity keys: xDSL values and operations hash by identity in Python code.
        values = {
            id(value): identity for identity, value in named_values.items()
        }
        return {
            "version": 1,
            "implicit_owners": {
                values[id(value)]: values[id(owner)]
                for value, owner in self.implicit_owners.items()
            },
            "operations": OperationRecords(
                self, operations, [op._operands for op in operations], values
            ),
            "guards": [
                {"value": values[id(value)], "field": name, "expected": result}
                for value, name, result in self.reads
            ],
            "output_values": [
                {"descriptor": descriptor, "value": values[id(value)]}
                for descriptor, value in self.output_values
            ],
        }


class OperationRecords:
    """Per-operation capture records, built from the captured operations on first request.

    Operand tuples are retained at capture time so that records describe the captured Program after later in-place rewrites such as repeat folding.
    """

    def __init__(self, provenance, operations, operands, values):
        self._source = (provenance, operations, operands, values)

    def build(self):
        from fhelium.ir.dialects import core

        provenance, operations, operand_lists, values = self._source
        generated = {id(value) for value in provenance.generated_values}
        generated_ids = {id(attr) for attr in provenance.generated_attributes}
        options = {id(op): value for op, value in provenance.options.items()}
        origins = (
            {id(op): value for op, value in provenance.origins.items()}
            if provenance.record_origins
            else {}
        )
        records = []
        for op, operands in zip(operations, operand_lists, strict=True):
            record = {
                "name": op.name,
                "operands": [values[id(value)] for value in operands],
                "results": [values[id(value)] for value in op.results],
                "generated_operands": [
                    i
                    for i, value in enumerate(operands)
                    if id(value) in generated
                ],
                "generated_attributes": [
                    name
                    for name, attr in op.attributes.items()
                    if id(attr) in generated_ids
                ],
                "options": options.get(id(op), {}),
            }
            origin = origins.get(id(op))
            if origin is not None:
                record["origin"] = origin
            if type(op) is core.MaterialRefOp:
                record["generated_material"] = id(op.value) in generated
            records.append(record)
        return records


def operation_records(metadata):
    """Return the per-operation records of capture metadata, building deferred records once."""
    records = metadata["operations"]
    if isinstance(records, OperationRecords):
        records = metadata["operations"] = records.build()
    return records


def json_value(value):
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if type(value) in {int, float, bool, str, type(None)}:
        return value
    return str(value)
