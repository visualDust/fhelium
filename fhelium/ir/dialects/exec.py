"""Execution schedules over typed storage buffers.

An exec block is the linked form of a Program: every value is a buffer with a storage class and, where link time determines it, a static layout (sizes, element strides, dtype and device). Operations call registered native or PyTorch operators by schema, launch prepared device kernels, create views, allocate, copy and zero storage, run process-group collectives, call other schedules, and call Python implementations. Structured control flow uses the ``scf`` and ``arith`` dialects on host scalars.

Schedules are imperative: an operation may write a buffer operand in place (``exec.copy``, ``exec.zero`` and operators whose schema marks an argument as written). Operation results name new buffers or views.
"""

from __future__ import annotations

from collections.abc import Sequence

from xdsl.dialects.builtin import (
    ArrayAttr,
    IndexType,
    IntAttr,
    IntegerAttr,
    NoneAttr,
    StringAttr,
)
from xdsl.ir import (
    Attribute,
    Dialect,
    ParametrizedAttribute,
    Region,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    IRDLOperation,
    attr_def,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    param_def,
    result_def,
    traits_def,
    var_operand_def,
    var_region_def,
    var_result_def,
)
from xdsl.traits import IsTerminator

STORAGE_CLASSES = frozenset(
    (
        "input",
        "output",
        "arena",
        "pinned",
        "material",
        "state",
        "internal",
        "view",
    )
)
"""Buffer storage classes.

``input`` buffers are schedule arguments; ``output`` buffers are allocated by the schedule and returned; ``arena`` buffers are schedule-owned temporaries; ``pinned`` buffers are schedule-allocated page-locked host storage; ``material`` buffers are bound tables, keys, plaintexts and link-time constants; ``state`` buffers are mutable external state such as random streams; ``internal`` buffers are allocated by a functional operator; ``view`` buffers alias the storage of their source.
"""


@irdl_attr_definition
class BufferType(ParametrizedAttribute, TypeAttribute):
    """Storage buffer with a storage class and its known layout facts.

    ``sizes`` and ``strides`` are element extents and element strides, or ``none`` when link time does not determine them. ``dtype`` and ``device`` are PyTorch names, or empty when unknown.
    """

    name = "exec.buffer"

    storage: StringAttr = param_def()
    sizes: ArrayAttr[IntegerAttr] | NoneAttr = param_def()
    strides: ArrayAttr[IntegerAttr] | NoneAttr = param_def()
    dtype: StringAttr = param_def()
    device: StringAttr = param_def()


@irdl_attr_definition
class ResourceType(ParametrizedAttribute, TypeAttribute):
    """Live resource handle of one kind, bound when the schedule is linked or loaded."""

    name = "exec.resource"

    kind: StringAttr = param_def()


@irdl_attr_definition
class OperandRef(ParametrizedAttribute):
    """Operator argument read from the operand at ``index``."""

    name = "exec.ref"

    index: IntAttr = param_def()


@irdl_attr_definition
class TorchConstant(ParametrizedAttribute):
    """PyTorch dtype, device, layout or memory-format operator argument."""

    name = "exec.torch"

    kind: StringAttr = param_def()
    value: StringAttr = param_def()


@irdl_op_definition
class CallOperatorOp(IRDLOperation):
    """Call one dispatcher operator with positional arguments.

    ``schema`` is the qualified operator and overload name, such as ``fhelium_rns_ops::add_standard`` or ``aten::select.int``. ``arguments`` lists every schema argument in order: ``exec.ref`` selects an operand, arrays hold list arguments, ``exec.torch`` holds dtype, device, layout and memory-format arguments, and other attributes are scalar constants. Results are the returned Tensors, a returned Tensor list unpacked into ``results``, or none for operators that only write arguments.
    """

    name = "exec.call_op"

    schema = attr_def(StringAttr)
    arguments = attr_def(ArrayAttr)
    inputs = var_operand_def()
    outputs = var_result_def()

    def __init__(
        self,
        schema: str,
        arguments: ArrayAttr,
        inputs: Sequence[SSAValue],
        result_types: Sequence[Attribute],
    ) -> None:
        super().__init__(
            operands=[inputs],
            result_types=[result_types],
            attributes={"schema": StringAttr(schema), "arguments": arguments},
        )


@irdl_op_definition
class LaunchOp(IRDLOperation):
    """Launch one prepared device kernel of the schedule's kernel table.

    ``arguments`` lists the kernel parameters in order as ``exec.ref`` operands and integer constants; ``grid`` is the static launch grid.
    """

    name = "exec.launch"

    kernel = attr_def(StringAttr)
    grid = attr_def(ArrayAttr)
    arguments = attr_def(ArrayAttr)
    inputs = var_operand_def()

    def __init__(
        self,
        kernel: str,
        grid: Sequence[int],
        arguments: ArrayAttr,
        inputs: Sequence[SSAValue],
    ) -> None:
        super().__init__(
            operands=[inputs],
            attributes={
                "kernel": StringAttr(kernel),
                "grid": ArrayAttr(IntegerAttr(g, 64) for g in grid),
                "arguments": arguments,
            },
        )


@irdl_op_definition
class AllocOp(IRDLOperation):
    """Allocate one buffer with the static layout of its result type."""

    name = "exec.alloc"

    result = result_def(BufferType)

    def __init__(self, result_type: BufferType) -> None:
        super().__init__(result_types=[result_type])


@irdl_op_definition
class ViewOp(IRDLOperation):
    """View the storage of ``source`` with static sizes, strides and an element offset relative to the source view."""

    name = "exec.view"

    source = operand_def()
    sizes = attr_def(ArrayAttr)
    strides = attr_def(ArrayAttr)
    offset = attr_def(IntegerAttr)
    result = result_def(BufferType)

    def __init__(
        self,
        source: SSAValue,
        sizes: Sequence[int],
        strides: Sequence[int],
        offset: int,
        result_type: BufferType,
    ) -> None:
        super().__init__(
            operands=[source],
            result_types=[result_type],
            attributes={
                "sizes": ArrayAttr(IntegerAttr(v, 64) for v in sizes),
                "strides": ArrayAttr(IntegerAttr(v, 64) for v in strides),
                "offset": IntegerAttr(offset, 64),
            },
        )


@irdl_op_definition
class CopyOp(IRDLOperation):
    """Copy ``source`` into the storage of ``target``."""

    name = "exec.copy"

    target = operand_def()
    source = operand_def()

    def __init__(self, target: SSAValue, source: SSAValue) -> None:
        super().__init__(operands=[target, source])


@irdl_op_definition
class ZeroOp(IRDLOperation):
    """Set every element of ``target`` to zero."""

    name = "exec.zero"

    target = operand_def()

    def __init__(self, target: SSAValue) -> None:
        super().__init__(operands=[target])


@irdl_op_definition
class MaterialOp(IRDLOperation):
    """Read the bound material ``symbol``: a table, key, plaintext or link-time constant."""

    name = "exec.material"

    symbol = attr_def(StringAttr)
    result = result_def(BufferType)

    def __init__(self, symbol: str, result_type: BufferType) -> None:
        super().__init__(
            result_types=[result_type],
            attributes={"symbol": StringAttr(symbol)},
        )


@irdl_op_definition
class StateOp(IRDLOperation):
    """Read the mutable Tensor ``field`` of the bound resource ``resource``."""

    name = "exec.state"

    resource = attr_def(StringAttr)
    kind = attr_def(StringAttr)
    field = attr_def(StringAttr)
    result = result_def(BufferType)

    def __init__(
        self, resource: str, kind: str, field: str, result_type: BufferType
    ) -> None:
        super().__init__(
            result_types=[result_type],
            attributes={
                "resource": StringAttr(resource),
                "kind": StringAttr(kind),
                "field": StringAttr(field),
            },
        )


@irdl_op_definition
class ResourceOp(IRDLOperation):
    """Read the live resource bound to ``symbol``."""

    name = "exec.resource"

    symbol = attr_def(StringAttr)
    result = result_def(ResourceType)

    def __init__(self, symbol: str, kind: str) -> None:
        super().__init__(
            result_types=[ResourceType(StringAttr(kind))],
            attributes={"symbol": StringAttr(symbol)},
        )


@irdl_op_definition
class CallScheduleOp(IRDLOperation):
    """Call the linked schedule ``callee`` with buffer operands."""

    name = "exec.call"

    callee = attr_def(StringAttr)
    inputs = var_operand_def()
    outputs = var_result_def()

    def __init__(
        self,
        callee: str,
        inputs: Sequence[SSAValue],
        result_types: Sequence[Attribute],
    ) -> None:
        super().__init__(
            operands=[inputs],
            result_types=[result_types],
            attributes={"callee": StringAttr(callee)},
        )


@irdl_op_definition
class PythonCallOp(IRDLOperation):
    """Call a Python implementation from the schedule's callable table.

    The callable receives the buffer operands and returns the result Tensors. Regions are schedules passed to it as callables over Tensor tuples.
    """

    name = "exec.python_call"

    callee = attr_def(StringAttr)
    inputs = var_operand_def()
    outputs = var_result_def()
    bodies = var_region_def()

    def __init__(
        self,
        callee: str,
        inputs: Sequence[SSAValue],
        result_types: Sequence[Attribute],
        bodies: Sequence[Region] = (),
    ) -> None:
        super().__init__(
            operands=[inputs],
            result_types=[result_types],
            regions=[list(bodies)],
            attributes={"callee": StringAttr(callee)},
        )


@irdl_op_definition
class GroupQueryOp(IRDLOperation):
    """Read this process's ``rank`` or the ``size`` of the process group ``group`` into a host integer."""

    name = "exec.group_query"

    kind = attr_def(StringAttr)
    group = operand_def(ResourceType)
    result = result_def(IndexType)

    def __init__(self, kind: str, group: SSAValue) -> None:
        super().__init__(
            operands=[group],
            result_types=[IndexType()],
            attributes={"kind": StringAttr(kind)},
        )


@irdl_op_definition
class CollectiveOp(IRDLOperation):
    """Run one collective of the process group ``group`` and wait for it on the current stream.

    The first ``input_count`` buffer operands are read and the others written. ``all_gather`` writes the input of group rank ``r`` to index ``r`` of the leading dimension of its output; ``broadcast`` writes the input of the group rank ``options[0]`` to its output on every rank.
    """

    name = "exec.collective"

    kind = attr_def(StringAttr)
    options = attr_def(ArrayAttr)
    input_count = attr_def(IntegerAttr)
    group = operand_def(ResourceType)
    buffers = var_operand_def()

    def __init__(
        self,
        kind: str,
        group: SSAValue,
        inputs: Sequence[SSAValue],
        outputs: Sequence[SSAValue],
        options: Sequence[int] = (),
    ) -> None:
        super().__init__(
            operands=[group, [*inputs, *outputs]],
            attributes={
                "kind": StringAttr(kind),
                "options": ArrayAttr(IntegerAttr(v, 64) for v in options),
                "input_count": IntegerAttr(len(inputs), 64),
            },
        )


@irdl_op_definition
class YieldOp(IRDLOperation):
    """Return buffers from an exec fragment."""

    name = "exec.yield"

    values = var_operand_def()
    traits = traits_def(IsTerminator())

    def __init__(self, *values: SSAValue) -> None:
        super().__init__(operands=[values])


def buffer_type(
    storage: str,
    sizes: Sequence[int] | None = None,
    strides: Sequence[int] | None = None,
    dtype: str = "",
    device: str = "",
) -> BufferType:
    """Construct a buffer type from Python layout facts."""
    return BufferType(
        StringAttr(storage),
        NoneAttr()
        if sizes is None
        else ArrayAttr(IntegerAttr(int(size), 64) for size in sizes),
        NoneAttr()
        if strides is None
        else ArrayAttr(IntegerAttr(int(stride), 64) for stride in strides),
        StringAttr(dtype),
        StringAttr(device),
    )


OPERATIONS = (
    CallOperatorOp,
    LaunchOp,
    AllocOp,
    ViewOp,
    CopyOp,
    ZeroOp,
    MaterialOp,
    StateOp,
    ResourceOp,
    CallScheduleOp,
    PythonCallOp,
    GroupQueryOp,
    CollectiveOp,
    YieldOp,
)

Exec = Dialect(
    "exec",
    list(OPERATIONS),
    [BufferType, ResourceType, OperandRef, TorchConstant],
)
"""Execution schedule dialect."""


__all__ = [
    "AllocOp",
    "CallOperatorOp",
    "CallScheduleOp",
    "CollectiveOp",
    "CopyOp",
    "Exec",
    "GroupQueryOp",
    "LaunchOp",
    "MaterialOp",
    "OPERATIONS",
    "OperandRef",
    "PythonCallOp",
    "ResourceOp",
    "ResourceType",
    "STORAGE_CLASSES",
    "BufferType",
    "StateOp",
    "TorchConstant",
    "ViewOp",
    "YieldOp",
    "ZeroOp",
    "buffer_type",
]
