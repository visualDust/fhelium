"""Validate statically bound indexed NTT schedules when linking a Program."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import cast

import torch
from xdsl.dialects.builtin import (
    ArrayAttr,
    StringAttr,
    UnrealizedConversionCastOp,
)
from xdsl.ir import BlockArgument, Operation, SSAValue

from fhelium.ir import Program
from fhelium.ir.dialects import ckks, core, fusion, ntt
from fhelium.native.wrapper import ntt_ops


def _source(value: SSAValue) -> SSAValue:
    """Follow storage-preserving casts and fusion argument bindings."""
    while True:
        owner = value.owner
        if (
            isinstance(owner, UnrealizedConversionCastOp)
            and len(owner.inputs) == len(owner.outputs) == 1
        ):
            value = owner.inputs[0]
        elif isinstance(value, BlockArgument) and isinstance(
            parent := value.owner.parent_op(), fusion.FusedOp
        ):
            value = parent.inputs[value.index]
        else:
            return value


def _schedule_operands(
    operation: Operation,
) -> Iterator[tuple[SSAValue, SSAValue]]:
    if isinstance(operation, ntt._NttOp):
        # The indexed layout carries twiddles, even indices and odd indices.
        if len(operation.tables) == 3:
            yield operation.tables[1], operation.tables[2]
        return
    if isinstance(operation, ckks.PrepareCompressedPlaintextOp):
        if operation.attributes.get("polynomial_domain") != StringAttr(
            "coefficient"
        ):
            yield operation.parameters[5], operation.parameters[6]
        return
    if operation.attributes.get("ntt_backend") != StringAttr("radix2_indexed"):
        return
    names = operation.attributes.get("parameter_names")
    if not isinstance(names, ArrayAttr):
        return
    parameters = (
        value
        for value in getattr(operation, "parameters")
        if not isinstance(_source(value).owner, core.ResourceRefOp)
    )
    operands = dict(
        zip(
            (cast(StringAttr, name).data for name in names),
            parameters,
            strict=True,
        )
    )
    for name, even in operands.items():
        # These are the operation's formal Tensor parameter names, not symbols
        # in the caller's material dictionary.
        if name.endswith(("forward_2", "inverse_2")):
            yield even, operands[name[:-1] + "3"]


def prepare_ntt_schedules(
    program: Program, materials: Mapping[Operation, object]
) -> None:
    """Validate bound schedules once before constructing the host callable.

    Input-dependent schedules remain caller-prepared Tensor operands. Index
    values and layout stay fixed during execution; numerical twiddles and RNS
    parameters continue to be read from their live Tensor bindings.
    """
    prepared: set[tuple[int, int]] = set()
    for operation in program.single_block("main").walk():
        for even_value, odd_value in _schedule_operands(operation):
            even_ref, odd_ref = (
                _source(even_value).owner,
                _source(odd_value).owner,
            )
            if not isinstance(even_ref, core.MaterialRefOp) or not isinstance(
                odd_ref, core.MaterialRefOp
            ):
                continue
            even = cast(torch.Tensor, materials[even_ref])
            odd = cast(torch.Tensor, materials[odd_ref])
            identity = (id(even), id(odd))
            if identity not in prepared:
                ntt_ops.validate_indexed_schedule(even, odd, 2 * even.size(1))
                prepared.add(identity)
