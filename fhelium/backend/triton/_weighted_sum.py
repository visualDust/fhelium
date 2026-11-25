r"""Fuse represented plaintext products and modular sums without temporary products."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from fhelium.backend.rns.operations import weighted_sum_operands
from fhelium.ir._operation_structure import (
    compressed,
    elementwise,
    plaintext_product,
)
from fhelium.ir.dialects import ckks, rns
from fhelium.native.wrapper import rns_ops

from ._expressions import (
    BoundGraph,
    Expr,
    InputLayout,
    ValueExpr,
    structured_value,
)
from ._pointwise import PointwisePlan, execution_parameters


@dataclass
class TritonWeightedSumImplementation:
    r"""Compute each row's sum in bounded NTT/Montgomery expression kernels.

    Each kernel evaluates a caller-selected consecutive range of terms. Its
    modular products and left-associated additions use the existing RNS
    expression generator. Dense and compressed plaintexts remain runtime
    operands; sparse implicit rows are loaded from their supplied Tensors.
    The implementation sets ``requires_request``: Compile selects it only when a caller names it or a fusion region requests it, and an unannotated weighted sum outside a region uses the native implementation.
    Chunk results are added modulo each prime without changing scale or depth.
    """

    name: str = "triton-montgomery-weighted-sum"
    operation_types: tuple = (
        rns.MontgomeryWeightedSumOp,
        rns.MontgomeryWeightedSumsOp,
    )
    supports_in_place: bool = False
    requires_request: bool = True
    _plans: dict[object, PointwisePlan] = field(
        default_factory=dict, init=False, repr=False
    )

    def resource_requirements(self, invocation):
        return ()

    def _row(self, ciphertexts, represented, parameters, radix, out=None):
        inputs = list(ciphertexts)
        inputs.extend(plaintext for plaintext, _, _ in represented)
        inputs.extend(
            implicit.unsqueeze(-1)
            for _, _, implicit in represented
            if implicit is not None
        )
        count = len(ciphertexts)
        layouts = tuple(
            InputLayout(tuple(value.shape), tuple(value.stride()), i < count)
            for i, value in enumerate(inputs)
        )
        kinds = tuple(layout for _, layout, _ in represented)
        # Parameter row stride and radix specialize the generated launcher.
        key = (
            inputs[0].dtype,
            inputs[0].device,
            layouts,
            kinds,
            parameters.stride(0),
            radix,
        )
        plan = self._plans.get(key)
        if plan is None:
            values = tuple(
                ValueExpr(
                    tuple(
                        Expr("input", data=(i, c))
                        for c in range(layout.components)
                    ),
                    layout.shape,
                    layout.component_axis,
                    -i - 1,
                )
                for i, layout in enumerate(layouts)
            )
            implicit_index = 2 * count
            total = None
            for i, (_, layout, implicit) in enumerate(represented):
                operands = [values[i], values[count + i]]
                if implicit is not None:
                    operands.append(values[implicit_index])
                    implicit_index += 1
                structure = (
                    plaintext_product()
                    if layout == "dense"
                    else compressed(
                        "multiply", len(operands) + 1, layout, False
                    )
                )
                term = structured_value(
                    structure,
                    dict(enumerate(operands)),
                    0,
                    rns.MultiplyPlaintextOp
                    if layout == "dense"
                    else ckks.MultiplyCompressedPlaintextOp,
                )
                total = (
                    term
                    if total is None
                    else structured_value(
                        elementwise("add", 2),
                        {0: total, 1: term},
                        0,
                        rns.AddStandardOp,
                    )
                )
            assert total is not None
            plan = PointwisePlan.build(
                BoundGraph(layouts, (total,), {0: total}),
                dtype=inputs[0].dtype,
            )
            self._plans[key] = plan
        return plan.execute(
            tuple(inputs),
            parameters,
            radix,
            None if out is None else {plan.graph.outputs[0].root: out},
        )[0]

    def execute(self, invocation, values, resources, *, in_place):
        del resources
        if torch.version.hip is not None:
            raise ValueError(
                "Triton weighted sums currently require the CUDA backend"
            )
        ciphertexts, represented, parameters = weighted_sum_operands(
            invocation, values
        )
        if any(
            value.ndim < 3 or value.shape != ciphertexts[0].shape
            for value in ciphertexts
        ):
            raise ValueError(
                "Weighted-sum ciphertexts require one shared [component, *batch, prime, coefficient] shape"
            )
        payloads = (
            *ciphertexts,
            *(p for p, _, _ in represented),
            *(i.unsqueeze(-1) for _, _, i in represented if i is not None),
        )
        parameters, radix = execution_parameters(
            invocation, payloads, parameters, in_place=in_place
        )
        if parameters.stride(1) != 1:
            raise ValueError(
                "Triton weighted-sum parameters require contiguous prime rows"
            )
        count = len(ciphertexts)
        chunk = int(invocation.attributes.get("term_chunk_size", count))
        if chunk <= 0:
            raise ValueError("Weighted-sum term chunk size must be positive")
        grouped = invocation.operation_type is rns.MontgomeryWeightedSumsOp
        shape = ciphertexts[0].shape
        # Group sums are written into their slices of one [component, group,
        # ...] result instead of being stacked afterwards.
        result = (
            torch.empty(
                (shape[0], len(represented) // count, *shape[1:]),
                dtype=ciphertexts[0].dtype,
                device=ciphertexts[0].device,
            )
            if grouped
            else None
        )
        outputs = []
        for start in range(0, len(represented), count):
            row = represented[start : start + count]
            total = None
            for offset in range(0, count, chunk):
                partial = self._row(
                    ciphertexts[offset : offset + chunk],
                    row[offset : offset + chunk],
                    parameters,
                    radix,
                    out=None
                    if result is None or total is not None
                    else result[:, start // count],
                )
                if total is None:
                    total = partial
                else:
                    for component in range(total.size(0)):
                        rns_ops.add_standard_(
                            total[component], partial[component], parameters
                        )
            # ciphertexts is nonempty, so every row has at least one chunk.
            assert total is not None
            outputs.append(total)
        return (outputs[0] if result is None else result,)
