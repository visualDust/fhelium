"""NTT selection with partial choices and caller-supplied table data."""

import pytest
import torch
from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, StringAttr
from xdsl.dialects.func import ReturnOp
from xdsl.ir import Block

from fhelium import Preset
from fhelium.backend import OperationBackend
from fhelium.backend.ntt import NttContext
from fhelium.backend.rns.context import RnsContext
from fhelium.compile import Compilation, Pipeline, prepare_material_bindings
from fhelium.compile.passes.backend import (
    AssignNttImplementationPass,
    SelectNttImplementationsPass,
)
from fhelium.config import CkksConfig
from fhelium.ir import Program
from fhelium.ir.dialects import core, ntt, rns


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_partial_ntt_selection_preserves_supplied_tables_and_open_device(
    device,
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    config = CkksConfig.parse(Preset.slots8192_scale40_depth7_int64)
    context = NttContext(RnsContext(config, device=device))
    ids = (2, 3)
    supplied = context.tensor_operands(ids, inverse=False)
    source = torch.randint(
        0, 65536, (2, len(ids), config.N), dtype=torch.int64, device=device
    )

    def integers(values):
        return ArrayAttr([IntegerAttr(x, 64) for x in values])

    typ = rns.RnsBundleType(
        {
            "shape": integers(source.shape),
            "strides": integers(source.stride()),
            "ring_dimension": IntegerAttr(config.N, 64),
            "prime_ids": integers(ids),
            "dtype": StringAttr("torch.int64"),
            "polynomial_domain": StringAttr("coefficient"),
            "residue_representation": StringAttr("standard"),
        }
    )
    transformed = typ.with_state(
        polynomial_domain=StringAttr("ntt"),
        residue_representation=StringAttr("montgomery"),
    )
    block = Block(arg_types=(typ,))
    parameters = core.MaterialRefOp(
        rns.RnsParametersType(), symbol="parameters"
    )
    twiddles = core.MaterialRefOp(
        core.MessageType(), symbol="supplied-forward-table"
    )
    forward = ntt.CoefficientStandardToNttMontgomeryOp(
        block.args[0], parameters, transformed, tables=(twiddles,)
    )
    inverse = ntt.NttMontgomeryToCoefficientStandardOp(forward, parameters, typ)
    block.add_ops(
        (parameters, twiddles, forward, inverse, ReturnOp(forward, inverse))
    )
    compilation = Compilation(
        Program.from_function(block, (transformed, typ)),
        material_bindings={
            "parameters": supplied[0],
            "supplied-forward-table": supplied[1],
        },
    )
    compilation.workspace[CkksConfig] = config
    backend = OperationBackend()
    selection = SelectNttImplementationsPass(backend.registry)
    assert not selection.run(compilation).changed
    compilation = Pipeline(
        (
            AssignNttImplementationPass(
                algorithm="radix2_indexed"
                if device == "cpu"
                else "radix2_compact",
                group_width=None if device == "cpu" else 16,
            ),
            selection,
        )
    ).run(compilation)
    assert prepare_material_bindings(compilation, resources=context) == ()
    assert (
        compilation.material_bindings["supplied-forward-table"] is supplied[1]
    )
    assert (
        "device"
        not in compilation.program.single_block("main").args[0].type.state.data
    )
    executable = backend.link(compilation)
    for _ in range(2):
        result, restored = executable.run(source)
        torch.testing.assert_close(restored, source, rtol=0, atol=0)
        assert result.shape == source.shape
        source.add_(1)
