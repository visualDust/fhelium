"""Program persistence with optional data and shared Tensor views."""

from pathlib import Path

import torch
from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.func import ReturnOp
from xdsl.ir import Block

from fhelium.artifacts import ArtifactStore
from fhelium.backend import OperationBackend
from fhelium.compile import Compilation
from fhelium.ir import Program
from fhelium.ir.dialects import core
from fhelium.serialization import load_compilation, save_compilation


def test_partial_bindings_preserve_views_without_enforcing_descriptions(
    tmp_path: Path,
) -> None:
    block = Block()
    references = tuple(
        core.MaterialRefOp(core.MessageType(), symbol=name)
        for name in ("a", "b", "again")
    )
    block.add_ops((*references, ReturnOp(*(ref.value for ref in references))))
    program = Program.from_function(
        block, tuple(ref.value.type for ref in references)
    )
    program.set_material_description(
        "a",
        {
            "label": "an intentionally replaced key",
            "kind": "RotationKey",
            "rotation_step": 3,
        },
    )
    base = torch.arange(32, dtype=torch.float64)
    left, right = base[2:14:2], base[6:18:2]
    compilation = Compilation(
        program,
        material_bindings={
            "a": left,
            "b": right,
            "again": left,
            "unused": torch.ones(4),
        },
    )
    compilation.workspace[object()] = object()
    path = tmp_path / "partial.safetensors"
    save_compilation(compilation, path, include_materials={"a", "b", "again"})
    restored = load_compilation(path)
    assert not restored.workspace
    assert (
        restored.program.material_descriptions == program.material_descriptions
    )
    assert (
        restored.material_bindings["a"] is restored.material_bindings["again"]
    )
    result = OperationBackend().link(restored).run()
    assert isinstance(result, tuple)
    a, b, again = result
    torch.testing.assert_close(a, left)
    torch.testing.assert_close(b, right)
    assert a is again
    a.add_(1)
    assert b[0].item() == 7
    assert base[6].item() == 6
    storage = torch.empty(0, dtype=torch.float64).set_(
        a.untyped_storage(), 0, (15,), (1,)
    )
    assert bool((storage[1::2] == 0).all())
    bare_path = tmp_path / "program.safetensors"
    save_compilation(compilation, bare_path)
    assert load_compilation(bare_path).material_bindings == {}


def test_artifacts_store_compilation_through_the_same_codec(
    tmp_path: Path,
) -> None:
    compilation = Compilation(
        Program(ModuleOp([])), material_bindings={"weight": torch.arange(5)}
    )
    compilation.program.set_material_description(
        "weight", {"label": "external weight"}
    )
    store = ArtifactStore(tmp_path / "store")
    external = store.put("program", compilation)
    loaded = store.get(external, expected_type=Compilation)
    assert loaded.material_bindings == {}
    complete = store.put("with-data", compilation, include_materials=True)
    loaded = store.get(complete, expected_type=Compilation)
    torch.testing.assert_close(
        loaded.material_bindings["weight"],
        compilation.material_bindings["weight"],
    )
    assert store.inspect(complete).ref == complete
    assert len(store.list()) == 2
