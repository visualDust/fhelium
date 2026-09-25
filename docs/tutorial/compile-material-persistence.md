# Program materials and persistence

**Example source:** [`examples/16_compile_material_persistence.py`](https://github.com/VisualDust/fhelium/blob/main/examples/16_compile_material_persistence.py)

Example 16 separates capture from execution preparation. It names two rotation keys, saves a Program with optional Tensor data, loads it, fills missing bindings, and executes with a newly constructed Backend. It introduces no new optimization policy; [Example 11](compile-jit.md) and [Example 12](compose-and-execute-compile-pipeline.md) cover compilation choices.

```bash
python examples/16_compile_material_persistence.py --include-materials partial
python examples/16_compile_material_persistence.py --include-materials none
python examples/16_compile_material_persistence.py --device cuda:0 --include-materials all
```

Use `--output-dir PATH` to retain the file. Otherwise the example cleans up its temporary directory after execution.

## Identity, description, and data

`material_names={"rotation_1": key}` at capture gives the fixed key a stable symbol. `Program.material_descriptions` records optional descriptive information, including its rotation step. The example adds a user note without changing data. Passes preserve existing symbols independently of operation order. Separate captures should use caller names when cross-capture identification is needed.

`Compilation.material_bindings` is the ordinary symbol-to-Tensor dictionary. Descriptions help users and optional preparation functions identify data; they do not constrain a subsequent assignment. Two equal descriptions do not identify the same secret or the same participant.

## Select what is saved

`save_compilation(..., include_materials=False)` saves only the Program. True includes all current bindings; a collection of symbols selects a subset. The example's default saves only `rotation_1`. Descriptions always travel with the Program, including descriptions whose data is absent.

`load_compilation(..., device=...)` restores the saved bindings. It does not run passes, generate keys, or construct an Engine. Shared storage and strided views among saved Tensors are preserved; only bytes covered by selected views are copied. Original addresses, devices, autograd history, workspace objects, Python callables, pass reports, and executables are not persisted. Data is unencrypted.

## Fill missing bindings and execute

`prepare_material_bindings(restored, resources=..., keys=...)` fills missing entries using caller-supplied resources and keys. Existing bindings are left untouched. Unsupported descriptions and ambiguous candidates remain unbound and are returned as symbols. Keys are not generated.

The example then assigns `rotation_2` directly and calls `OperationBackend().link(restored)`. That assignment can deliberately select a different Tensor; linking does not authenticate it against the description. Numerical implementations retain their actual execution ABI requirements.

Changing a bound Tensor's contents is visible to execution. Replacing its binding with another Tensor requires relinking. Each returned rotation is compared with the original Eager computation using the same key. The Program contains native-supported rotations, so this example needs no arithmetic-lowering pipeline.

[ArtifactStore](artifact-store.md) can store the same Compilation representation under a logical name.

::: details Source
<<< @/../examples/16_compile_material_persistence.py
:::
