# Bind, save, load, and deploy a Program

Program deployment combines a portable calculation with the numerical materials and live execution resources needed on the destination. This procedure names fixed key operands, saves a chosen subset of Tensor data, restores missing bindings, and links an executable.

## Prerequisites

Install a CPU-capable build for the complete example. Create or load keys under the application's key-custody policy. A deployed evaluator needs compatible evaluation keys and inputs; keep the secret key on the decrypting side. A Compilation can come from manual IR, either capture frontend, or a callable specialization's `compilation` property.

## 1. Capture a calculation with named materials

```python
import torch
import fhelium as fh
from fhelium import compile as fc
from fhelium.eager import Engine

config = fh.CkksConfig.parse(fh.Preset.slots8192_scale40_depth7_int64)
engine = Engine(config)
secret = engine.create_secret_key(device="cpu")
public = engine.create_public_key(secret, device="cpu")
rotation = engine.create_rotation_key(1, secret, device="cpu")
clear = torch.linspace(-0.05, 0.05, config.num_slots, dtype=torch.float64, device="cpu")
source = engine.encrypt_message(clear, public, device="cpu")

def rotate(value):
    return engine.rotate_with_key(value, rotation)

captured = fc.capture_eager(
    rotate,
    arguments={"value": source},
    material_names={"rotation_1": rotation},
)
print(captured.program.material_descriptions)
```

A material symbol identifies a Tensor operand. Its description helps a caller supply the data. `captured.material_bindings["rotation_1"]` contains the key's Tensor payload; the Program records its use in dataflow. Materials share one ordinary dictionary across passes, and lookup does not generate data.

## 2. Choose what to save

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from fhelium.serialization import save_compilation, load_compilation

with TemporaryDirectory() as directory:
    path = Path(directory) / "rotation.safetensors"
    save_compilation(captured, path, include_materials=False)
    restored = load_compilation(path, device="cpu")
```

`include_materials=False` saves the Program with external materials; `True` saves all currently bound Tensors; a collection such as `{"rotation_1"}` saves selected symbols. Saving is unencrypted and does not recognize sensitive data hidden in arbitrary Tensors. Synchronize with writers before saving, and apply storage access control and encryption at rest externally.

Tensor views, strides, repeated objects, and shared storage are preserved for included bindings. Original addresses, original device placement, autograd history, Compile workspaces, pass reports, and executables are not saved. Loading restores saved storage on the requested device; it does not execute the Program or prepare kernels and resources.

## 3. Supply missing data or replace an assignment

```python
from fhelium.backend.ckks import CkksDeviceResources

resources = CkksDeviceResources(config=config, device="cpu")
unresolved = fc.prepare_material_bindings(
    restored,
    resources=resources,
    keys={"rotation_1": rotation},
)
print("unresolved materials:", unresolved)
```

The optional preparation helper fills missing bindings from supplied providers and keys. It never overwrites an existing binding and never generates a missing key. Missing or ambiguous selections remain unresolved. Named key entries make direct assignments; the caller is responsible for parameter identity, key relation, placement, and payload compatibility.

Direct assignment also permits intentional replacement:

```python
restored.material_bindings["rotation_1"] = rotation.data
```

A linked executable retains the bound Tensor objects. Assign replacements before linking again. Bind live process groups and other non-Tensor handles through the destination Backend workspace.

## 4. Link on the destination and validate

```python
from fhelium.backend import OperationBackend

executable = OperationBackend().link(restored)
result = executable.run(source)
decoded = engine.decrypt_message(result, secret)
print("max absolute error:", float((decoded - torch.roll(clear, 1)).abs().max()))
print(executable.manifest)
```

For a manual Compilation, complete the selected lowering and operand preparation before this step. Restore any configuration or caller workspace entries required by additional passes; they were not serialized. Linking requires the implementation and execution facts consumed by the chosen operations. A portable Program can retain unresolved facts until this preparation stage.

The observable outcome is a restored, newly linked calculation with the intended decoded result. `examples/16_compile_material_persistence.py` exercises omitted, partial, and complete material inclusion. [ArtifactStore](manage-artifacts.md) adds logical names and checked generations for stored values; [device placement](switch-cpu-cuda.md) covers moving data and recreating live state. See [Materials and preparation](../developer/materials-and-preparation.md) and [Compilation persistence](../developer/compilation-persistence.md) for the contracts.
