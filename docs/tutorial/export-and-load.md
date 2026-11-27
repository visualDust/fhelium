# Export and load a compiled callable

**Example source:** [`examples/15_compile_export_load.py`](https://github.com/VisualDust/fhelium/blob/main/examples/15_compile_export_load.py)

`fhelium.compile.export` writes the linked execution of a compiled callable for a set of call signatures to one artifact file. `fhelium.compile.load` reads the file in another process and returns a callable that runs without capture, transformation, linking, or Triton compilation. Example 15 separates the two roles of an encrypted service: a client that owns the secret key and a server that holds only the artifact and the evaluation keys.

## Run the example

```bash
python examples/15_compile_export_load.py
python examples/15_compile_export_load.py --device cpu --output-dir /tmp/fhelium-export
```

The client writes `layer.fhelium`, the rotation key file, and the encrypted input, starts the server as a separate Python process, and decrypts the server's `output.fhelium`. It prints the file sizes and the result depth, scale, and error. With `--output-dir`, the files remain and the example prints a command line for `examples/cpp/load_artifact` that reads the same artifact, key, and input.

## Client: compile and export

```python
@fc.compile
def layer(value):
    plain = engine.prepare_compressed_plaintext(weight, depth=value.depth, device=value.device)
    rotated = engine.add(value, engine.rotate_with_key(value, rotation_key))
    product = engine.multiply_plaintext(engine.coefficient_domain_to_ntt_domain(rotated), plain)
    return engine.rescale_to_next_depth(engine.ntt_domain_to_coefficient_domain(product))

fc.export(layer, directory / "layer.fhelium", signatures=[(x,)], overwrite=True)
fh.save_value(rotation_key, directory / "key-rotation-1.fhelium", overwrite=True)
fh.save_value(x, directory / "input.fhelium", overwrite=True)
```

`layer` computes $(x+\operatorname{roll}(x,1))\odot w$ for a 16-slot period tiled over 8192 slots, with one rescale from depth 0 to depth 1. `signatures` lists example calls; `export` prepares each one and stores its linked schedule as entry `call<i>` together with a copy planned with CUDA Graph segments. The weight is stored as its raw values with a prepare entry that encodes the compressed plaintext when the artifact is loaded. The rotation key is recorded as a key description; its data stays out of the artifact, and the client saves it as a separate value file.

## Server: load and call

```python
keys = [fh.load_value(path, device=device) for path in sorted(directory.glob("key-*.fhelium"))]
layer = fc.load(directory / "layer.fhelium", keys=keys, device=device)
x = fh.load_value(directory / "input.fhelium", device=device, expected_type=Ciphertext)
fh.save_value(layer(x), directory / "output.fhelium", overwrite=True)
```

The server runs in its own process and never sees the secret key or the source of `layer`. `fc.load` matches each key object to the key descriptions recorded in the artifact by value type and metadata; a key read from a value file is compact and is expanded on its device before binding. Loading creates an Engine for the recorded CKKS configuration, runs the artifact's table and prepare entries once, and returns a `LoadedFunction`. A call selects the stored signature whose argument conditions match and raises `SpecializationMiss` when none matches.

## Client: decrypt

The client reads `output.fhelium` and decrypts it with its secret key. The artifact dominates the file sizes: it contains the planned schedules, kernel images, and the recorded tables and weight inputs; the rotation key file is the next largest.

[Deploy a compiled callable](../how-to/deploy-compiled-callable.md) describes the export options, key matching, the manifest, and the C++ loading interface. [Save and relink a Program](../how-to/save-and-relink-program.md) covers Program files for pipelines that bind materials themselves.

::: details Source
<<< @/../examples/15_compile_export_load.py
:::
