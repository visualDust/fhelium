# Customize a Compile pass and pipeline

**Example source:** [`examples/14_compile_custom_pass.py`](https://github.com/VisualDust/fhelium/blob/main/examples/14_compile_custom_pass.py)

Example 14 implements a caller-defined Compile pass that recognizes a captured `torch.matmul` with a compile-time square matrix and rewrites it as a baby-step/giant-step (BSGS) encrypted matrix-vector schedule. The source applies an illustrative elementwise square after the linear result, adding one ciphertext-ciphertext multiplication. Caller-selected passes then place relinearization and rescales, assign depths and actual scales, lower the schedule into RNS/NTT operations, link materials and evaluation keys, and execute it through the Backend.

The example demonstrates a transformation and execution architecture. Its CPU execution validates the selected matrix-multiplication schedule for the shown inputs.

## Run the example

```bash
python examples/14_compile_custom_pass.py
```

The output presents three Programs:

1. the captured Program containing `torch.call` for `torch.matmul`;
2. the semantic BSGS Program produced by the custom pass;
3. the executable RNS/NTT Program produced by the complete selected pipeline.

## Captured operation

The source callable is ordinary PyTorch:

```python
def matrix_vector(x):
    linear = torch.matmul(_WEIGHT, x)
    return linear * linear
```

`_WEIGHT` is a module-level Tensor. PyTorch FX captures it as a graph-external `fhelium.material.ref`, and FHElium retains it in `Compilation.material_bindings`. PyTorch FX preserves encrypted matmul as a structural `torch.call`; the custom pass assigns its FHElium meaning.

A runtime matrix argument would not have a compile-time Tensor payload. This example therefore restricts the pattern to one captured square matrix material multiplied by one encrypted vector.

## Cyclic-diagonal identity

For an `n × n` matrix `A`, define cyclic diagonal `D_k` by

$$
D_k[r] = A[r, (r-k) \bmod n].
$$

Let `R_s` denote `torch.roll(..., shifts=s)`. The direct diagonal form is

$$
Ax = \sum_{k=0}^{n-1} D_k \odot R_k(x),
$$

where `⊙` is pointwise multiplication.

Choose a baby-step width `b` and write `k = jb + i`. The BSGS form used by the pass is

$$
Ax = \sum_j R_{jb}\left(
  \sum_i R_i(x) \odot R_{-jb}(D_{jb+i})
\right),
$$

for indices `jb+i < n`.

The adjustment `R_{-jb}(D_{jb+i})` is required: the outer giant rotation also rotates every diagonal coefficient in the group.

For the example's `n=8` and `b=3`, the encrypted schedule uses baby rotations `1` and `2`, then giant rotations `3` and `6`. It retains eight plaintext multiplications and seven ciphertext additions while reducing the independent input-rotation family.

## Custom pass responsibilities

`LowerConstantMatmulToBsgsPass` performs the following work:

1. find `torch.call` with target `torch.matmul`;
2. locate the matrix's `MaterialRefOp` and Tensor in `Compilation.material_bindings`;
3. compute the eight cyclic diagonals, their group-specific adjustments, and periodic copies covering the configured CKKS slot count;
4. store those derived Tensors under readable material symbols;
5. emit reusable baby rotations;
6. emit semantic pointwise multiplication and addition for each giant group;
7. emit the giant rotations and final group reduction;
8. replace the original matmul result with the BSGS result;
9. publish the selected matrix size, baby width, group count, and material symbols in the Compile workspace.

The pass receives both the baby width and the CKKS slot count:

```python
LowerConstantMatmulToBsgsPass(
    baby_step=3,
    slot_count=config.num_slots,
)
```

Dead-value elimination can then remove the original matrix reference after the custom rewrite has replaced its only use.

## Lowering through existing operations

The custom pass emits only shared semantic operations and material references:

- `fhelium_semantic.roll`;
- `fhelium_semantic.multiply`;
- `fhelium_semantic.add`;
- `fhelium.material.ref`.

The subsequent passes classify encrypted/public operand roles, resolve logical rotation steps to caller-bound key operands, prepare each diagonal as a multiplication-ready plaintext, and lower the encrypted schedule to shared CKKS, RNS, and NTT operations.

The selected `InsertRelinearizationPass` materializes the square activation's three-component result as a two-component ciphertext. Independently, the selected `LateRescalePass` consolidates the BSGS plaintext-product rescales within each add tree. A rotation is a barrier in this conservative policy because moving a rescale across key switching changes the active-Q key-switch work and error. The three giant groups therefore produce three rescales rather than eight; the square activation contributes one additional rescale. `AssignCkksDepthsPass` then reads those concrete transitions, and `AssignCkksScalesPass` computes each actual scale using the complete dropped Q-group product.

This keeps BSGS as one caller-selected algebraic transformation while preserving other matrix-multiplication representations.

## Clear and encrypted checks

The example evaluates the same BSGS formula with clear float64 PyTorch tensors, applies the elementwise square, and compares it with the captured `(_WEIGHT @ x) ** 2` computation. This check catches diagonal indexing, rotation-sign, and post-matvec dataflow mistakes in the example schedule.

The eight-element input and diagonals are repeated periodically across all configured slots. This makes a full-ring CKKS rotation agree with the intended eight-element cyclic rotation instead of padding the remaining slots with zeros. The example encrypts that tiled input, calls `ProgramExecutable.run(ciphertext)`, decrypts the returned `Ciphertext`, and compares every decoded slot with the tiled clear result.

The example prepares numerical tables and key data separately from the Program:

```python
fh_compile.prepare_material_bindings(
    compiled,
    resources=device_resources,
    keys=(*rotation_keys, *((relinearization_key,) if relinearization_key is not None else ())),
)
backend = OperationBackend()
executable = backend.link(compiled)
```

The automatic preparation step adds parameters and key data to `Compilation.material_bindings`, leaving captured matrix materials in place. Backend linking reads this dictionary without generation or description-based correctness checks.

Backend implementations still consume and return Tensor payloads. The executable owns the public Program boundary: it unwraps the input ciphertext, runs the linked Tensor operations, and reconstructs the output ciphertext from the concrete result state assigned by Compile passes.

## Transformation domain

The transformation rule in this example is defined for:

- a square matrix captured as a constant Tensor material;
- matrix-on-the-left vector multiplication;
- one-dimensional encrypted semantic input;
- a caller-selected baby-step width.

Dynamic matrices, rectangular matrices, batched matmul, transposed layouts, and other packing conventions require transformation rules that represent their own operand and layout semantics.

::: details Source

<<< @/../examples/14_compile_custom_pass.py

:::

## Next steps

- [Compose and execute built-in Compile passes](compose-and-execute-compile-pipeline.md) shows the end-to-end path without defining a new pass.
- [Rank-local collective IR](rank-local-collective-ir.md) shows another caller-selected representation change.
- [IR operation and implementation index](../developer/ir-operation-implementation-index.md) lists the operations used by the resulting Program.
