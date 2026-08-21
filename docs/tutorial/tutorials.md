# Tutorials

## First evaluator

This program encrypts two vectors, evaluates addition and multiplication, and
decrypts the results.

```python
import torch
import fhelium as fh

engine = fh.CkksEngine(fh.Preset.slots8192_scale40_levels7_int64, device="cpu")

x = torch.linspace(-0.05, 0.05, 32, dtype=torch.float64)
y = torch.linspace(0.02, -0.02, 32, dtype=torch.float64)

ct_x = engine.encrypt_message(x)
ct_y = engine.encrypt_message(y)

# Addition preserves level and scale.
ct_sum = engine.add(ct_x, ct_y)

# Multiplication exposes representation conversion, relinearization, and a
# post-product rescale.
x_ntt = engine.coefficient_domain_to_ntt_domain(ct_x)
y_ntt = engine.coefficient_domain_to_ntt_domain(ct_y)
triplet = engine.multiply(x_ntt, y_ntt)
ct_product = engine.rescale_to_next_level(engine.relinearize(triplet))

sum_clear = engine.decrypt_message(ct_sum, is_real=True)[: x.numel()]
product_clear = engine.decrypt_message(ct_product, is_real=True)[: x.numel()]

torch.testing.assert_close(sum_clear, x + y, atol=2e-5, rtol=0)
torch.testing.assert_close(product_clear, x * y, atol=2e-5, rtol=0)
```

The separate operations are intentional:

1. in this example, fresh ciphertexts use `engine.config.default_scale` as
   their initial actual scale $\Delta$;
2. `coefficient_domain_to_ntt_domain` enters NTT/Montgomery representation;
3. `multiply` accepts two two-component NTT ciphertexts and returns a
   three-component NTT ciphertext;
4. `relinearize` returns the ordinary two-component form, and
   `rescale_to_next_level` then consumes one Q prime and records the product scale
   divided by that prime.

See [Scale and level lifecycle](../concepts/ckks/scale-and-level-lifecycle.md)
for the level/scale laws,
[Evaluator operation transitions](../concepts/ckks/evaluator-operation-transitions.md)
for the broader state machine, and
[`CkksEngine`](../api/fhelium/engine/ckks_engine.md#ckksengine) for the generated
method reference. The same program runs with `device="cuda:0"` when the native
build includes CUDA; see
[Choose and switch a local execution device](../how-to/switch-cpu-cuda.md).
Before optimizing or deploying this evaluator, continue with
[Screen NTT backends](../how-to/screen-ntt-backends.md) to compare the
implementations compatible with the target device and preset.

## Choose a tutorial

Each tutorial follows one maintained numbered file under
[`examples/`](https://github.com/VisualDust/fhelium/tree/main/examples). Choose
a track by goal; the numbers preserve the source mapping and do not impose one
mandatory reading order.

## CKKS

| Example | Tutorial | Main question |
| --- | --- | --- |
| [01](https://github.com/VisualDust/fhelium/blob/main/examples/01_basic_ckks_flow.py) | [Basic CKKS workflow](basic-ckks-workflow.md) | How do encryption, three-component multiplication state, rotation, and decryption fit together? |
| [02](https://github.com/VisualDust/fhelium/blob/main/examples/02_key_materials.py) | [Key material lifecycle](key-materials.md) | What state does each key store, and which cryptographic relations remain application-owned? |
| [04](https://github.com/VisualDust/fhelium/blob/main/examples/04_modulus_chain_depth.py) | [Modulus-chain depth](modulus-chain-depth.md) | How do the level-transition budget, configured chain depth, security budget, and ciphertext size relate? |
| [05](https://github.com/VisualDust/fhelium/blob/main/examples/05_explicit_scale_management.py) | [Explicit scale management](explicit-scale-management.md) | How does a program track the actual dropped prime and keep level alignment separate from scale policy? |
| [06](https://github.com/VisualDust/fhelium/blob/main/examples/06_explicit_state_late_relinearization_ntt.py) | [Late relinearization and NTT reuse](late-relinearization-and-ntt-reuse.md) | When can products remain three-component and operands remain in NTT form? |

## Performance

| Example | Tutorial | Main question |
| --- | --- | --- |
| [07](https://github.com/VisualDust/fhelium/blob/main/examples/07_rotation_hoisting_benchmark.py) | [Rotation hoisting](rotation-hoisting.md) | When does a grouped rotation request avoid repeated decomposition work? |

## Distributed execution

| Example | Tutorial | Main question |
| --- | --- | --- |
| [08](https://github.com/VisualDust/fhelium/blob/main/examples/08_spmd_independent_ciphertexts.py) | [Independent ciphertexts](spmd-independent-ciphertexts.md) | When should SPMD code scatter and gather independent encrypted values? |
| [09](https://github.com/VisualDust/fhelium/blob/main/examples/09_spmd_rotation_parallel_mxv.py) | [Rotation-parallel matrix-vector](spmd-rotation-parallel-matvec.md) | How are additive diagonal terms and direct rotation keys partitioned across processes? |
| [10](https://github.com/VisualDust/fhelium/blob/main/examples/10_spmd_limb_parallel_pipeline.py) | [Limb-parallel pipeline](spmd-limb-parallel-pipeline.md) | Which operations are RNS-row local, and where must every expected active row be reconstructed? |

## Execution and lifecycle

| Example | Tutorial | Main question |
| --- | --- | --- |
| [03](https://github.com/VisualDust/fhelium/blob/main/examples/03_plaintext_ciphertext_memory.py) | [Values, memory, and persistence](value-memory-and-persistence.md) | How do device movement, value files, and artifact policy differ? |
| [11](https://github.com/VisualDust/fhelium/blob/main/examples/11_cuda_graph_matrix_vector.py) | [CUDA Graph matrix-vector](cuda-graph-matvec.md) | How are static keys and weights separated from changing request ciphertexts? |
| [12](https://github.com/VisualDust/fhelium/blob/main/examples/12_reusable_value_buffer.py) | [Reusable value buffers](reusable-value-buffer.md) | How can pinned-host tiles stream through two fixed CUDA allocations? |
| [13](https://github.com/VisualDust/fhelium/blob/main/examples/13_explicit_residency.py) | [Explicit residency plans and CUDA leases](explicit-residency.md) | How do opaque handles, lazy local locations, optional budgets, scoped reservations, and event-backed CUDA leases compose? |
| [14](https://github.com/VisualDust/fhelium/blob/main/examples/14_automatic_residency.py) | [Automatic residency admission](automatic-residency.md) | How does a working-set request become a deterministic, inspectable, state-bound admission decision under managed pressure? |
| [15](https://github.com/VisualDust/fhelium/blob/main/examples/15_homogeneous_batching.py) | [Homogeneous batching](homogeneous-batching.md) | How does a leading message batch compare with an explicit loop? |
| [16](https://github.com/VisualDust/fhelium/blob/main/examples/16_compressed_plaintext.py) | [Compressed plaintexts](compressed-plaintext.md) | When can an operation-ready plaintext use the versioned compressed encoded-axis layout? |

## Features

Read the [bootstrapping composition and range requirements](../concepts/ckks/composable-bootstrapping.md)
or the [multiparty supported security scope](../how-to/use-multiparty-ckks.md) before the
corresponding workflow.

| Example | Tutorial | Main question |
| --- | --- | --- |
| [17](https://github.com/VisualDust/fhelium/blob/main/examples/17_ckks_bootstrap_logn16.py) | [Refresh with composable CKKS bootstrapping](composable-ckks-bootstrap.md) | How are approximation, polynomial evaluation, transforms, periodic reduction, keys, and range evidence composed? |
| [18](https://github.com/VisualDust/fhelium/blob/main/examples/18_multiparty_ckks.py) | [Multiparty CKKS](multiparty-ckks.md) | How do stateless collective-key and unsafe output arithmetic phases fit together under application-owned protocol state? |
| [19](https://github.com/VisualDust/fhelium/blob/main/examples/19_unified_jit.py) | [JIT programs](unified-jit.md) | How do PyTorch tracing, a mixed-dialect `Program`, selected passes, a retained workspace, readiness, and encrypted execution compose? |
| [20](https://github.com/VisualDust/fhelium/blob/main/examples/20_jit_textual_ir.py) | [Import and execute JIT textual IR](jit-textual-ir.md) | How can versioned textual IR preserve an application operation and execute it through a bound handler? |
| [21](https://github.com/VisualDust/fhelium/blob/main/examples/21_jit_custom_pipeline.py) | [Compose a custom JIT pipeline](jit-custom-pipeline.md) | How does a custom pass publish retained workspace analysis inside an inspectable pipeline? |

::: info Static documentation build
The site build does not run native workloads. CPU and CUDA validation exercise
applicable examples separately from VitePress generation.
:::

Use [Concepts](../concepts/index.md) for the underlying invariants,
[How-to guides](../how-to/index.md) for focused tasks, and the
[API reference](../api/index.md) for signatures.
