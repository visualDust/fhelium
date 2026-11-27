# Troubleshooting

Docs: [Diagnose a value-state mismatch](https://fhelium.550w.host/how-to/diagnose-value-state-mismatch), [Diagnose Compile failures](https://fhelium.550w.host/how-to/diagnose-compile-failures), [Diagnose a Residency failure](https://fhelium.550w.host/how-to/diagnose-residency-failure), [Diagnose a distributed hang](https://fhelium.550w.host/how-to/diagnose-distributed-hang).

## Eager values and keys

| Message | Cause | Fix |
| --- | --- | --- |
| `ValueError: Ciphertext multiplication requires NTT/Montgomery inputs; convert with coefficient_domain_to_ntt_domain` | `multiply` on coefficient-domain operands | Convert both operands with `coefficient_domain_to_ntt_domain` |
| `StateError: add operands disagree on depth: 0 and 1; align them before the operation` (also `prime_ids`, basis, domain, components) | Operands in different states | `mod_switch_to_depth(value, other.depth)` on the shallower operand; convert domains to match |
| `ScaleMismatchError: add requires exactly equal scales: lhs.scale=..., rhs.scale=..., ratio=...` | Operands with different actual scales (`add`, `subtract`, `add_plaintext`, `sum_ciphertexts`, `sum_plaintext_products`) | Match rescale paths, or adjust one operand with `multiply_constant` and rescale |
| `MaximumDepthError: Ciphertext modulus-chain depth is exhausted: depth=7, maximum_depth=7.` | More rescales than the preset provides | Larger-depth preset, fewer multiplicative levels, or bootstrapping |
| `KeyError: 'No installed rotation-key path can realize step S'` | Rotation by step without an installed key, automatic generation disabled | Create `engine.create_rotation_key(S, sk)` and pass it (`rotate_with_key`) or install it |
| `KeyError: 'No relinearization key is installed'` | `relinearize(v)` without a key | `engine.create_relinearization_key(sk)`, pass as second argument |
| `RuntimeError: Secret-key generation is disabled` | Installed-key path needs a secret key | Create keys explicitly |
| `ValueError: <op> requires <Key> material on cuda:N, but the selected key is on ...` | Key on another device | Create keys with `device=`, `.to(device)`, or `Engine(allow_automatic_key_replication=True)` |
| `ValueError: The periodic message must admit a shorter encoded axis` | `prepare_compressed_plaintext` with a full-length message that does not tile | Pass the period (e.g. 16 values) or use an uncompressed plaintext |
| `NativeExtensionError` on import or first operation | Native library missing or ABI mismatch with installed PyTorch | `python scripts/check_env.py`; reinstall a matching wheel ([Installation](https://fhelium.550w.host/tutorial/installation)) |
| Large decryption error | Scale too small, or input outside the polynomial approximation interval | Check `value.scale`, input range vs `chebyshev_fit(..., domain=...)`, compare per stage with clear values |

## Capture (`fhelium.compile.CaptureError`)

| Message | Cause | Fix |
| --- | --- | --- |
| `Captured computation cannot branch on Tensor or ciphertext data` | `if`/`while` on decrypted, Tensor, or ciphertext data | Move the decision outside; branch on static metadata (`value.depth`) or a Python scalar argument |
| `Captured computation cannot mutate bound object state` / `... bound mapping state` | Assigning an attribute or dict entry of a captured object | Set state before the call; return new values |
| `<Type>.<method> cannot run inside a compiled computation; call it between compiled calls and pass the resulting values as arguments` | Residency manager, scope, controller, or lease called inside | Call between compiled calls, pass leased values |
| `<role> requires an evaluation key` | A keyed operation without a key the capture can bind | Create the key and pass it to the operation or `fc.compile(..., keys=...)` |
| `Captured functions cannot declare *args or **kwargs` | Variadic signature | Use explicit parameters |
| `Operands disagree on scale; capture does not insert alignment` (also `depth`) | Mismatched operands | Align as in Eager |

## Compiled calls, cache, export, load

| Message | Cause | Fix |
| --- | --- | --- |
| `SpecializationMiss: No prepared specialization matches this invocation; ...` | `on_miss="error"` or loaded artifact; argument conditions (depth, scale, shape, scalar value) differ | `compiled.prepare(*args)` or export with that signature; check which argument differs |
| `stats()["cache"] == {'ineligible': 'Unsupported Python cache syntax: ImportFrom'}` | Source outside the cacheable subset (also `Unsupported method call: <name>`, `Unsupported external callable`, `Dynamic call targets are not cached`) | Optional: move imports and unsupported calls out of the function; otherwise each process captures again |
| Many specializations, slow first calls | A scalar argument varies by value, or inputs at different depths | Pass varying values as Tensors/plaintexts; normalize depth |
| `ExportError: Schedules with Python calls cannot be exported: ...` | Operation without a native implementation in the schedule | Remove the Python-only operation or keep the callable in-process |
| `ExportError: ... the executable is lowered per input layout` | Lowering depends on runtime input layouts | Not exportable; use in-process compile |
| `ValueError: No supplied key matches artifact keys <sym> (<role>)` | `fc.load(keys=...)` missing a key or wrong layout | Supply the key the exporter used (same step, depth layout); save keys with `fh.save_value` |
| `Artifact identity mismatch: <field>` | Different FHElium native build, Triton, Torch ABI, or CUDA runtime between export and load | Load with the same build, or re-export |
| `Unsupported artifact schema version in <path>` | Artifact from another FHElium version | Re-export |
| Steady latency higher than expected; `detached_outputs` grows | Previous result still referenced at the next call | `del out` before the next call or copy to host |
| `fallback_reason`: `Graph execution requires Tensor inputs on one CUDA device` | A Tensor, ciphertext, or key argument is on the host | Move arguments to the device before the call |
| `RuntimeError: the artifact was captured for process rank R; this process has rank S` | Exported function read `torch.distributed.get_rank()` | Export per rank, or stop reading the rank in the function |
| `ValueError: supply the process group for artifact group '<sym>'` | Artifact uses a non-default process group | `fc.load(..., groups={'<sym>': group})` |
| `RuntimeError: the artifact runs collectives or was captured for one process rank; initialize torch.distributed before loading it` | Loading before `init_process_group` | Initialize first |
| Hang in `torch.distributed.destroy_process_group()` | Live CUDA Graphs of compiled callables or artifacts hold captured NCCL work; the direct call does not release them | Use `fhelium.distributed.shutdown(group)` or let the process exit; reload artifacts with `fc.load(groups=...)` after re-creating groups |
| Cache hit binds the wrong process group | Groups created in a different order than in the process that stored the entry (cache records c10d names) | Create groups in the same order in every process |
| Segment `choice == "direct"` | Replay not faster than direct execution, or capture unsafe (`fallback_reason`) | Usually fine; `graph=True` to force |
