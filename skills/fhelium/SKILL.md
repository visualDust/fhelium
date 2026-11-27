---
name: fhelium
description: Write, compile, deploy, debug, and measure CKKS homomorphic-encryption programs with FHElium (`import fhelium`, `fhelium.eager.Engine`, `fhelium.compile`). Use for encrypted arithmetic on PyTorch tensors, choosing presets and keys, compiling with `@fc.compile`, CUDA Graph execution, exporting compiled artifacts, and diagnosing FHElium errors or performance.
---

# FHElium

FHElium is a PyTorch library for CKKS homomorphic encryption. Programs are written as Eager `Engine` calls on encrypted values, verified against clear computations, then compiled with `fhelium.compile` for repeated execution and export. Documentation: https://fhelium.550w.host.

## Mental model

- Values (`fh.Ciphertext`, `fh.Plaintext`, `fh.CompressedPlaintext`) carry CKKS state: `depth` (consumed-depth index, 0 = full Q basis; `engine.depth_remaining(value)` gives the remaining rescales), actual `scale` (per value, not the preset default), polynomial domain (`coefficient` or `ntt`), modulus basis, component count, and device.
- Every Engine operation has a documented state transition (see its docstring). FHElium does not insert alignment: binary operations need matching depth, scale, and domain, and the program aligns them (`mod_switch_to_depth`, `rescale_to_next_depth`, domain conversions). Eager raises `ScaleMismatchError` for unequal scales and `StateError` naming the field for depth, prime IDs, basis, domain, or component mismatches; capture under `fc.compile` applies the same rules (`CaptureError: Operands disagree on <field>`).
- `Engine(config)` owns a CKKS context, not a device. Factories take `device=`; operations run where their operands live; `.to(device)` moves values.
- Keys are application objects. Create them with `engine.create_*_key(...)` and pass them to operations or collect them in `fh.EvaluationKeySet`. The Engine's installed-key convenience methods (`rotate_by_step`, `rotation_key`, `relinearize` without a key) generate missing keys from the installed secret key unless `Engine(..., allow_automatic_key_generation=False)`; compiled and exported code never generates keys.
- Residue dtype follows the preset's prime widths: primes below 2^30 use int32 residues, wider primes int64.
- `fc.compile` captures a function once per input condition (a *specialization*), links it to an exec schedule, and runs it on a native Runner, with CUDA Graphs of its segments on CUDA.

Concepts: [value model](https://fhelium.550w.host/concepts/ckks/value-model-and-identity), [value-state transitions](https://fhelium.550w.host/concepts/ckks/value-state-transitions), [key lifecycle](https://fhelium.550w.host/concepts/ckks/key-lifecycle), [compiled execution](https://fhelium.550w.host/concepts/execution/compiled-execution).

## Workflow

a. **Environment.** Run `python scripts/check_env.py` (or `fhelium version` and `fhelium cuda info`). Stop if native is unavailable; see [Installation](https://fhelium.550w.host/tutorial/installation) and [Inspect runtime, memory, and CUDA topology](https://fhelium.550w.host/how-to/inspect-runtime-and-cuda).

b. **Preset.** Count the multiplicative depth of the computation (each rescale consumes one depth) and the slots needed. Presets are named `slots{n}-scale{bits}-depth{L}`, e.g. `fh.Preset.slots8192_scale40_depth7`; `-boot{k}` presets (`slots32768-scale40-depth{20,24,29}-boot{4,8,13}`, `slots32768-scale50-depth{24,28}-boot{8,12}`) reserve a bootstrap segment and leave k application depths after each refresh. See [Choose a preset and chain depth](https://fhelium.550w.host/how-to/choose-preset-and-depth), example 03.

c. **Eager first.** Write the computation with Engine calls, decrypt, and compare with the same computation on clear tensors. Start from `scripts/skeleton.py`. Report max abs error with the preset and the depth transition. See [Basic CKKS workflow](https://fhelium.550w.host/tutorial/basic-ckks-workflow), [Diagnose a value-state mismatch](https://fhelium.550w.host/how-to/diagnose-value-state-mismatch).

d. **Compile.** Wrap the verified function or method with `@fc.compile`; check it reproduces the Eager result. See [Choose compile boundaries](https://fhelium.550w.host/how-to/choose-compile-boundaries), [Compile a callable](https://fhelium.550w.host/how-to/compile-callable), example 12.

e. **Measure.** Separate first call and steady state; read `compiled.stats()`. See `references/performance.md`, [Measure and optimize a workload](https://fhelium.550w.host/how-to/measure-and-optimize), examples 12, 17.

f. **Deploy.** `fc.export(compiled, path, signatures=[(x,)])`, then `fc.load(path, keys=[...])` in Python or `fhelium::Artifact` in C++. See [Deploy a compiled callable](https://fhelium.550w.host/how-to/deploy-compiled-callable), example 15, `examples/cpp/load_artifact.cpp`.

## Rules

- Verify numerics by decrypting and comparing with a clear reference. Never loosen a tolerance to make a result pass; find the cause.
- Prefer structured operations over Python loops of rotations: `engine.sum_plaintext_products`, `engine.sum_rotations`, `engine.rotate_many_by_steps`, `fhelium.algorithms.linear`, `fhelium.algorithms.polynomial`. Loops inside compiled functions unroll.
- Create every key the program needs before compiling. For linear transforms, `linear.plan(t).required_rotation_offsets(t)` lists the rotation steps; for bootstrapping, `bootstrap.create_evaluation_keys(engine, plan, secret_key)`; for a Program, `fhelium.ir.analyze_evaluation_key_requirements(program)`. Missing keys are reported by the error; create them, do not rely on automatic generation.
- Prepare constant operands once when they are reused across calls: `prepared = linear.prepare(engine, like, transform, keys=keys)` outside, `linear.apply(engine, x, prepared=prepared)` inside a compiled function, so diagonals are encoded once; likewise `bootstrap.prepare(...)` then `bootstrap.bootstrap(..., prepared=...)`. `linear.apply(engine, x, transform, keys=keys)` and `polynomial.evaluate(engine, x, f, polynomial.plan(f), keys=...)` also work inside compiled functions.
- Keep runtime decisions outside compiled functions: data-dependent branches, request routing, key selection, Residency calls. Pass per-request data (ciphertexts, per-user weights) as arguments; Python scalars specialize by value.
- Release compiled results before the next call when they are not needed (`del out`); a result still referenced when the same specialization runs again is copied out of the output region. Stream inputs with `fhelium.runtime.DoubleBuffer`.
- Size weights and keys before choosing a preset or placement: encode periodic diagonals as `CompressedPlaintext` at the consuming depth, and stream weights or keys that do not fit with `DoubleBuffer` one step ahead; see "Weights and keys" in `references/performance.md`.
- Report timings with first call and steady latency, peak memory, preset, input→output depth, and the full GPU name.

## References

- `references/api-map.md` — modules and their main entry points.
- `references/programming.md` — presets, keys, state transitions, structured operations, bootstrapping, placement.
- `references/compile-deploy.md` — compile boundaries, options, cache, export/load, C++, serving.
- `references/performance.md` — measurement recipe, Graph statistics, memory.
- `references/troubleshooting.md` — error messages, causes, fixes.
- `scripts/check_env.py`, `scripts/skeleton.py` — environment report and a minimal verified Eager-then-compile program.
