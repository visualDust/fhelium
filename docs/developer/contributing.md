# Contributing to FHElium

Contributions should preserve FHElium's mathematical semantics and
value-state invariants across Python, PyTorch tensors, C++/CUDA operators,
generated API reference, examples, and documentation.

## Prepare the source tree

Clone the repository, create an isolated environment with the supported Python
version, and install the selected CPU-only or CUDA-enabled PyTorch build first.
Then install the build tools, the project, and the development dependencies using the
workflow defined by the repository's `pyproject.toml` and `uv.lock`.

After the editable native build, install the declared development group:

```bash
python -m pip install --group dev
pre-commit install
```

Native binaries are specific to the Python, PyTorch, CUDA, and C++ application
binary interfaces (ABIs) and to the GPU architectures selected when they were
built. Do not validate a change against an unrelated cached wheel.

## Before changing code

1. Read the [Developer Guide](index.md) and the relevant subsystem
   page.
2. Read the
   [Mathematical notation and cross-layer invariants](mathematical-notation-and-invariants.md)
   when the change affects CKKS state, RNS/NTT representation, tensor layout,
   scale, level, keys, or a native operation.
3. Identify the public import path, state transition, mutation/aliasing rule,
   numerical oracle, and smallest regression that exercises the affected
   semantics or invariant.
4. Keep unrelated staged and unstaged work unchanged.

## Validation order

Run the smallest affected test first. Broaden validation according to the
surface changed:

```bash
ruff check fhelium tests examples scripts
ruff format --check fhelium tests examples scripts
pyright
pytest -q
python scripts/generate_api_docs.py
npm --prefix docs run typecheck
npm --prefix docs run build
```

Native, CUDA, distributed, packaging, or opt-in bootstrap changes require their
corresponding targeted builds and representative workloads in addition to this
baseline. Record commands, results, skipped validation, and remaining
risk in the contribution description.

Do not loosen a numerical tolerance merely to make a failure pass. Reconcile
the mathematical error model, compare controlled cases, and inspect the
observed error distribution before proposing any change to the acceptance
criterion.

## Documentation changes

Every public API change must update the generated docstring source and the
curated page that places the symbol in the API hierarchy. Every maintained
numbered example must retain a direct tutorial source link and a concrete
opening explanation.

Follow the [documentation contributor guide](documentation.md) for page roles,
API directives, generated-reference commands, diagrams, source links, and site
validation.

## Scope and review

FHElium uses direct, coherent API changes rather than indefinite compatibility
aliases for unreleased or intentionally breaking surfaces. A contribution should
state:

- the problem and supported behavior after the change;
- affected public paths and serialized formats;
- mathematical, numerical, security, and ownership assumptions;
- validation evidence and hardware/software environment;
- migration steps when existing public behavior or requirements change.

Security-sensitive changes need a documented threat model. Performance claims
need a reproducible benchmark definition and environment; a faster isolated
kernel is not sufficient evidence for a faster CKKS workload.

## Useful entry points

- [Repository documentation conventions](documentation.md)
- [System architecture](../concepts/architecture/system-overview.md)
- [Native operator workflow](native-operator-workflow.md)
- [Installation and native source build](../tutorial/installation.md)
- [Benchmark framework API](../api/fhelium/benchmarks/model.md)
