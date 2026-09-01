# Contributing to FHElium

Contributions should preserve FHElium's mathematical semantics and
value-state invariants across Python, PyTorch tensors, C++/CUDA operators,
generated API reference, examples, and documentation.

## Prepare the source tree

Choose one environment workflow for a checkout. Use separate virtual
environments when validating both workflows because the uv environment selects
the locked Torch build while the pip environment preserves a Torch build chosen
by the contributor.

### Locked uv environment

The tracked lock defines the default developer environment:

```bash
uv sync --locked
source .venv/bin/activate
pre-commit install
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. The sync
installs the development tools and builds FHElium as an editable package.

### Environment with a selected Torch build

Create and activate a virtual environment, install the intended Torch package,
then build FHElium without build isolation:

```bash
python -m pip install --group build
python -m pip install \
  --editable . --verbose --no-build-isolation --no-cache-dir
python -m pip install --group dev
pre-commit install
```

`--no-build-isolation` lets native configuration inspect the selected Torch
ABI. `--no-cache-dir` prevents reuse of a wheel built for another Python,
Torch, CUDA, or C++ ABI.

### Development tools and repository metadata

The development files have separate responsibilities:

- `pyproject.toml` declares build and development dependency groups;
- `uv.lock` records the locked developer resolution;
- `packaging/release_matrix.json` declares the Python, Torch, CUDA, operating
  system, and artifact configurations used for releases;
- `justfile` provides optional shortcuts and does not define dependencies or
  release support.

Running `just` without a recipe lists available commands. Cleanup requires a
named recipe such as `just clean-build`; no default command deletes build or
environment files. `just check` runs Ruff, Pyright, and pytest.

Use the build shortcut matching the active environment when native source
changes:

```bash
just NATIVE_BACKENDS=CPU build-uv
just NATIVE_BACKENDS=CPU+CUDA build-pip
```

The shortcuts rebuild the editable extension and refresh the ignored
`build/compile_commands.json` used by `.clangd`. The refresh step selects the
ABI-specific database for the active CPython interpreter and replaces uv's
temporary isolated-build Torch include paths with the active environment's
persistent Torch include paths. To refresh editor data after a direct build,
run:

```bash
python scripts/refresh_compile_commands.py
```

The lock defines the reproducible default development environment.
`packaging/release_matrix.json` defines the
formal Python, Torch, CUDA, and operating-system artifact configurations.

Native binaries are specific to the Python, PyTorch, CUDA, and C++ application
binary interfaces (ABIs) and to the GPU architectures selected when they were
built. Do not validate a change against an unrelated cached wheel.

## Before changing code

1. Read the [Developer Guide](index.md) and the relevant subsystem
   page.
2. Read the
   [Terminology and mathematical model](../concepts/terminology-and-mathematical-model.md)
   when the change affects CKKS state, RNS/NTT representation, tensor layout,
   scale, level, keys, or a native operation.
3. Identify the affected interface or owned subsystem and the smallest check
   that exercises the proposed change. For evaluator changes, also identify
   the state transition, mutation or aliasing rule, and numerical oracle.
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
curated page that places the symbol in the API hierarchy. Every numbered example must retain a direct tutorial source link and a concrete
opening explanation.

Follow the [documentation contributor guide](documentation.md) for page roles,
API directives, generated-reference commands, diagrams, source links, and site
validation.

## Review evidence

FHElium uses direct, coherent API changes for unreleased or intentionally
breaking surfaces. A contribution should
state:

- the problem and supported behavior after the change;
- affected public paths and serialized formats;
- mathematical, numerical, security, and ownership assumptions when affected;
- validation evidence and hardware/software environment;
- migration steps when existing public behavior or requirements change.

Security-sensitive changes need a documented threat model. Performance claims
need a reproducible benchmark definition and environment that measures the
claimed CKKS workload.

## Useful entry points

- [Repository documentation conventions](documentation.md)
- [System architecture](../concepts/architecture/system-overview.md)
- [Native operator workflow](native-operator-workflow.md)
- [Installation and native source build](../tutorial/installation.md)
- [Benchmark framework API](../api/fhelium/benchmarks/model.md)
