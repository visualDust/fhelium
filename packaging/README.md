# FHElium packaging tools

This directory contains the commands used by the manually dispatched release
workflow. They validate artifact identity, build Linux and Windows wheels,
prepare the static Simple Repository, and publish or probe release objects.
Project behavior is tested by pytest; these commands do not test CKKS, CSPRNG,
JIT, Residency, or application workloads.

## Declarations

- `release_matrix.json` declares four Linux and four Windows configuration
  identities, each built for CPython 3.12 and 3.13 (16 cells total).
- `release_matrix.schema.json` validates the JSON structure.
- `matrix.py` validates cross-field rules and projects the documentation install
  catalog.
- `manylinux_2_28*.Dockerfile` define the three Linux build environments.

```bash
python -I packaging/matrix.py validate
python -I packaging/matrix.py cells
python -I packaging/matrix.py show torch213-cu130
```

## Wheel construction

- `build_wheel.py` is the workflow entry point for one declared cell.
- `linux_wheel.py` builds one manylinux wheel inside its selected container.
- `linux_wheel_check.py` checks Linux wheel metadata, ELF dependencies, native
  manifest, RPATH, and CUDA image declarations.
- `linux_cuda_smoke.py` installs a Linux CUDA wheel and executes one native CUDA
  operator on a real GPU.
- `windows_wheel.py` selects the declared MSVC/SDK/CUDA environment, builds one
  Windows wheel, and checks its metadata, PE dependencies, native manifest,
  local-path absence, and CUDA images.

The Windows builder uses a short, stable directory below `~/.fhelium-build` so
MSVC and CUDA host compilation do not embed machine-specific source paths.
FHElium's CMake configuration supplies deterministic MSVC and CUDA host-link
options. Windows wheels import Torch's `libiomp5md.dll`; they must not import
VCOMP.

## Repository preparation and publication

- `prepare_release.py` collects all declared wheels, writes the release
  manifest, and generates PEP 503 HTML and PEP 691 JSON pages.
- `merge_repository.py` merges previously published wheel records into the new
  cumulative pages.
- `repository_check.py` checks the candidate tree, manifest hashes, catalog,
  and HTML/JSON projections.
- `publish_release.py` uploads immutable artifacts before mutable index pages.

Published wheel objects are immutable. An existing object is accepted only when
its bytes match the candidate. Index pages are cumulative and are published
after every wheel and release manifest is available.

## Remote identity probes

- `release_identity.py` checks project version and local/remote tag identity.
- `repository_probe.py` checks canonical Simple Repository routing and content
  negotiation.
- `artifact_probe.py` compares public wheel and manifest bytes with the local
  release manifest.
- `pypi_probe.py` compares the public sdist with the locally built sdist.

These probes verify release artifacts and services. They do not execute project
functional tests.

## Workflow

`.github/workflows/release.yml` is manual only. Its default `build-only` mode:

1. validates the source and matrix;
2. builds eight Linux wheels and eight Windows wheels on separate runners;
3. executes minimal native wheel checks;
4. prepares and validates one combined repository candidate;
5. preserves the candidate as a workflow artifact.

Protected `publish` mode performs publication only after both platform builds
and candidate preparation succeed. The Windows runner receives no PyPI or R2
publication credentials.
