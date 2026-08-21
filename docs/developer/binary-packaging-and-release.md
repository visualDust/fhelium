# Binary packaging and release

FHElium publishes one Python project through PyPI source distributions and
configuration-specific binary repositories. This page describes the supported
wheel identities and the manually dispatched release workflow.

## Distribution model

Ordinary wheel tags identify Python, ABI, and operating-system platform. They
do not identify the selected Torch build, Torch CUDA variant, C++ ABI, CUDA
Toolkit, or embedded GPU architectures. FHElium therefore gives each Torch
configuration its own Simple Repository root under
`https://download.fhelium.550w.host`.

A wheel contains the complete FHElium Python package and native extension.
Torch, its CUDA runtime, and other Python dependencies remain external. Linux
extensions use a Torch-relative runtime path. Windows extensions use Torch's DLL
loading process and its `libiomp5md.dll` OpenMP runtime.

## Supported matrix

[`packaging/release_matrix.json`](https://github.com/VisualDust/fhelium/blob/main/packaging/release_matrix.json)
declares the release cells. Its schema is
[`packaging/release_matrix.schema.json`](https://github.com/VisualDust/fhelium/blob/main/packaging/release_matrix.schema.json).

| Configuration | Linux | Windows | Torch |
|---|---:|---:|---|
| `torch213-cu130` | yes | yes | `2.13.0+cu130` |
| `torch213-cpu` | yes | yes | `2.13.0+cpu` |
| `torch212-cu129` | yes | no | `2.12.1+cu129` |
| `torch212-cu130` | no | yes | `2.12.1+cu130` |
| `torch212-cpu` | yes | yes | `2.12.1+cpu` |

CPython 3.12 and 3.13 produce eight Linux and eight Windows cells. The
nonrectangular CUDA matrix reflects the official Torch wheel sets: Torch 2.12.1
uses CUDA 12.9 on Linux and CUDA 13.0 on Windows.

```bash
python -I packaging/matrix.py validate
python -I packaging/matrix.py cells
python -I packaging/matrix.py show torch213-cu130
```

## Tool responsibilities

The commands under [`packaging/`](https://github.com/VisualDust/fhelium/tree/main/packaging)
follow the release lifecycle:

- `matrix.py` validates declarations and generates install-selector data;
- `build_wheel.py` selects the Linux or Windows wheel builder;
- `linux_wheel.py`, `linux_wheel_check.py`, and `linux_cuda_smoke.py` own Linux
  construction, archive inspection, and real-GPU execution;
- `windows_wheel.py` owns Windows construction and PE/CUDA-image inspection;
- `prepare_release.py`, `merge_repository.py`, `repository_check.py`, and
  `publish_release.py` own the static repository lifecycle;
- `release_identity.py`, `repository_probe.py`, `artifact_probe.py`, and
  `pypi_probe.py` compare local release inputs with remote identities.

These commands validate packaging properties. Functional correctness for CKKS,
CSPRNG, JIT, Residency, and applications belongs to the project test suite, not
the packaging tools.

## Linux wheels

Linux cells build in one of three pinned `manylinux_2_28_x86_64` images: CPU,
CUDA 12.9, or CUDA 13.0. The builder installs the declared Torch wheel, builds
the selected FHElium native backends, repairs the wheel with `auditwheel`, and
checks:

- wheel and CPython ABI tags;
- pinned Torch metadata and the native build manifest;
- ELF dependencies and Torch-relative RPATH;
- expected CUDA runtime linkage;
- declared SASS and PTX architectures.

CUDA wheels are then installed in a clean host environment and execute one
FHElium native CUDA operator on a real GPU.

## Windows wheels

Windows cells build on the Windows x64 self-hosted runner with the matrix-selected
CPython, Torch, Visual Studio 2022 toolset, Windows SDK, and CUDA Toolkit. Each
cell uses a clean short build directory below `~/.fhelium-build`.

The builder checks:

- `win_amd64` and CPython extension tags;
- pinned Torch metadata and the native build manifest;
- PE dependency use of Torch's `libiomp5md.dll` without VCOMP;
- absence of source, build, and Toolkit paths in the extension;
- declared SASS and PTX architectures for CUDA wheels.

The Windows build runner has no R2 credentials, PyPI publishing authority, or
GitHub contents-write permission.

## Static repository

`prepare_release.py` collects the 16 wheels, computes their SHA-256 digests,
writes the release manifest, and generates PEP 503 HTML and PEP 691 JSON pages.
`merge_repository.py` merges previously published wheel records into the new
cumulative pages. `repository_check.py` checks manifest bytes, hashes,
`Requires-Python`, install-selector records, and both Simple Repository
representations.

Published wheel and manifest objects are immutable. Publication uploads those
objects before replacing cumulative index pages. Existing objects are accepted
only when their bytes match the candidate.

## Workflow

[`.github/workflows/release.yml`](https://github.com/VisualDust/fhelium/blob/main/.github/workflows/release.yml)
is manual only. Its default `build-only` mode:

1. validates the tagged source and release matrix;
2. runs the project source checks;
3. builds eight Linux wheels and eight Windows wheels on separate runners;
4. runs minimal artifact and native-operator checks;
5. prepares one combined static-repository candidate;
6. preserves the candidate as a workflow artifact.

Protected `publish` mode is a separate operator choice. It publishes immutable
R2 objects, verifies the PyPI source distribution, updates cumulative indexes,
checks public Linux and Windows installs, and finally creates the GitHub Release.
FHElium 0.10.0 remains immutable; Windows wheels belong to a later release.

## Scope of validation

After modifying packaging code, run static checks for the changed scripts first.
Run actual wheel builds only when build logic, native linkage, matrix identities,
or the release candidate has changed. Packaging changes do not require adding
packaging-specific pytest files.
