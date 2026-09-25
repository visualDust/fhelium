# Binary packaging and release

FHElium publishes one Python project through PyPI source distributions and configuration-specific binary repositories. This page describes the supported wheel identities and the manually dispatched release workflow.

## Distribution model

Ordinary wheel tags identify Python, ABI, and operating-system platform. They do not identify the selected Torch build, Torch CUDA variant, C++ ABI, CUDA Toolkit, or embedded GPU architectures. FHElium therefore gives each Torch configuration its own Simple Repository root under `https://download.fhelium.550w.host`.

A wheel contains the complete FHElium Python package and native extension. Torch, its CUDA runtime, and other Python dependencies remain external. Linux extensions use a Torch-relative runtime path. Windows extensions use Torch's DLL loading process and its `libiomp5md.dll` OpenMP runtime.

## Supported matrix

[`packaging/release_matrix.json`](https://github.com/VisualDust/fhelium/blob/main/packaging/release_matrix.json) declares the release cells. Its schema is [`packaging/release_matrix.schema.json`](https://github.com/VisualDust/fhelium/blob/main/packaging/release_matrix.schema.json).

| Configuration | Linux | Windows | Torch |
|---|---:|---:|---|
| `torch213-cu130` | yes | yes | `2.13.0+cu130` |
| `torch213-cpu` | yes | yes | `2.13.0+cpu` |
| `torch212-cu129` | yes | no | `2.12.1+cu129` |
| `torch212-cu130` | no | yes | `2.12.1+cu130` |
| `torch212-cpu` | yes | yes | `2.12.1+cpu` |

CPython 3.12 and 3.13 produce eight Linux and eight Windows cells. The nonrectangular CUDA matrix reflects the official Torch wheel sets: Torch 2.12.1 uses CUDA 12.9 on Linux and CUDA 13.0 on Windows.

```bash
python -I packaging/matrix.py validate
python -I packaging/matrix.py cells
python -I packaging/matrix.py show torch213-cu130
```

## Tool responsibilities

The commands under [`packaging/`](https://github.com/VisualDust/fhelium/tree/main/packaging) follow the release lifecycle:

- `matrix.py` validates declarations and generates install-selector data;
- `build_wheel.py` selects the Linux or Windows wheel builder and orchestrates installed-wheel verification, including target-host GPU execution for Linux CUDA cells;
- `linux_wheel.py` and `linux_wheel_check.py` own Linux construction and archive inspection;
- `windows_wheel.py` owns Windows construction and PE/CUDA-image inspection;
- `prepare_release.py`, `merge_repository.py`, `repository_check.py`, and `publish_release.py` own the static repository lifecycle;
- `release_identity.py`, `repository_probe.py`, `artifact_probe.py`, and `pypi_probe.py` compare local release inputs with remote identities.

Archive checks inspect wheel metadata, dependencies, and embedded native images. Installed-wheel checks load the extension and execute the selected native operations.

## Linux wheels

Linux cells build in one of three pinned `manylinux_2_28_x86_64` images: CPU, CUDA 12.9, or CUDA 13.0. The builder installs the declared Torch wheel, builds the selected FHElium native backends, repairs the wheel with `auditwheel`, and checks:

- wheel and CPython ABI tags;
- pinned Torch metadata and the native build manifest;
- ELF dependencies and Torch-relative RPATH;
- expected CUDA runtime linkage;
- declared SASS and PTX architectures.

CUDA wheels are then installed beneath the caller-owned `--work-root` on the target host and execute one FHElium native CUDA operator on a real GPU.

## Windows wheels

Windows cells build on the Windows x64 self-hosted runner with the matrix-selected CPython, Torch, Visual Studio 2022 toolset, Windows SDK, and CUDA Toolkit. Each cell uses a clean directory below the caller-owned `--work-root`. The release workflow keeps the physical root below the runner's temporary directory and maps it through a temporary short drive alias to satisfy Windows path-length limits. It removes the alias after the build. Standalone builds can select their working directory through `--work-root`.

The builder checks:

- `win_amd64` and CPython extension tags;
- pinned Torch metadata and the native build manifest;
- PE dependency use of Torch's `libiomp5md.dll` without VCOMP;
- absence of source, build, and Toolkit paths in the extension;
- declared SASS and PTX architectures for CUDA wheels.

The Windows build runner has no R2 credentials, PyPI publishing authority, or GitHub contents-write permission.

## Static repository

`prepare_release.py` collects the 16 wheels, computes their SHA-256 digests, writes the release manifest, and generates PEP 503 HTML and PEP 691 JSON pages. `merge_repository.py` merges previously published wheel records into the new cumulative pages. `repository_check.py` checks manifest bytes, hashes, `Requires-Python`, install-selector records, and both Simple Repository representations.

Published wheel and manifest objects are immutable. Publication uploads those objects before replacing cumulative index pages. Existing objects are accepted only when their bytes match the candidate.

## Workflow

[`.github/workflows/release.yml`](https://github.com/VisualDust/fhelium/blob/main/.github/workflows/release.yml) is manual only. Its default `build-only` mode:

1. checks out the branch, commit SHA, or tag selected by `source_ref` and validates the release matrix;
2. runs the project source checks;
3. builds eight Linux wheels and eight Windows wheels on separate runners;
4. runs minimal artifact and native-operator checks;
5. prepares one combined static-repository candidate;
6. preserves the candidate as a workflow artifact.

`build_scope=windows-only` resolves the source identity, validates the release matrix, and runs all eight Windows cells. It skips project verification, Linux wheels, release-candidate preparation, and publication. This option builds the Windows subset independently.

Protected `publish` mode is a separate operator choice. It requires an existing release tag matching the project version and `build_scope=all`. It publishes immutable R2 objects, verifies the PyPI source distribution, and updates cumulative indexes. Separate credential-free Linux and Windows jobs then install every declared wheel from its public index and execute CPU or CUDA operations. Once both jobs pass, the workflow preserves the source distribution, release manifest, and generated installation-catalog patch as a workflow artifact. The generated installation catalog is applied after the corresponding public artifacts are available, so installation selectors reference published wheel recipes. FHElium 0.10.0 remains immutable; Windows wheels begin with FHElium 0.20.0.
