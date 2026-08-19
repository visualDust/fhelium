# Binary release architecture

This directory owns FHElium's declarative release matrix, Linux wheel build,
validation, and static Python package-index preparation.

## Distribution and hosting model

FHElium has one distribution name: `fhelium`.

- PyPI publishes `fhelium` source distributions for local compilation against
  the target environment's preinstalled Torch.
- Cloudflare R2 stores complete, precompiled `fhelium` wheels. GitHub Releases
  stores the release manifest and source distribution for release records. The
  current matrix uses the R2 host
  `download.fhelium.550w.host`.
- Four configuration-specific Simple Repository roots expose wheels through
  PEP 503 HTML and PEP 691 JSON projections.
- The documentation selector installs exact Torch first and then installs
  `fhelium` from the matching configuration index.

Standard wheel tags identify Python ABI, operating system, and machine
architecture. They do not identify Torch version, Torch CUDA variant, C++ ABI,
CUDA Toolkit, or GPU architecture. Incompatible wheels with the same FHElium
version must therefore remain in different Simple Repository roots:

```text
/torch213-cu130/simple/fhelium/
/torch212-cu129/simple/fhelium/
/torch213-cpu/simple/fhelium/
/torch212-cpu/simple/fhelium/
```

Each project page lists CPython 3.12 and 3.13 wheels. pip selects the compatible
Python ABI from ordinary wheel tags. The object layout is:

```text
/artifacts/<fhelium-version>/<configuration>/<wheel-filename>
/artifacts/<fhelium-version>/manifest.json
```

Published wheel objects are immutable. PEP 503 links contain `#sha256=` URL
fragments and `data-requires-python`. PEP 691 JSON records the same URL, digest,
and Python requirement. The release manifest owns URLs, sizes, hashes,
configurations, Python ABIs, and publication state.

A static host serves the PEP 503 `index.html` files directly. PEP 691 clients
request JSON through content negotiation at the same project URL; implementing
that negotiation is owned by the read-only Cloudflare Worker in `cloudflare/`.
The Worker maps trailing-slash Simple Repository routes to `index.html` or
`index.json` according to the request `Accept` header and reads all objects
through a private R2 binding. It has no write route or upload credential.
`prepare_static_index.py` generates both representations from the same
manifest.

## Release matrix

`release_matrix.json` is the machine-readable source of truth.
`release_matrix.schema.json` defines its structural schema, and
`release_matrix.py` enforces semantic invariants and projects the documentation
catalog. The matrix defines four Linux configurations for CPython 3.12 and
3.13:

| Configuration | Torch | CUDA Toolkit | Native backends |
| --- | --- | --- | --- |
| `torch213-cu130` | `2.13.0+cu130` | 13.0.3 | CPU+CUDA |
| `torch212-cu129` | `2.12.1+cu129` | 12.9.1 | CPU+CUDA |
| `torch213-cpu` | `2.13.0+cpu` | none | CPU |
| `torch212-cpu` | `2.12.1+cpu` | none | CPU |

CUDA architectures are matrix-owned release inputs: SM 80, 86, 89, 90, 100,
and 120. They are not a user installation choice.

```bash
python packaging/release_matrix.py validate
python packaging/release_matrix.py list
python packaging/release_matrix.py show torch213-cu130
python packaging/release_matrix.py docs-catalog \
  --project-version 0.10.0 \
  --output docs/.vitepress/theme/data/install-catalog.json
```

The documentation catalog records publication state for each binary recipe.
Only recipes with `published: true` appear as prebuilt installation methods.
The release workflow emits a catalog whose four validated binary recipes are
enabled; that catalog is the input to the documentation deployment associated
with the release.

## Linux wheel build

`build_release.py` selects the matrix-owned builder and invokes
`build_manylinux_wheel.py` for one configuration and Python ABI:

```bash
python packaging/build_release.py \
  --configuration torch213-cu130 \
  --python-abi cp313-cp313 \
  --output wheelhouse \
  --build-image \
  --smoke
```

The harness installs exact matrix Torch, validates runtime identity, builds one
complete `fhelium` wheel without PEP 517 isolation, keeps Torch/c10 and CUDA
runtime libraries external, repairs the wheel with auditwheel, validates ELF
and manifest metadata, records an exact `Requires-Dist: torch==...` matching
the configuration, and optionally runs an isolated native CPU smoke. CUDA
configurations additionally require a real GPU smoke when Docker GPU
passthrough is unavailable.

## Static index preparation

Collect final wheels under one configuration directory each. The preparer
accepts either direct wheel files or builder-style `<abi>/wheelhouse/` trees:

```text
release-input/
  torch213-cu130/
    cp312-cp312/wheelhouse/<wheel>
    cp313-cp313/wheelhouse/<wheel>
  torch212-cu129/...
  torch213-cpu/...
  torch212-cpu/...
```

Prepare and validate a release repository tree:

```bash
python packaging/prepare_static_index.py \
  --input release-input \
  --output prepared-site \
  --prepared-at 2026-08-13T20:00:00Z
python packaging/merge_published_indexes.py prepared-site
python packaging/validate_static_index.py prepared-site
```

The preparation command copies wheel files, computes SHA-256 digests, reads
`Requires-Python` from wheel metadata, and generates:

- PEP 503 root and project HTML pages;
- PEP 691 root and project JSON documents;
- one release manifest with `published: true`, also copied to the tree root;
- one installation catalog with the four binary recipes enabled;
- the immutable artifact directory layout.

`prepare_static_index.py` is deterministic for a supplied `--prepared-at`
value, or for `SOURCE_DATE_EPOCH`, and performs no network operations.
Publication is owned by
`publish_static_index.py`; `verify_public_release.py` retrieves the version
manifest and every wheel from their public URLs and validates byte sizes and
SHA-256 identities before the index pages are published.
`validate_repository_routing.py` verifies the canonical trailing-slash HTML
routes and PEP 691 content negotiation before a normal release changes public
objects.
`merge_published_indexes.py` reads each currently published PEP 691 project
document and merges its wheel records into the new project page, preserving
older releases. A missing public project document is accepted only during
explicit repository initialization; normal releases require every project
document and retain all of its validated records.

## Release workflow

`.github/workflows/release.yml` is the release entry point. It is manually
dispatched with an existing annotated or lightweight release tag. The tag must
equal `v<project.version>`, point at the checked-out commit, and contain the
release matrix selected for that version.
`initialize_repository` is selected only when establishing empty Simple
Repository roots. Normal releases leave it false and fail if any existing
project index cannot be retrieved.
`prerelease` marks the resulting GitHub Release as a pre-release; it does not
change the Python package version published to PyPI.

The `build-and-prepare` job runs on the trusted self-hosted Linux x86-64 worker
labeled `fhelium-release` and `gpu`. It performs source validation, builds all
eight wheels, runs isolated CPU smokes and real CUDA execution, builds the
sdist, prepares the static repository, and preserves the release bundle as a
GitHub Actions artifact. The `publish` job uses the protected `release`
environment and starts only after its required approval. It publishes and
verifies immutable R2 objects, publishes and verifies the sdist through PyPI
Trusted Publishing, publishes the four index roots, and verifies each public
index through pip. A final job with no PyPI OIDC or R2 credentials creates the
GitHub Release record.

Before dispatching a release, update `project.version`, the release matrix,
the checked-in source-install catalog, release-facing documentation, and
`CITATION.cff` together; validate the complete tree; create the version commit;
and push a `v<project.version>` tag for that exact commit. The workflow never
chooses or changes a Torch/CUDA matrix.

The self-hosted worker requires Docker, CPython 3.12 and 3.13, NVIDIA driver
access, `gh`, and npm. Ruff, Pyright, pytest, Python build, Twine, and the AWS
client are installed at their workflow-owned versions. The protected release
environment supplies secrets `R2_ACCESS_KEY_ID` and `R2_SECRET_ACCESS_KEY` plus
variables `R2_ENDPOINT_URL` and `R2_BUCKET`. PyPI authorizes this workflow
through OIDC; no PyPI API token is stored on the worker. After public resolver
validation, the workflow includes a Git patch in the GitHub Release that
enables the released binary recipes in the documentation catalog. Applying
that patch to `main` enables prebuilt commands and triggers the independent
Vercel documentation deployment.

## Installation

A published CUDA configuration produces commands of this form:

```bash
python -m pip install \
  --index-url https://download.pytorch.org/whl/cu130 \
  "torch==2.13.0+cu130"

python -m pip install --only-binary=fhelium \
  --extra-index-url \
  https://download.fhelium.550w.host/torch213-cu130/simple/ \
  "fhelium==0.10.0"
```

PyPI remains the primary index for ordinary dependencies; the trusted FHElium
index contributes only FHElium wheels. `--only-binary=fhelium` prevents the
prebuilt route from silently falling back to the PyPI sdist.

## Release validation and ordering

Every configuration and Python ABI must pass:

1. matrix schema and semantic validation;
2. exact Torch artifact and Toolkit-package availability checks;
3. builder image digest, Python SOABI, Torch runtime, Torch CUDA, C++ ABI, and
   Toolkit major/minor checks;
4. wheel filename/tag, native manifest, backend set, `DT_NEEDED`, RUNPATH, and
   build-path checks;
5. fresh-environment installation after exact Torch;
6. CPU native operation smoke and real CUDA execution for CUDA wheels;
7. sdist, static checks, native tests, and release provenance;
8. static HTML/JSON/manifest URL, hash, size, and coverage validation;
9. canonical Simple Repository HTML/JSON routing checks; and
10. resolver execution for every documentation install recipe.

Publication order is artifacts first and indexes last:

1. build and validate all promised wheels;
2. prepare HTML, JSON, hashes, and the version manifest;
3. upload wheel objects under immutable keys;
4. retrieve public URLs and verify their hashes;
5. publish and verify the source distribution on PyPI;
6. publish the configuration index pages;
7. verify the four public indexes with pip; and
8. prepare the documentation-catalog update and GitHub Release record.

A broken artifact is withdrawn with PEP 503 `data-yanked` and PEP 691 `yanked`
metadata. A published wheel is never replaced in place.

## Source builds

Source builds use target-environment Torch and the selected native backends:

```bash
python -m pip install "scikit-build-core>=1.0.3"
CMAKE_ARGS="-DFHELIUM_NATIVE_BACKENDS=CPU+CUDA" \
  python -m pip install --no-binary=fhelium \
    --no-build-isolation --no-cache-dir --verbose "fhelium==0.10.0"
```

CUDA source builds require the Toolkit major version to match Torch CUDA; the
same minor version is preferred. CPU-only builds select
`FHELIUM_NATIVE_BACKENDS=CPU` and require no CUDA Toolkit.

The sdist metadata retains the supported Torch range. The build backend marks
dependencies as dynamic between sdist and wheel so formal wheels can replace
that range with their exact matrix Torch local version.
