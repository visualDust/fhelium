clean-build: # .so in build directories, and build folder
    rm -rf build;
    rm -rf CMakeFiles;
    find fhelium/native -maxdepth 2 -type f -name "*.so" -exec rm -v {} \;
    find fhelium/native/torchops -maxdepth 1 -type f -name "_build_manifest.*.json" -exec rm -v {} \;
    echo "Cleaned up shared object files in fhelium/native"

clean-cache: # disposable developer-tool caches; never delete package resources
    rm -rf .ruff_cache .mypy_cache .pyright

clean-pycache: # all __pycache__ directories
    rm -rf .pytest_cache;
    find fhelium -type d -name "__pycache__" -exec rm -rv {} \;

clean-venv: # remove virtual environment folder
    rm -rf .venv;

clean-all: clean-build clean-cache clean-pycache clean-venv

# number of build threads (override with `just BUILD_JOBS=16 build-uv`)
BUILD_JOBS := `python -c "import os; print(os.cpu_count() or 1)"`

# native backends: AUTO, CPU, CUDA, or CPU+CUDA
NATIVE_BACKENDS := "AUTO"

export CMAKE_BUILD_PARALLEL_LEVEL := BUILD_JOBS
export CMAKE_ARGS := "-DFHELIUM_NATIVE_BACKENDS=" + NATIVE_BACKENDS

bootstrap-uv: # create the locked developer environment and editable native build
    uv --preview-features extra-build-dependencies sync --locked

build-uv: # rebuild FHElium inside the uv-managed environment
    uv --preview-features extra-build-dependencies sync --locked --reinstall-package fhelium

bootstrap-pip: # selected Torch must already be installed in the active environment
    python -m pip install --group build
    python -m pip install --editable . --verbose --no-build-isolation --no-cache-dir
    python -m pip install --group dev

build-pip: # rebuild FHElium against the active environment's selected Torch
    python -m pip install --editable . --verbose --no-build-isolation --no-cache-dir

generate-prime-catalog:
    python scripts/generate_prime_catalog.py --force

wrappers-generate:
    python scripts/generate_native_wrappers.py --verbose

wrappers-check:
    python scripts/generate_native_wrappers.py --check

docs-sync:
    npm --prefix docs ci

docs-serve:
    npm --prefix docs run dev

docs-build:
    npm --prefix docs run build

lint:
    ruff check .
    ruff format --check .

typecheck:
    pyright

test:
    pytest -q

check: lint typecheck test

pre-commit:
    pre-commit run --all-files


trace:
    TORCH_LOGS="graph_breaks" fhelium benchmark
