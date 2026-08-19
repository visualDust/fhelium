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

# number of build threads (override with `just install BUILD_JOBS=16`)
BUILD_JOBS := `nproc`

# native backends: AUTO, CPU, CUDA, or CPU+CUDA
NATIVE_BACKENDS := "AUTO"

build: install # the supported developer build is an editable install

install:
    CMAKE_BUILD_PARALLEL_LEVEL={{BUILD_JOBS}} \
    CMAKE_ARGS="-DFHELIUM_NATIVE_BACKENDS={{NATIVE_BACKENDS}}" \
    pip install --editable . --verbose --no-build-isolation --no-cache-dir

install-uv:
    CMAKE_BUILD_PARALLEL_LEVEL={{BUILD_JOBS}} \
    CMAKE_ARGS="-DFHELIUM_NATIVE_BACKENDS={{NATIVE_BACKENDS}}" \
    uv pip install --editable . --verbose --no-build-isolation --no-cache

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
    ruff check fhelium tests examples scripts
    ruff format --check fhelium tests examples scripts

typecheck:
    pyright

test:
    pytest -q

check: lint typecheck test

pre-commit:
    pre-commit run --all-files


trace:
    TORCH_LOGS="graph_breaks" fhelium benchmark
