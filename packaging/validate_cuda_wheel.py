#!/usr/bin/env python3
"""Install one configured wheel in a clean environment and execute CUDA."""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType


def executable(value: str) -> Path:
    resolved = shutil.which(value)
    if resolved is None:
        raise argparse.ArgumentTypeError(
            f"Python interpreter is not executable: {value}"
        )
    return Path(resolved).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--python", type=executable, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def load_matrix_module() -> ModuleType:
    path = Path(__file__).with_name("release_matrix.py")
    spec = importlib.util.spec_from_file_location(
        "_fhelium_release_matrix", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[spec.name]
    return module


def run(*command: str) -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("LD_LIBRARY_PATH", None)
    subprocess.run(command, check=True, env=environment, cwd="/")


def main() -> None:
    args = parse_args()
    wheel = args.wheel.resolve()
    configuration = (
        load_matrix_module()
        .load_release_matrix()
        .configuration(args.configuration)
    )
    if not configuration.has_cuda:
        raise ValueError(f"{configuration.id} is not a CUDA configuration")
    with tempfile.TemporaryDirectory(
        prefix=f"fhelium-{configuration.id}-cuda-smoke-"
    ) as temporary:
        environment = Path(temporary) / "venv"
        run(str(args.python), "-m", "venv", str(environment))
        python = environment / "bin" / "python"
        run(
            str(python),
            "-m",
            "pip",
            "install",
            configuration.torch_requirement,
            "--index-url",
            configuration.torch_index_url,
        )
        run(
            str(python),
            "-m",
            "pip",
            "install",
            str(wheel),
            "--extra-index-url",
            configuration.torch_index_url,
        )
        code = f"""
import json
import torch
import fhelium
from fhelium.native import native_status
from fhelium.native.cuda import get_cuda_device_properties

assert str(torch.__version__) == {configuration.torch_runtime_version!r}
assert torch.version.cuda == {configuration.torch_cuda_version!r}
status = native_status()
assert status.available, status
assert set(status.backends) == {{'cpu', 'cuda'}}, status
device = torch.device({args.device!r})
lhs = torch.tensor([[[1, 2, 3, 4]]], dtype=torch.int64, device=device)
parameters = torch.zeros((8, 1), dtype=torch.int64, device=device)
parameters[0, 0] = 34
actual = torch.ops.fhelium_rns_ops.add_canonical(lhs, lhs, parameters)
expected = torch.tensor([[[2, 4, 6, 8]]], dtype=torch.int64)
assert torch.equal(actual.cpu(), expected)
print(json.dumps({{
    'torch': str(torch.__version__),
    'torch_cuda': torch.version.cuda,
    'device': str(device),
    'devices': get_cuda_device_properties(),
    'native': str(status),
}}, default=str))
"""
        run(str(python), "-I", "-c", code)


if __name__ == "__main__":
    main()
