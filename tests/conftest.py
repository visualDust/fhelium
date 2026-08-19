"""Test import guard for local source checkouts.

This repository is often tested after another editable FHElium checkout was
installed in the same Python environment.  scikit-build's editable redirecting
finder is placed before ``PathFinder`` and can otherwise import that older
checkout instead of the current working tree.  Remove only that redirecting
finder so tests exercise the local source tree.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if (_REPO_ROOT / "fhelium" / "__init__.py").exists():
    sys.meta_path = [
        finder
        for finder in sys.meta_path
        if type(finder).__module__ != "_editable_skbc_fhelium"
    ]
    root_text = str(_REPO_ROOT)
    sys.path = [entry for entry in sys.path if entry != root_text]
    sys.path.insert(0, root_text)
    for module_name, module in tuple(sys.modules.items()):
        if module_name != "fhelium" and not module_name.startswith("fhelium."):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is not None and not Path(
            module_file
        ).resolve().is_relative_to(_REPO_ROOT):
            del sys.modules[module_name]

    import fhelium

    imported_package = Path(fhelium.__file__).resolve().parent
    expected_package = (_REPO_ROOT / "fhelium").resolve()
    if imported_package != expected_package:
        raise RuntimeError(
            "pytest imported FHElium from the wrong checkout: "
            f"{imported_package} != {expected_package}"
        )


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip hardware-marked tests when their runtime capability is absent."""

    from fhelium.native import native_backend_available

    cuda_ready = torch.cuda.is_available() and native_backend_available("cuda")
    gpu_skip = pytest.mark.skip(
        reason="requires visible CUDA and a CUDA-enabled FHElium native build"
    )
    multigpu_skip = pytest.mark.skip(
        reason="requires at least two visible CUDA devices"
    )
    try:
        pinned_probe = torch.empty(0, device="cpu", pin_memory=True)
        pinned_memory_ready = (
            pinned_probe.device.type == "cpu" and pinned_probe.is_pinned()
        )
    except RuntimeError:
        pinned_memory_ready = False
    pinned_memory_skip = pytest.mark.skip(
        reason="requires a PyTorch pinned-host allocator"
    )
    for item in items:
        if item.get_closest_marker("gpu") is not None and not cuda_ready:
            item.add_marker(gpu_skip)
        if item.get_closest_marker("multigpu") is not None and (
            not cuda_ready or torch.cuda.device_count() < 2
        ):
            item.add_marker(multigpu_skip)
        if (
            item.get_closest_marker("pinned_memory") is not None
            and not pinned_memory_ready
        ):
            item.add_marker(pinned_memory_skip)
