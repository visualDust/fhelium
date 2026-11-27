"""Resolve runtime dependencies for source and formal wheel builds."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping


# FHElium's Triton backend imports `triton`. Linux CUDA Torch wheels already
# request it; on Windows, triton-windows supplies the importable `triton`
# implementation under a different distribution name. The 3.8 and 3.9 series
# match the Triton releases required by PyTorch 2.14 and 2.15 and publish
# win_amd64 wheels for CPython 3.12 and 3.13.
DEPENDENCIES = [
    "click>=8.4.2",
    "mpmath>=1.3.0,<1.4",
    "numpy>=2.5.1",
    "pydot>=4.0.1,<5",
    "psutil>=6.1,<8",
    "rich>=15.0.0",
    "safetensors>=0.8.0",
    "textual>=8.2.8",
    "torch>=2.14,<2.16",
    "triton-windows>=3.8,<3.10; sys_platform == 'win32'",
    "xdsl>=0.68,<0.69",
]


def dynamic_metadata(
    settings: Mapping[str, object], project: Mapping[str, object]
) -> dict[str, list[str]]:
    """Return runtime requirements with a pinned Torch release identity."""

    if settings != {"field": "dependencies"}:
        raise ValueError(f"invalid dynamic metadata settings: {settings!r}")
    requirement = os.environ.get("FHELIUM_RELEASE_TORCH_REQUIREMENT")
    dependencies = [
        requirement if requirement and item.startswith("torch>=") else item
        for item in DEPENDENCIES
    ]
    runtime_requirements = json.loads(
        os.environ.get("FHELIUM_RELEASE_RUNTIME_REQUIREMENTS", "[]")
    )
    if not isinstance(runtime_requirements, list) or any(
        not isinstance(item, str) or not item for item in runtime_requirements
    ):
        raise ValueError(
            "Release runtime requirements must be a JSON string array"
        )
    return {"dependencies": [*dependencies, *runtime_requirements]}


def dynamic_wheel(settings: Mapping[str, object]) -> dict[str, bool]:
    """Declare that wheel dependencies may differ from sdist metadata."""

    if settings != {"field": "dependencies"}:
        raise ValueError(f"invalid dynamic metadata settings: {settings!r}")
    return {"dependencies": True}
