"""Resolve runtime dependencies for source and formal wheel builds."""

from __future__ import annotations

import os
from collections.abc import Mapping


# The supported triton-csprng 0.1 series publishes platform-neutral wheels.
# Its base metadata does not request the `triton` distribution; that
# requirement belongs only to its optional cuda/dev extras. On Windows,
# triton-windows supplies the importable `triton` implementation under a
# different distribution name. The pinned 3.7.1.post27 release publishes both
# cp312 and cp313 win_amd64 wheels and is the identity validated by the Windows
# release/install checks.
DEPENDENCIES = [
    "click>=8.4.2",
    "mpmath>=1.3.0,<1.4",
    "numpy>=2.5.1",
    "pydot>=4.0.1,<5",
    "rich>=15.0.0",
    "safetensors>=0.8.0",
    "textual>=8.2.8",
    "torch>=2.10,<2.14",
    "triton-csprng>=0.1.4,<0.2",
    "triton-windows==3.7.1.post27; sys_platform == 'win32'",
    "xdsl>=0.68,<0.69",
]


def dynamic_metadata(
    settings: Mapping[str, object], project: Mapping[str, object]
) -> dict[str, list[str]]:
    """Return runtime requirements with a pinned Torch release identity."""

    if settings != {"field": "dependencies"}:
        raise ValueError(f"invalid dynamic metadata settings: {settings!r}")
    requirement = os.environ.get("FHELIUM_RELEASE_TORCH_REQUIREMENT")
    if requirement is None:
        return {"dependencies": DEPENDENCIES}
    dependencies = [
        requirement if item.startswith("torch>=") else item
        for item in DEPENDENCIES
    ]
    return {"dependencies": dependencies}


def dynamic_wheel(settings: Mapping[str, object]) -> dict[str, bool]:
    """Declare that wheel dependencies may differ from sdist metadata."""

    if settings != {"field": "dependencies"}:
        raise ValueError(f"invalid dynamic metadata settings: {settings!r}")
    return {"dependencies": True}
