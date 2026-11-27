#!/usr/bin/env python3

"""Report the FHElium installation an agent is about to use.

Prints the FHElium, Python, PyTorch, and CUDA versions, native-library status, NTT
backends, GPUs, and presets. The same facts come from ``fhelium version`` and
``fhelium cuda info``.
"""

from __future__ import annotations

import platform
import sys


def main() -> int:
    try:
        import torch

        import fhelium as fh
        from fhelium.native import native_status
    except ImportError as error:
        print(f"fhelium import failed: {error}")
        return 1

    print(f"fhelium     {fh.__version__}")
    print(f"python      {platform.python_version()} ({sys.executable})")
    print(
        f"torch       {torch.__version__}, CUDA {torch.version.cuda}, "
        f"cxx11 ABI {torch.compiled_with_cxx11_abi()}"
    )

    status = native_status()
    print(f"native      available={status.available} ({status.reason})")
    for detail in status.details:
        print(f"            {detail}")
    print(f"backends    {', '.join(status.backends) or 'none'}")
    for preset in (
        fh.Preset.slots8192_scale40_depth7,
        fh.Preset.slots65536_scale40_depth75,
    ):
        log_n = fh.CkksConfig.parse(preset).logN
        backends = ", ".join(fh.compatible_ntt_backends(log_n))
        print(f"ntt N=2^{log_n:<3d} {backends}")

    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            print(
                f"cuda:{index}      {props.name}, sm_{props.major}{props.minor}, "
                f"{props.total_memory / 2**30:.1f} GiB"
            )
    else:
        print("cuda        not available")

    print("presets     " + ", ".join(preset.value for preset in fh.Preset))
    return 0 if status.available else 1


if __name__ == "__main__":
    sys.exit(main())
