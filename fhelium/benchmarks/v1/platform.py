"""Best-effort platform and FHElium build provenance collection."""

from __future__ import annotations

import ctypes
import json
import math
import os
import platform
import re
import subprocess
import sys
import sysconfig
from collections.abc import Callable, Mapping, Sequence
from enum import Enum
from importlib import metadata
from pathlib import Path
from typing import Any, TypeVar

import fhelium

from .model import FHEliumBuildIdentity, PlatformSnapshot, ProbeError

_ALLOWED_ENVIRONMENT = (
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "CUDA_MODULE_LOADING",
    "CUDA_LAUNCH_BLOCKING",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "CUDA_CACHE_DISABLE",
    "CUDA_FORCE_PTX_JIT",
    "NCCL_DEBUG",
    "NCCL_DEBUG_SUBSYS",
    "NCCL_P2P_DISABLE",
    "NCCL_IB_DISABLE",
    "NCCL_SHM_DISABLE",
    "NCCL_SOCKET_IFNAME",
    "NCCL_ALGO",
    "NCCL_PROTO",
    "NCCL_MIN_NCHANNELS",
    "NCCL_MAX_NCHANNELS",
    "NCCL_LAUNCH_MODE",
    "NCCL_NET",
    "NCCL_CUMEM_ENABLE",
)
_SENSITIVE_ARGUMENT = re.compile(
    r"(?i)(?:api[-_]?key|authorization|credential|passwd|password|secret|token)"
)
_URL_CREDENTIALS = re.compile(r"(://)[^/@\s]+@")
_T = TypeVar("_T")


def _portable_text(value: str) -> str:
    """Redact URL credentials and host-specific home paths in text."""

    result = _URL_CREDENTIALS.sub(r"\1<redacted>@", value)
    home = str(Path.home())
    if home and home != "/":
        result = re.sub(
            re.escape(home) + r"(?=$|[/\\])",
            "~",
            result,
        )
    return result


def _collected_json(value: Any, *, path: str = "probe") -> Any:
    """Normalize common probe-return types to finite JSON data."""

    if value is None or type(value) in (bool, int):
        return value
    if isinstance(value, str):
        return _portable_text(value)
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Path):
        return value.name
    if isinstance(value, Enum):
        return _collected_json(value.value, path=path)
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = str(key)
            if normalized_key in result:
                raise ValueError(
                    f"{path} has keys that collide after string conversion"
                )
            result[normalized_key] = _collected_json(
                item, path=f"{path}.{normalized_key}"
            )
        return result
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [
            _collected_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]

    # NumPy and Torch scalar values expose item(); avoid importing NumPy here.
    item = getattr(value, "item", None)
    if callable(item):
        scalar = item()
        if scalar is not value:
            return _collected_json(scalar, path=path)
    raise TypeError(f"{path} contains unsupported {type(value).__name__}")


def _error(probe: str, error: BaseException) -> ProbeError:
    return ProbeError(
        probe=probe,
        error_type=type(error).__name__,
        message=_portable_text(str(error)) or "(exception carried no message)",
    )


def _probe(
    name: str,
    function: Callable[[], _T],
    errors: list[ProbeError],
    default: _T,
) -> _T:
    try:
        return function()
    except Exception as error:  # platform collection must remain best effort
        errors.append(_error(name, error))
        return default


def sanitize_invocation(arguments: Sequence[str]) -> tuple[str, ...]:
    """Remove credential-like values and host-specific home-directory paths."""

    sanitized: list[str] = []
    redact_next = False
    for index, original in enumerate(arguments):
        argument = str(original)
        if index == 0:
            argument = Path(argument).name or "<program>"
        if redact_next:
            sanitized.append("<redacted>")
            redact_next = False
            continue
        if _SENSITIVE_ARGUMENT.search(argument):
            if "=" in argument:
                key, _separator, _value = argument.partition("=")
                sanitized.append(f"{key}=<redacted>")
            elif argument.startswith("-"):
                sanitized.append(argument)
                redact_next = True
            else:
                sanitized.append("<redacted>")
            continue
        argument = _portable_text(argument)
        sanitized.append(argument or "<empty>")
    return tuple(sanitized)


def _system() -> dict[str, Any]:
    uname = platform.uname()
    return {
        "system": uname.system,
        "release": uname.release,
        "version": uname.version,
        "machine": uname.machine,
        "node": "<redacted>",
        "platform": platform.platform(),
    }


def _cpu() -> dict[str, Any]:
    processor = platform.processor()
    model = ""
    physical_cores: set[tuple[str, str]] = set()
    package_ids: set[str] = set()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        records = (
            cpuinfo.read_text(encoding="utf-8", errors="replace")
            .strip()
            .split("\n\n")
        )
        for record in records:
            fields = {
                line.partition(":")[0].strip().lower(): line.partition(":")[
                    2
                ].strip()
                for line in record.splitlines()
                if ":" in line
            }
            model = model or fields.get("model name", "")
            package_id = fields.get("physical id")
            core_id = fields.get("core id")
            if package_id is not None:
                package_ids.add(package_id)
            if package_id is not None and core_id is not None:
                physical_cores.add((package_id, core_id))
    physical_core_count = len(physical_cores) or None
    package_count = len(package_ids) or None
    if platform.system() == "Darwin":

        def sysctl(name: str) -> str:
            completed = subprocess.run(
                ["sysctl", "-n", name],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            return completed.stdout.strip() if completed.returncode == 0 else ""

        model = sysctl("machdep.cpu.brand_string") or model
        physical = sysctl("hw.physicalcpu")
        packages = sysctl("hw.packages")
        physical_core_count = int(physical) if physical.isdigit() else None
        package_count = int(packages) if packages.isdigit() else None
    return {
        "logical_count": os.cpu_count(),
        "physical_core_count": physical_core_count,
        "package_count": package_count,
        "model": model or processor or platform.machine(),
        "architecture": platform.machine(),
    }


def _memory() -> dict[str, Any]:
    if sys.platform == "win32":
        from ctypes import wintypes

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", wintypes.DWORD),
                ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        global_memory_status = ctypes.WinDLL(
            "kernel32", use_last_error=True
        ).GlobalMemoryStatusEx
        global_memory_status.argtypes = (ctypes.POINTER(_MemoryStatusEx),)
        global_memory_status.restype = wintypes.BOOL
        if not global_memory_status(ctypes.byref(status)):
            error = ctypes.get_last_error()
            raise ctypes.WinError(error)

        total_bytes = int(status.ullTotalPhys)
        available_bytes = int(status.ullAvailPhys)
        if total_bytes <= 0:
            raise ValueError(
                "GlobalMemoryStatusEx returned non-positive physical memory"
            )
        if not 0 <= available_bytes <= total_bytes:
            raise ValueError(
                "GlobalMemoryStatusEx returned physical-memory values outside "
                "the valid range"
            )
        return {
            "total_bytes": total_bytes,
            "available_bytes": available_bytes,
        }

    page_size = os.sysconf("SC_PAGE_SIZE")
    physical_pages = os.sysconf("SC_PHYS_PAGES")
    available_pages = None
    try:
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
    except (OSError, ValueError):
        pass
    return {
        "total_bytes": page_size * physical_pages,
        "available_bytes": (
            None if available_pages is None else page_size * available_pages
        ),
    }


def _python() -> dict[str, Any]:
    return {
        "version": platform.python_version(),
        "implementation": sys.implementation.name,
        "compiler": platform.python_compiler(),
        "executable": Path(sys.executable).name,
        "soabi": sysconfig.get_config_var("SOABI") or "",
        "ext_suffix": sysconfig.get_config_var("EXT_SUFFIX") or "",
    }


def _distribution() -> dict[str, Any]:
    distribution = metadata.distribution("fhelium")
    return {
        "name": distribution.metadata.get("Name", "fhelium"),
        "version": distribution.version,
        "location": "<redacted>",
    }


def _source_root() -> Path:
    module_path = Path(fhelium.__file__).resolve()
    for parent in module_path.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return module_path.parent


def _git(root: Path, *arguments: str, allow_empty: bool = False) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(message or f"git exited {completed.returncode}")
    result = completed.stdout.strip()
    if not result and not allow_empty:
        raise RuntimeError("git returned no value")
    return result


def _sanitize_remote(value: str) -> str:
    return _URL_CREDENTIALS.sub(r"\1<redacted>@", value)


def _source_git() -> dict[str, Any]:
    root = _source_root()
    top_level = Path(_git(root, "rev-parse", "--show-toplevel")).resolve()
    commit = _git(top_level, "rev-parse", "HEAD")
    try:
        branch = _git(
            top_level,
            "symbolic-ref",
            "--quiet",
            "--short",
            "HEAD",
            allow_empty=True,
        )
    except RuntimeError:
        branch = ""
    status = _git(
        top_level,
        "status",
        "--porcelain",
        "--untracked-files=normal",
        allow_empty=True,
    )
    remote = ""
    try:
        remote = _sanitize_remote(
            _git(top_level, "config", "--get", "remote.origin.url")
        )
    except RuntimeError:
        pass
    return {
        "root": "<redacted>",
        "commit": commit,
        "branch": branch or None,
        "dirty": bool(status),
        "remote_origin": remote or None,
    }


def _native() -> dict[str, Any]:
    from fhelium.native import native_status

    status = native_status()
    manifest: Any = None
    if status.manifest_path.is_file():
        manifest = json.loads(status.manifest_path.read_text(encoding="utf-8"))
    return _collected_json(
        {
            "available": status.available,
            "reason": status.reason,
            "details": status.details,
            "binary_path": status.binary_path.name,
            "manifest_path": status.manifest_path.name,
            "backends": status.backends,
            "manifest": manifest,
        },
        path="native",
    )


def _torch() -> dict[str, Any]:
    import torch

    abi_query = getattr(torch, "compiled_with_cxx11_abi", None)
    cxx11_abi = (
        bool(abi_query())
        if callable(abi_query)
        else bool(torch._C._GLIBCXX_USE_CXX11_ABI)
    )
    nccl_version: Any = None
    nccl = getattr(torch.cuda, "nccl", None)
    if nccl is not None:
        version_query = getattr(nccl, "version", None)
        if callable(version_query):
            try:
                nccl_version = version_query()
            except Exception:
                nccl_version = None
    return _collected_json(
        {
            "version": torch.__version__,
            "cuda_build_version": torch.version.cuda,
            "cxx11_abi": cxx11_abi,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "nccl_version": nccl_version,
        },
        path="torch",
    )


def _cuda() -> dict[str, Any]:
    from fhelium.native.cuda import get_cuda_info

    return _collected_json(get_cuda_info(test_p2p_bandwidth=False), path="cuda")


def collect_platform(
    *,
    invocation: Sequence[str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> PlatformSnapshot:
    """Collect normalized reproducibility data without requiring every probe.

    Optional probe failures are represented in ``probe_errors``. CUDA device
    and P2P inventory deliberately disables the disruptive bandwidth test.
    Environment capture is limited to a fixed CUDA/NCCL allowlist.
    """

    errors: list[ProbeError] = []

    def json_probe(
        name: str, function: Callable[[], Mapping[str, Any]]
    ) -> dict[str, Any]:
        def collect() -> dict[str, Any]:
            value = _collected_json(function(), path=name)
            if not isinstance(value, dict):
                raise TypeError(f"{name} probe did not return an object")
            return value

        return _probe(name, collect, errors, {})

    system = json_probe("system", _system)
    cpu = json_probe("cpu", _cpu)
    memory = json_probe("memory", _memory)
    python = json_probe("python", _python)
    distribution = json_probe("distribution", _distribution)
    source_git = json_probe("source-git", _source_git)
    native = json_probe("native", _native)
    torch = json_probe("torch", _torch)
    cuda = json_probe("cuda", _cuda)

    source_environment = os.environ if environ is None else environ
    environment = {
        name: _portable_text(str(source_environment[name]))
        for name in _ALLOWED_ENVIRONMENT
        if name in source_environment
    }
    arguments = sys.argv if invocation is None else invocation

    return PlatformSnapshot(
        system=system,
        cpu=cpu,
        memory=memory,
        python=python,
        fhelium_build=FHEliumBuildIdentity(
            version=fhelium.__version__,
            distribution=distribution,
            source_git=source_git,
            native=native,
        ),
        torch=torch,
        cuda=cuda,
        environment=environment,
        invocation=sanitize_invocation(arguments),
        probe_errors=tuple(errors),
    )
