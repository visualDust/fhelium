"""Small helper utilities shared by FHElium examples.

The examples intentionally avoid heavy third-party table/CLI dependencies so they
can be copied into notebooks or scripts easily.
"""

from __future__ import annotations

import argparse
import csv
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import torch

from fhelium import (
    DEFAULT_CPU_NTT_BACKEND,
    DEFAULT_NTT_BACKEND,
    SUPPORTED_NTT_BACKENDS,
    CkksEngine,
    Preset,
    compatible_ntt_backends,
)
from fhelium.config import CkksConfig


def preset_names() -> list[str]:
    return [preset.value for preset in Preset]


def parse_preset(name: str) -> Preset:
    try:
        return Preset(name)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"unknown preset {name!r}; choose one of {', '.join(preset_names())}"
        ) from exc


def add_engine_args(
    parser: argparse.ArgumentParser,
    *,
    default_preset: str = Preset.slots32768_scale40_levels34_int64.value,
    default_device: str = "cpu",
    include_num_scale_primes: bool = False,
) -> None:
    parser.add_argument(
        "--preset",
        choices=preset_names(),
        default=default_preset,
        help=f"CKKS parameter preset. Default for this example: {default_preset}.",
    )
    parser.add_argument(
        "--device",
        default=default_device,
        help=(
            "PyTorch execution device used by the example. "
            f"Default: {default_device}."
        ),
    )
    parser.add_argument(
        "--ntt-backend",
        choices=SUPPORTED_NTT_BACKENDS,
        default=None,
        help=(
            "Optional NTT backend name, e.g. radix16_compact. Strict "
            "fixed-radix backends reject incompatible logN values; omit to "
            "use the config default."
        ),
    )
    if include_num_scale_primes:
        parser.add_argument(
            "--num-scale-primes",
            type=int,
            default=None,
            help="Override the preset scale-prime and public-level count.",
        )


def make_engine(args: argparse.Namespace) -> CkksEngine:
    preset = parse_preset(args.preset)
    if getattr(args, "num_scale_primes", None) is not None:
        cfg = CkksConfig.parse(
            preset,
            num_scale_primes=args.num_scale_primes,
        )
    else:
        cfg = CkksConfig.parse(preset)
    device = torch.device(args.device)
    ntt_backend = args.ntt_backend or (
        DEFAULT_CPU_NTT_BACKEND if device.type == "cpu" else DEFAULT_NTT_BACKEND
    )
    if args.ntt_backend is not None:
        compatible = compatible_ntt_backends(cfg.logN)
        if args.ntt_backend not in compatible:
            raise ValueError(
                f"NTT backend {args.ntt_backend!r} is incompatible with "
                f"logN={cfg.logN}; compatible backends: {compatible!r}"
            )
    return CkksEngine(
        cfg,
        device=device,
        ntt_backend=ntt_backend,
    )


def sync_if_cuda(device: torch.device | str | None = None) -> None:
    if not torch.cuda.is_available():
        return
    if device is None:
        torch.cuda.synchronize()
        return
    selected = torch.device(device)
    if selected.type == "cuda":
        torch.cuda.synchronize(selected)


def time_ms(
    fn: Callable[[], Any],
    *,
    warmup: int = 1,
    runs: int = 5,
    device: torch.device | str | None = None,
) -> tuple[dict[str, float], Any]:
    """CUDA-synchronized timing helper returning summary stats and last result."""

    last = None
    for _ in range(warmup):
        last = fn()
        sync_if_cuda(device)

    times: list[float] = []
    for _ in range(runs):
        sync_if_cuda(device)
        start = time.perf_counter()
        last = fn()
        sync_if_cuda(device)
        times.append((time.perf_counter() - start) * 1e3)

    values = torch.tensor(times, dtype=torch.float64)
    return (
        {
            "mean_ms": float(values.mean()),
            "min_ms": float(values.min()),
            "max_ms": float(values.max()),
            "std_ms": float(values.std(correction=0)),
        },
        last,
    )


def as_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    return torch.as_tensor(x)


def error_stats(
    got: Any, expected: Any, slots: int | None = None
) -> dict[str, float]:
    got_tensor = as_tensor(got)
    expected_tensor = as_tensor(expected)
    if slots is not None:
        got_tensor = got_tensor[:slots]
        expected_tensor = expected_tensor[:slots]
    abs_diff = torch.abs(got_tensor - expected_tensor)
    ref_abs = torch.abs(expected_tensor)
    rms = float(torch.sqrt(torch.mean(abs_diff**2)))
    ref_rms = float(torch.sqrt(torch.mean(ref_abs**2)))
    max_abs = float(abs_diff.max())
    return {
        "max_abs": max_abs,
        "rms": rms,
        "mean_abs": float(abs_diff.mean()),
        "rel_max": max_abs / max(float(ref_abs.max()), 1.0),
        "rel_rms": rms / max(ref_rms, 1.0),
    }


def format_bytes(num_bytes: float | None) -> str:
    if num_bytes is None:
        return "n/a"
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0 or unit == "GiB":
            return (
                f"{value:.2f} {unit}" if unit != "B" else f"{int(value)} {unit}"
            )
        value /= 1024.0
    return f"{num_bytes} B"


def print_table(headers: Iterable[str], rows: Iterable[Iterable[Any]]) -> None:
    headers = list(headers)
    str_rows = [["" if v is None else str(v) for v in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in str_rows:
        widths = [max(w, len(v)) for w, v in zip(widths, row)]

    def fmt(row: Iterable[str]) -> str:
        return "  ".join(v.ljust(w) for v, w in zip(row, widths))

    print(fmt(headers))
    print(fmt(["-" * w for w in widths]))
    for row in str_rows:
        print(fmt(row))


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def small_complex_vector(
    slots: int, *, seed: int = 0, scale: float = 0.01
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    real = torch.randn(slots, generator=generator) * scale
    imag = torch.randn(slots, generator=generator) * scale
    return real + 1j * imag
