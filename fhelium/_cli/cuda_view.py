"""CUDA device inventory and peer-topology rendering for the CLI."""

from __future__ import annotations

from typing import Any

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table


def _bytes(value: float) -> str:
    amount = float(value)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    unit = units[0]
    for candidate in units:
        unit = candidate
        if abs(amount) < 1024.0 or candidate == units[-1]:
            break
        amount /= 1024.0
    return f"{amount:.2f} {unit}"


def _mhz(value: float) -> str:
    return f"{float(value) / 1000.0:,.0f} MHz"


def _boolean(value: Any) -> str:
    return "[green]yes[/green]" if value else "[dim]no[/dim]"


def _device_summary_table(devices: dict[int, dict[str, Any]]) -> Table:
    table = Table(expand=True, show_lines=False)
    table.add_column("GPU", style="bold cyan", justify="right", no_wrap=True)
    table.add_column("Name", overflow="ellipsis", no_wrap=True, ratio=1)
    table.add_column("CC", justify="center")
    table.add_column("SMs", justify="right")
    table.add_column("Memory", justify="right")
    for device_id, props in devices.items():
        table.add_row(
            str(device_id),
            str(props.get("name", "unknown")),
            str(props.get("computeCapability", "?")),
            str(props.get("multiProcessorCount", "?")),
            _bytes(props.get("totalGlobalMem", 0)),
        )
    return table


def _property_table(title: str, rows: tuple[tuple[str, Any], ...]) -> Panel:
    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column(style="cyan", no_wrap=True)
    table.add_column(justify="right")
    for name, value in rows:
        table.add_row(name, str(value))
    return Panel(table, title=title, border_style="dim")


def _device_detail(device_id: int, props: dict[str, Any]) -> Panel:
    compute = _property_table(
        "Compute",
        (
            ("Compute capability", props.get("computeCapability", "?")),
            ("Multiprocessors", props.get("multiProcessorCount", "?")),
            ("Core clock", _mhz(props.get("clockRate", 0))),
            ("Warp size", props.get("warpSize", "?")),
            ("Threads / block", props.get("maxThreadsPerBlock", "?")),
            (
                "Threads / SM",
                props.get("maxThreadsPerMultiProcessor", "?"),
            ),
            ("Async engines", props.get("asyncEngineCount", "?")),
        ),
    )
    memory = _property_table(
        "Memory",
        (
            ("Global memory", _bytes(props.get("totalGlobalMem", 0))),
            ("L2 cache", _bytes(props.get("l2CacheSize", 0))),
            (
                "Shared / block",
                _bytes(props.get("sharedMemPerBlock", 0)),
            ),
            (
                "Shared / SM",
                _bytes(props.get("sharedMemPerMultiprocessor", 0)),
            ),
            ("Bus width", f"{props.get('memoryBusWidth', '?')} bit"),
            ("Memory clock", _mhz(props.get("memoryClockRate", 0))),
        ),
    )
    features = _property_table(
        "Runtime features",
        (
            ("Unified addressing", _boolean(props.get("unifiedAddressing"))),
            ("Managed memory", _boolean(props.get("managedMemory"))),
            (
                "Concurrent managed access",
                _boolean(props.get("concurrentManagedAccess")),
            ),
            ("Map host memory", _boolean(props.get("canMapHostMemory"))),
            ("Device overlap", _boolean(props.get("deviceOverlap"))),
            (
                "Cooperative launch",
                _boolean(props.get("cooperativeLaunch")),
            ),
        ),
    )
    return Panel(
        Group(compute, memory, features),
        title=f"GPU {device_id} · {props.get('name', 'unknown')}",
        border_style="cyan",
    )


def render_device_info(
    console: Console,
    devices: dict[int, dict[str, Any]],
    *,
    detail_device: int | None = None,
) -> None:
    total_memory = sum(
        int(props.get("totalGlobalMem", 0)) for props in devices.values()
    )
    console.print(
        Panel(
            f"[bold]{len(devices)} CUDA device(s)[/bold]  ·  "
            f"aggregate memory {_bytes(total_memory)}",
            title="FHElium CUDA inventory",
            border_style="cyan",
        )
    )
    console.print(_device_summary_table(devices))
    if detail_device is not None:
        console.print(_device_detail(detail_device, devices[detail_device]))


def _connectivity_table(p2p: dict[str, Any]) -> Table:
    count = int(p2p.get("numDevices", 0))
    access = p2p.get("canAccess", [])
    table = Table(title="Peer access", expand=False)
    table.add_column("from \\ to", style="cyan", no_wrap=True)
    for destination in range(count):
        table.add_column(f"GPU {destination}", justify="center")
    for source in range(count):
        row = [f"GPU {source}"]
        for destination in range(count):
            if source == destination:
                row.append("[dim]local[/dim]")
            else:
                available = bool(access[source][destination])
                row.append(
                    "[bold green]P2P[/bold green]"
                    if available
                    else "[bold red]host[/bold red]"
                )
        table.add_row(*row)
    return table


def _matrix_value(matrix, source: int, destination: int) -> float:
    try:
        return float(matrix[source][destination])
    except (IndexError, TypeError, ValueError):
        return 0.0


def _bandwidth_table(p2p: dict[str, Any]) -> Table | None:
    count = int(p2p.get("numDevices", 0))
    uni_host = p2p.get("unidirectionalBandwidthNoP2P", [])
    uni_p2p = p2p.get("unidirectionalBandwidthP2P", [])
    bi_host = p2p.get("bidirectionalBandwidthNoP2P", [])
    bi_p2p = p2p.get("bidirectionalBandwidthP2P", [])
    if not any((uni_host, uni_p2p, bi_host, bi_p2p)):
        return None
    if not any(
        _matrix_value(matrix, source, destination) > 0
        for matrix in (uni_host, uni_p2p, bi_host, bi_p2p)
        for source in range(count)
        for destination in range(count)
        if source != destination
    ):
        return None

    table = Table(title="Measured link bandwidth", expand=True)
    table.add_column("link", style="cyan")
    table.add_column("host staged →", justify="right")
    table.add_column("P2P →", justify="right")
    table.add_column("host staged ↔", justify="right")
    table.add_column("P2P ↔", justify="right")
    for source in range(count):
        for destination in range(source + 1, count):
            table.add_row(
                f"GPU {source} ↔ GPU {destination}",
                f"{_matrix_value(uni_host, source, destination):.2f} GB/s",
                f"{_matrix_value(uni_p2p, source, destination):.2f} GB/s",
                f"{_matrix_value(bi_host, source, destination):.2f} GB/s",
                f"{_matrix_value(bi_p2p, source, destination):.2f} GB/s",
            )
    return table


def render_topology(
    console: Console,
    devices: dict[int, dict[str, Any]],
    p2p: dict[str, Any],
    *,
    bandwidth_requested: bool,
) -> None:
    count = int(p2p.get("numDevices", len(devices)))
    access = p2p.get("canAccess", [])
    directed_links = sum(
        bool(access[source][destination])
        for source in range(count)
        for destination in range(count)
        if source != destination
    )
    possible = count * max(0, count - 1)
    console.print(
        Panel(
            f"[bold]{count} CUDA device(s)[/bold]  ·  "
            f"P2P links {directed_links}/{possible}  ·  "
            f"bandwidth {'measured' if bandwidth_requested else 'not requested'}",
            title="FHElium CUDA topology",
            border_style="magenta",
        )
    )
    console.print(_connectivity_table(p2p))
    bandwidth = _bandwidth_table(p2p)
    if bandwidth is not None:
        console.print(bandwidth)
    elif bandwidth_requested:
        console.print(
            Panel(
                "The native probe returned no non-zero bandwidth samples.",
                border_style="yellow",
            )
        )
    else:
        console.print(
            "[dim]Use --bandwidth to run the native transfer benchmark.[/dim]"
        )
