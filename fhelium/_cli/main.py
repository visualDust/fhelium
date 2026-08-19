import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from fhelium.config import Preset

console = Console()


@click.group()
@click.pass_context
def cli(ctx):
    ctx.ensure_object(dict)


@cli.command(name="version")
def version_command():
    """Print the version of fhelium"""
    import fhelium

    console.print(
        Panel(
            f"fhelium version: {fhelium.__version__}",
            title="Version",
            title_align="left",
            border_style="cyan",
        )
    )


@cli.group(name="benchmark", invoke_without_command=True)
@click.option(
    "--file",
    "benchmark_file",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    help="Load custom benchmark registrations from a Python file.",
)
@click.pass_context
def benchmark_group(ctx, benchmark_file: str | None) -> None:
    """Open the benchmark TUI or use a non-interactive subcommand."""

    from pathlib import Path

    if ctx.invoked_subcommand == "v1":
        if benchmark_file is not None:
            raise click.UsageError(
                "--file applies only to independent benchmark workloads, not "
                "the fixed Benchmark v1 specification"
            )
        ctx.ensure_object(dict)
        return

    from fhelium.benchmarks.cli import load_benchmark_file
    from fhelium.benchmarks.registry import load_builtin_benchmarks

    registry = load_builtin_benchmarks()
    if benchmark_file is not None:
        load_benchmark_file(Path(benchmark_file).resolve(), registry)
    ctx.ensure_object(dict)
    ctx.obj["benchmark_registry"] = registry

    if ctx.invoked_subcommand is None:
        from fhelium.benchmarks.tui import run_tui

        run_tui(registry)


@benchmark_group.command(name="list")
@click.pass_context
def benchmark_list(ctx) -> None:
    """List registered benchmarks and profiles."""

    registry = ctx.obj["benchmark_registry"]
    table = Table(title="FHElium benchmarks", expand=False)
    table.add_column("name", style="cyan", no_wrap=True)
    table.add_column("category")
    table.add_column("profiles")
    for definition in registry:
        table.add_row(
            definition.name,
            definition.category,
            ", ".join(profile.name for profile in definition.profiles),
        )
    console.print(table)


@benchmark_group.command(name="run")
@click.argument("benchmark_name")
@click.option("--profile", "profile_name", help="Named benchmark profile.")
@click.option(
    "--set",
    "overrides",
    multiple=True,
    help="Override one profile parameter as KEY=JSON_VALUE.",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=str),
    help="Write the structured result as JSON.",
)
@click.pass_context
def benchmark_run(
    ctx,
    benchmark_name: str,
    profile_name: str | None,
    overrides: tuple[str, ...],
    output: str | None,
) -> None:
    """Run one benchmark without opening the TUI."""

    from pathlib import Path

    from fhelium.benchmarks.cli import parse_overrides
    from fhelium.benchmarks.io import write_json_atomic
    from fhelium.benchmarks.render import print_result

    registry = ctx.obj["benchmark_registry"]
    try:
        definition = registry.get(benchmark_name)
        profile = definition.profile(profile_name).with_overrides(
            parse_overrides(overrides)
        )
    except (KeyError, ValueError) as error:
        raise click.ClickException(str(error)) from error

    def progress(message: str) -> None:
        console.print(f"[dim]{message}[/dim]")

    try:
        result = definition.runner(profile, progress)
    except Exception as error:
        raise click.ClickException(
            f"{type(error).__name__}: {error}"
        ) from error
    print_result(result, console=console)
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(path, result.to_dict())
        console.print(f"[green]Saved {path.resolve()}[/green]")


@benchmark_group.group(name="v1")
def benchmark_v1_group() -> None:
    """Run the immutable FHElium Benchmark v1 specification."""


@benchmark_v1_group.command(name="run")
@click.option(
    "--device",
    default="cpu",
    show_default=True,
    help="Execute every fixed v1 case on cpu or an indexed cuda:N device.",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=str),
    help="Write the checkpointed Benchmark v1 report to this JSON file.",
)
def benchmark_v1_run(device: str, output: str | None) -> None:
    """Run every fixed Benchmark v1 case on one selected local device."""

    import sys
    from datetime import UTC, datetime
    from pathlib import Path

    from fhelium.benchmarks.v1 import BenchmarkExecution, BenchmarkRunner
    from fhelium.benchmarks.v1.model import ExecutionBackend

    try:
        execution = BenchmarkExecution(
            backend=(
                ExecutionBackend.CPU
                if device == "cpu"
                else ExecutionBackend.CUDA
            ),
            device=device,
        )
    except (TypeError, ValueError) as error:
        raise click.UsageError(str(error)) from error

    if output is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output_path = Path(f"fhelium-benchmark-v1-{timestamp}.json")
    else:
        output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def progress(case_id: str, message: str) -> None:
        console.print(f"[dim][{case_id}] {message}[/dim]")

    try:
        report = BenchmarkRunner().run(
            execution=execution,
            output_path=output_path,
            invocation=tuple(sys.argv),
            progress=progress,
        )
    except Exception as error:
        raise click.ClickException(
            f"{type(error).__name__}: {error}"
        ) from error

    summary = Table(title=f"FHElium Benchmark v1 — {report.status.value}")
    summary.add_column("case")
    summary.add_column("category")
    summary.add_column("status")
    summary.add_column("detail")
    for case in report.cases:
        detail = ""
        if case.unavailable is not None:
            detail = case.unavailable.reason
        elif case.failure is not None:
            detail = case.failure.message
        summary.add_row(case.title, case.category, case.status.value, detail)
    console.print(summary)
    console.print(f"[green]Saved {output_path.resolve()}[/green]")
    if report.suggested_exit_code:
        raise click.ClickException(
            "The report was written, but one or more Benchmark v1 cases "
            "failed or were interrupted."
        )


@benchmark_group.group(name="recommend")
def benchmark_recommend_group() -> None:
    """Run benchmark workloads with measured execution recommendations that recommend execution policies."""


@benchmark_recommend_group.command(name="ntt")
@click.option(
    "--suite",
    type=click.Choice(("kernel", "ckks-primitive"), case_sensitive=True),
    default="kernel",
    show_default=True,
    help="Evidence tier used to rank compatible NTT backends.",
)
@click.option(
    "--preset",
    "preset_name",
    type=click.Choice(tuple(preset.value for preset in Preset)),
    default=Preset.slots32768_scale40_levels34_int64.value,
    show_default=True,
)
@click.option("--device", default="cuda:0", show_default=True)
@click.option(
    "--warmup", type=click.IntRange(min=0), default=3, show_default=True
)
@click.option(
    "--runs", type=click.IntRange(min=1), default=10, show_default=True
)
@click.option(
    "--repetitions",
    type=click.IntRange(min=1),
    default=3,
    show_default=True,
)
@click.option(
    "--backend",
    "backends",
    multiple=True,
    help="Restrict the comparison to exact backend names; repeat as needed.",
)
@click.option("--seed", type=int, default=20260723, show_default=True)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=str),
    help="Write the recommendation and its measurement evidence as JSON.",
)
def benchmark_recommend_ntt(
    suite: str,
    preset_name: str,
    device: str,
    warmup: int,
    runs: int,
    repetitions: int,
    backends: tuple[str, ...],
    seed: int,
    output: str | None,
) -> None:
    """Recommend an exact NTT backend without changing any default."""

    from pathlib import Path

    from fhelium.benchmarks.io import write_json_atomic
    from fhelium.benchmarks.render import print_result
    from fhelium.benchmarks.standalone.ntt_recommendation import (
        recommend_ntt_backend,
    )

    def progress(message: str) -> None:
        console.print(f"[dim]{message}[/dim]")

    try:
        result = recommend_ntt_backend(
            suite=suite,
            preset_name=preset_name,
            device=device,
            warmup=warmup,
            runs=runs,
            repetitions=repetitions,
            requested_backends=backends,
            seed=seed,
            progress=progress,
        )
    except Exception as error:
        raise click.ClickException(
            f"{type(error).__name__}: {error}"
        ) from error

    print_result(result, console=console)
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(path, result.to_dict())
        console.print(f"[green]Saved {path.resolve()}[/green]")


@cli.group(name="cuda")
def cuda_group():
    """Inspect CUDA devices and peer topology."""


@cuda_group.command(name="info")
@click.option(
    "--device",
    "device_id",
    type=int,
    help="Show a detailed panel for one device after the summary.",
)
@click.option(
    "--json", "as_json", is_flag=True, help="Emit machine-readable JSON."
)
def cuda_info(device_id: int | None, as_json: bool) -> None:
    """Show a concise CUDA inventory and optional device details."""

    import json

    from fhelium._cli.cuda_view import render_device_info
    from fhelium.native.cuda import get_cuda_device_properties

    try:
        devices = get_cuda_device_properties()
    except Exception as error:
        raise click.ClickException(str(error)) from error
    if not devices:
        raise click.ClickException("No CUDA devices found")
    if device_id is not None and device_id not in devices:
        raise click.ClickException(
            f"Unknown CUDA device {device_id}; available: {list(devices)}"
        )
    if as_json:
        click.echo(json.dumps({"devices": devices}, indent=2))
        return
    render_device_info(console, devices, detail_device=device_id)


@cuda_group.command(name="topo")
@click.option(
    "--bandwidth",
    is_flag=True,
    help="Run the native host-staged and P2P transfer benchmark.",
)
@click.option(
    "--json", "as_json", is_flag=True, help="Emit machine-readable JSON."
)
def cuda_topo(bandwidth: bool, as_json: bool) -> None:
    """Show CUDA peer access and optional measured bandwidth."""

    import json

    from fhelium._cli.cuda_view import render_topology
    from fhelium.native.cuda import get_cuda_info

    try:
        info = get_cuda_info(test_p2p_bandwidth=bandwidth)
    except Exception as error:
        raise click.ClickException(str(error)) from error
    if not info or not info.get("p2p"):
        raise click.ClickException("No CUDA peer topology is available")
    if as_json:
        click.echo(json.dumps(info, indent=2))
        return
    render_topology(
        console,
        info.get("devices", {}),
        info["p2p"],
        bandwidth_requested=bandwidth,
    )


if __name__ == "__main__":
    cli()
