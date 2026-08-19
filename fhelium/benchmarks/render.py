"""Rich rendering shared by benchmark CLI and TUI."""

from __future__ import annotations

from typing import Any

from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from fhelium.benchmarks.model import BenchmarkResult


def _format(value: Any) -> str:
    if isinstance(value, float):
        if value != 0.0 and (abs(value) < 1e-3 or abs(value) >= 1e5):
            return f"{value:.4e}"
        return f"{value:.4f}"
    if isinstance(value, (list, tuple, dict)):
        return str(value)
    return str(value)


def _rows_table(rows: list[dict[str, Any]]) -> Table | None:
    if not rows:
        return None
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    table = Table(title="Measurements", expand=True)
    for column in columns:
        table.add_column(column, no_wrap=column == "backend")
    for row in rows:
        table.add_row(*(_format(row.get(column, "")) for column in columns))
    return table


def _mapping_table(title: str, values: dict[str, Any]) -> Table | None:
    if not values:
        return None
    table = Table(title=title, expand=True)
    table.add_column("name", style="cyan")
    table.add_column("value")
    for name, value in values.items():
        table.add_row(name, _format(value))
    return table


def result_renderables(result: BenchmarkResult) -> tuple[RenderableType, ...]:
    renderables: list[RenderableType] = [
        Panel(
            f"benchmark={result.benchmark}\nprofile={result.profile}",
            title="Benchmark result",
            border_style="cyan",
        )
    ]
    for table in (
        _rows_table(result.rows),
        _mapping_table("Scalars", result.scalars),
        _mapping_table("Metadata", result.metadata),
    ):
        if table is not None:
            renderables.append(table)
    if result.notes:
        notes = Text("\n".join(f"• {note}" for note in result.notes))
        renderables.append(Panel(notes, title="Notes", border_style="dim"))
    return tuple(renderables)


def result_group(result: BenchmarkResult) -> Group:
    return Group(*result_renderables(result))


def print_result(
    result: BenchmarkResult, *, console: Console | None = None
) -> None:
    output = console or Console()
    for renderable in result_renderables(result):
        output.print(renderable)
