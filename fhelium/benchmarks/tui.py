"""Textual benchmark and profile selector."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from rich.console import Console
from rich.markup import escape
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.events import Key
from textual.widgets import Label, ListItem, ListView, Static

from fhelium.benchmarks.model import (
    BenchmarkDefinition,
    BenchmarkProfile,
)
from fhelium.benchmarks.registry import BenchmarkRegistry
from fhelium.benchmarks.render import print_result


class BenchmarkApp(App[tuple[str, str] | None]):
    """Select a benchmark/profile; run it only after the TUI has exited."""

    TITLE = "FHElium Benchmarks"
    BINDINGS = [
        ("r", "run_benchmark", "Run"),
        ("q", "quit", "Quit"),
    ]
    CSS = """
    Screen {
        background: #07111f;
        color: #d7e2f0;
    }

    #app-shell {
        width: 100%;
        height: 100%;
        padding: 1 2;
    }

    #brand-bar {
        layout: horizontal;
        width: 100%;
        height: 3;
        padding: 0 2;
        margin-bottom: 1;
        background: #0b1a2d;
        border: round #223a59;
    }

    #brand-title {
        width: 1fr;
        height: 1;
        color: #eaf4ff;
        text-style: bold;
        content-align-vertical: middle;
    }

    #brand-subtitle {
        width: auto;
        height: 1;
        color: #7f9bbd;
        text-align: right;
        content-align-vertical: middle;
    }

    #selector-area {
        layout: horizontal;
        width: 100%;
        height: 1fr;
    }

    .selector-pane {
        layout: vertical;
        height: 100%;
        padding: 1;
        background: #0a1728;
        border: round #22344d;
    }

    #benchmark-pane {
        width: 3fr;
        margin-right: 1;
    }

    #profile-pane {
        width: 2fr;
    }

    .selector-pane.active {
        background: #0d1d31;
        border: heavy #36c5f0;
    }

    .pane-eyebrow {
        height: 1;
        color: #36c5f0;
        text-style: bold;
    }

    .pane-title {
        height: 2;
        color: #eef7ff;
        text-style: bold;
        content-align-vertical: middle;
    }

    ListView {
        height: 1fr;
        margin-top: 1;
        padding: 0 1;
        background: #071321;
        border: none;
    }

    ListView:focus {
        background: #0a192a;
    }

    ListItem {
        height: 3;
        padding: 1;
        color: #9fb2ca;
    }

    ListItem.--highlight {
        background: #12304a;
        color: #ffffff;
        text-style: bold;
    }

    .description-card {
        height: 10;
        margin-top: 1;
        padding: 1 2;
        background: #081422;
        border-top: solid #223a59;
        color: #a9bdd5;
        overflow-y: auto;
    }

    #profile-description {
        height: 14;
    }

    #selection-bar {
        width: 100%;
        height: 3;
        margin-top: 1;
        padding: 0 2;
        background: #0b1a2d;
        border: round #223a59;
        color: #8ea9c7;
        content-align-vertical: middle;
    }

    #shortcut-bar {
        width: 100%;
        height: 3;
        margin-top: 1;
        padding: 1 2;
        background: #071321;
        color: #708baa;
        content-align-horizontal: center;
    }
    """

    def __init__(self, registry: BenchmarkRegistry):
        super().__init__()
        self.registry = registry
        self.definitions = registry.values()
        self.current_definition: BenchmarkDefinition | None = None
        self.current_profile: BenchmarkProfile | None = None
        self.focus_on_benchmark = True

    def compose(self) -> ComposeResult:
        with Vertical(id="app-shell"):
            with Horizontal(id="brand-bar"):
                yield Static(
                    "FHELIUM  /  PERFORMANCE LAB",
                    id="brand-title",
                )
                yield Static(
                    "Choose a workload and a reproducible execution profile.",
                    id="brand-subtitle",
                )
            with Horizontal(id="selector-area"):
                with Vertical(
                    id="benchmark-pane",
                    classes="selector-pane",
                ):
                    yield Static("01  BENCHMARK", classes="pane-eyebrow")
                    yield Static(
                        "What do you want to measure?", classes="pane-title"
                    )
                    yield ListView(id="benchmark-list")
                    yield Static(
                        "",
                        id="benchmark-description",
                        classes="description-card",
                    )
                with Vertical(
                    id="profile-pane",
                    classes="selector-pane",
                ):
                    yield Static("02  PROFILE", classes="pane-eyebrow")
                    yield Static("How should it run?", classes="pane-title")
                    yield ListView(id="profile-list")
                    yield Static(
                        "",
                        id="profile-description",
                        classes="description-card",
                    )
            yield Static("", id="selection-bar")
            yield Static(
                "↑ ↓  SELECT     ← →  PANEL     R  RUN     Q  QUIT",
                id="shortcut-bar",
            )

    async def on_mount(self) -> None:
        benchmark_list = self.query_one("#benchmark-list", ListView)
        await benchmark_list.extend(
            ListItem(
                Label(
                    f"[bold #36c5f0]{escape(item.category.upper())}[/]  "
                    f"{escape(item.title)}"
                )
            )
            for item in self.definitions
        )
        if self.definitions:
            benchmark_list.index = 0
            await self._select_benchmark(0)
        self._focus_benchmark_panel()

    def _update_panel_style(self) -> None:
        benchmark_pane = self.query_one("#benchmark-pane", Vertical)
        profile_pane = self.query_one("#profile-pane", Vertical)
        benchmark_pane.set_class(self.focus_on_benchmark, "active")
        profile_pane.set_class(not self.focus_on_benchmark, "active")

    def _focus_benchmark_panel(self) -> None:
        self.focus_on_benchmark = True
        self.set_focus(self.query_one("#benchmark-list", ListView))
        self._update_panel_style()

    def _focus_profile_panel(self) -> None:
        if self.current_definition is None:
            return
        self.focus_on_benchmark = False
        profile_list = self.query_one("#profile-list", ListView)
        if profile_list.index is None:
            profile_list.index = 0
        self.set_focus(profile_list)
        self._update_panel_style()
        self._select_profile(profile_list.index or 0)

    async def _select_benchmark(self, index: int | None) -> None:
        if index is None or not 0 <= index < len(self.definitions):
            return
        definition = self.definitions[index]
        self.current_definition = definition
        self.current_profile = None
        self.query_one("#benchmark-description", Static).update(
            f"[bold #eaf4ff]{escape(definition.title)}[/]\n\n"
            f"{escape(definition.description.strip())}"
        )

        profile_list = self.query_one("#profile-list", ListView)
        await profile_list.clear()
        await profile_list.extend(
            ListItem(Label(f"[bold]{escape(profile.name)}[/]"))
            for profile in definition.profiles
        )
        if definition.profiles:
            profile_list.index = 0
            self._select_profile(0)
        self._update_selection_bar()

    def _select_profile(self, index: int | None) -> None:
        definition = self.current_definition
        if (
            definition is None
            or index is None
            or not 0 <= index < len(definition.profiles)
        ):
            return
        profile = definition.profiles[index]
        self.current_profile = profile
        parameters = "\n".join(
            f"[dim #66819f]{escape(str(key))}[/]  "
            f"{escape(_compact_value(value))}"
            for key, value in profile.parameters.items()
        )
        self.query_one("#profile-description", Static).update(
            f"[bold #eaf4ff]{escape(profile.description)}[/]\n\n{parameters}"
        )
        self._update_selection_bar()

    def _update_selection_bar(self) -> None:
        definition = self.current_definition
        profile = self.current_profile
        if definition is None or profile is None:
            text = "Select a benchmark and profile"
        else:
            text = (
                f"[dim]READY[/]  [bold #36c5f0]{escape(definition.title)}[/]"
                f"  [#607d9d]/[/]  [bold]{escape(profile.name)}[/]"
            )
        self.query_one("#selection-bar", Static).update(text)

    async def on_list_view_highlighted(
        self, event: ListView.Highlighted
    ) -> None:
        if event.list_view.id == "benchmark-list":
            await self._select_benchmark(event.list_view.index)
        elif event.list_view.id == "profile-list":
            self._select_profile(event.list_view.index)

    async def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.list_view.id == "benchmark-list":
            await self._select_benchmark(event.list_view.index)
            self._focus_profile_panel()
        elif event.list_view.id == "profile-list":
            self._select_profile(event.list_view.index)
            self.action_run_benchmark()

    def on_key(self, event: Key) -> None:
        if event.key == "left" and not self.focus_on_benchmark:
            event.stop()
            event.prevent_default()
            self._focus_benchmark_panel()
        elif event.key == "right" and self.focus_on_benchmark:
            event.stop()
            event.prevent_default()
            self._focus_profile_panel()

    def action_run_benchmark(self) -> None:
        if self.current_definition is None or self.current_profile is None:
            self.notify("Select a benchmark and profile", severity="warning")
            return
        self.exit((self.current_definition.name, self.current_profile.name))


def _compact_value(value) -> str:
    if isinstance(value, Mapping):
        return f"{len(value)} entries"
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = [str(item) for item in value]
        if len(values) > 4:
            return f"{', '.join(values[:3])}, …  ({len(values)} items)"
        return ", ".join(values)
    return str(value)


def run_tui(registry: BenchmarkRegistry) -> None:
    """Select in Textual, then run and render in the restored terminal."""

    selection = BenchmarkApp(registry).run()
    if selection is None:
        return
    benchmark_name, profile_name = selection
    definition = registry.get(benchmark_name)
    profile = definition.profile(profile_name)
    console = Console()
    console.rule(f"[bold cyan]{definition.title}[/bold cyan] · {profile.name}")

    def progress(message: str) -> None:
        console.print(f"[dim]{message}[/dim]")

    try:
        result = definition.runner(profile, progress)
    except Exception as error:
        console.print(
            f"[bold red]Benchmark failed:[/bold red] "
            f"{type(error).__name__}: {error}"
        )
        return
    print_result(result, console=console)
