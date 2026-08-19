"""Benchmark discovery and custom benchmark registration."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from importlib import import_module

from fhelium.benchmarks.model import BenchmarkDefinition


class BenchmarkRegistry:
    """Ordered process-local collection of named benchmark definitions."""

    def __init__(self) -> None:
        self._definitions: dict[str, BenchmarkDefinition] = {}

    def register(
        self,
        definition: BenchmarkDefinition,
        *,
        replace: bool = False,
    ) -> BenchmarkDefinition:
        """Register and return ``definition`` under its unique public name.

        Args:
            definition: Complete benchmark definition to retain by reference.
            replace: Replace an existing definition with the same name.

        Raises:
            KeyError: If the name already exists and ``replace`` is false.
        """

        if definition.name in self._definitions and not replace:
            raise KeyError(
                f"Benchmark {definition.name!r} is already registered"
            )
        self._definitions[definition.name] = definition
        return definition

    def get(self, name: str) -> BenchmarkDefinition:
        """Return the definition named ``name`` or raise ``KeyError``."""

        try:
            return self._definitions[name]
        except KeyError as error:
            choices = ", ".join(self._definitions)
            raise KeyError(
                f"Unknown benchmark {name!r}; choices: {choices}"
            ) from error

    def values(self) -> tuple[BenchmarkDefinition, ...]:
        """Return definitions in registration order."""

        return tuple(self._definitions.values())

    def __iter__(self) -> Iterator[BenchmarkDefinition]:
        return iter(self._definitions.values())

    def __len__(self) -> int:
        return len(self._definitions)


registry = BenchmarkRegistry()
_builtins_loaded = False


def register_benchmark(
    definition: BenchmarkDefinition | None = None,
    *,
    replace: bool = False,
) -> BenchmarkDefinition | Callable[[BenchmarkDefinition], BenchmarkDefinition]:
    """Register a definition on the public process-global registry.

    The function accepts a definition directly or can be used as a decorator.
    Custom benchmark-file hooks should prefer their supplied registry instead
    of relying on this process-global object.
    """

    def register(value: BenchmarkDefinition) -> BenchmarkDefinition:
        return registry.register(value, replace=replace)

    return register(definition) if definition is not None else register


def load_builtin_benchmarks() -> BenchmarkRegistry:
    """Load built-in definitions once and return the global registry."""

    global _builtins_loaded
    if not _builtins_loaded:
        import_module("fhelium.benchmarks.v1.operations")
        import_module("fhelium.benchmarks.v1.ntt")
        import_module("fhelium.benchmarks.v1.matrix")
        import_module("fhelium.benchmarks.v1.polynomial")
        import_module(
            "fhelium.benchmarks.standalone."
            "ckks_operator_latency_and_rotation_hoisting"
        )
        import_module(
            "fhelium.benchmarks.standalone.ntt_backend_single_operation_latency"
        )
        import_module("fhelium.benchmarks.standalone.packed_matrix_vector")
        import_module(
            "fhelium.benchmarks.standalone.spmd_collectives_and_rotation_matvec"
        )

        _builtins_loaded = True
    return registry
