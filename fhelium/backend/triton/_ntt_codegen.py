"""Generate compact NTT endpoint stages with fused component expressions."""

from __future__ import annotations

import hashlib
import linecache
from typing import Any


def compile_endpoint(source: str) -> Any:
    """Make generated Triton source inspectable without a source-tree file."""
    import triton
    import triton.language as tl

    from ._arithmetic import (
        add_lazy,
        canonicalize,
        montgomery_mul,
        montgomery_reduce,
        subtract_lazy,
    )

    filename = f"<fhelium-ntt-{hashlib.sha256(source.encode()).hexdigest()}>"
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(True),
        filename,
    )
    namespace = {
        "tl": tl,
        "montgomery_mul": montgomery_mul,
        "montgomery_reduce": montgomery_reduce,
        "__name__": __name__,
        "add_lazy": add_lazy,
        "subtract_lazy": subtract_lazy,
        "canonicalize": canonicalize,
    }
    exec(compile(source, filename, "exec", dont_inherit=True), namespace)
    function = namespace["endpoint"]
    return triton.jit(
        do_not_specialize_on_alignment=[
            name
            for name in function.__code__.co_varnames[
                : function.__code__.co_argcount
            ]
            if function.__annotations__.get(name) is not tl.constexpr
        ],
    )(function)


class StageEmitter:
    """Emit exact compact-table butterflies for one stage interval.

    Tiled intervals exchange lanes through reshapes and permutations. Global
    intervals keep a register tuple per lane, matching grouped native stages.
    All butterflies retain the native lazy interval and operation order.
    """

    def __init__(
        self, *, n: int, start: int, end: int, inverse: bool, tiled: bool
    ) -> None:
        self.n = n
        self.log_n = n.bit_length() - 1
        self.start = start
        self.end = end
        self.inverse = inverse
        self.tiled = tiled
        self.lines: list[str] = []
        self.serial = 0

    def name(self, prefix: str) -> str:
        self.serial += 1
        return f"{prefix}_{self.serial}"

    def indices(self) -> tuple[str, ...]:
        """Return coefficient-index vectors loaded by this endpoint."""
        if self.tiled:
            width = 1 << (self.end - self.start)
            self.lines.extend(
                (
                    f"tile_base = tl.program_id(0).to(tl.int64) * {width}",
                    f"index = tile_base + tl.arange(0, {width})",
                    "mask = index < N",
                )
            )
            return ("index",)
        group = 1 << (self.end - self.start)
        spacing = 1 << (self.start if self.inverse else self.log_n - self.end)
        self.lines.extend(
            (
                "lane = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)",
                f"mask = lane < {self.n // group}",
                f"initial_group = lane // {spacing}",
                f"base = initial_group * {group * spacing} + lane % {spacing}",
            )
        )
        names = []
        for i in range(group):
            name = f"index_{i}"
            self.lines.append(f"{name} = base + {i * spacing}")
            names.append(name)
        return tuple(names)

    def _butterfly(self, u: str, v: str, twiddle: str) -> tuple[str, str]:
        lo, hi = self.name("lo"), self.name("hi")
        total, difference = self.name("sum"), self.name("difference")
        if self.inverse:
            self.lines.extend(
                (
                    f"{total} = {u} + {v}",
                    f"{difference} = {u} + 2 * q - {v}",
                    f"{difference} = tl.where({difference} < 2 * q, {difference}, {difference} - 2 * q)",
                    f"{lo} = tl.where({total} < 2 * q, {total}, {total} - 2 * q)",
                    f"{hi} = montgomery_mul({twiddle}, {difference}, q, k, RADIX)",
                )
            )
        else:
            product = self.name("product")
            self.lines.extend(
                (
                    f"{product} = montgomery_mul({twiddle}, {v}, q, k, RADIX)",
                    f"{total} = {u} + {product}",
                    f"{difference} = {u} + 2 * q - {product}",
                    f"{lo} = tl.where({total} < 2 * q, {total}, {total} - 2 * q)",
                    f"{hi} = tl.where({difference} < 2 * q, {difference}, {difference} - 2 * q)",
                )
            )
        return lo, hi

    def transform(self, values: tuple[str, ...]) -> tuple[str, ...]:
        if self.tiled:
            return (self._tile(values[0]),)
        registers = list(values)
        group = len(registers)
        for stage in range(self.start, self.end):
            r = stage - self.start
            stride = 1 << (r if self.inverse else self.end - stage - 1)
            for base in range(0, group, 2 * stride):
                subgroup = base // (2 * stride)
                twiddle = self.name("twiddle")
                if self.inverse:
                    group_index = f"initial_group * {1 << (self.end - stage - 1)} + {subgroup}"
                    table_base = self.n >> (stage + 1)
                else:
                    group_index = f"initial_group * {1 << r} + {subgroup}"
                    table_base = 1 << stage
                self.lines.append(
                    f"{twiddle} = tl.load(twiddles + row * TWIDDLE_STRIDE + {table_base} + {group_index}, mask=mask, other=0).to(tl.int64)"
                )
                for offset in range(stride):
                    low, high = base + offset, base + offset + stride
                    registers[low], registers[high] = self._butterfly(
                        registers[low], registers[high], twiddle
                    )
        return tuple(registers)

    def _tile(self, value: str) -> str:
        width = 1 << (self.end - self.start)
        for stage in range(self.start, self.end):
            stride = 1 << (stage if self.inverse else self.log_n - stage - 1)
            groups = width // (2 * stride)
            even, odd = self.name("even"), self.name("odd")
            lo_index, twiddle = self.name("lo_index"), self.name("twiddle")
            self.lines.extend(
                (
                    f"{even}, {odd} = tl.split(tl.reshape({value}, ({groups}, 2, {stride})).permute(0, 2, 1))",
                    f"{lo_index} = tile_base + tl.arange(0, {groups})[:, None] * {2 * stride} + tl.arange(0, {stride})[None, :]",
                )
            )
            table_base = self.n >> (stage + 1) if self.inverse else 1 << stage
            self.lines.append(
                f"{twiddle} = tl.load(twiddles + row * TWIDDLE_STRIDE + {table_base} + {lo_index} // {2 * stride}).to(tl.int64)"
            )
            low, high = self._butterfly(even, odd, twiddle)
            value = self.name("tile")
            self.lines.append(
                f"{value} = tl.reshape(tl.join({low}, {high}).permute(0, 2, 1), ({width},))"
            )
        return value
