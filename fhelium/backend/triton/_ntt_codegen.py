"""Generate compact NTT endpoint stages with fused component expressions."""

from __future__ import annotations

import hashlib
import linecache
from typing import Any


def compile_endpoint(source: str) -> Any:
    """Make generated Triton source inspectable without a source-tree file."""
    import triton
    import triton.language as tl
    from triton import knobs

    from ._arithmetic import (
        add_lazy,
        canonicalize,
        montgomery_mul,
        montgomery_reduce,
        subtract_lazy,
        shoup32_mul,
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
        "shoup32_mul": shoup32_mul,
    }
    exec(compile(source, filename, "exec", dont_inherit=True), namespace)
    function = namespace["endpoint"]
    alignment_independent = [
        name
        for name in function.__code__.co_varnames[
            : function.__code__.co_argcount
        ]
        if function.__annotations__.get(name) is not tl.constexpr
    ]
    if knobs.runtime.interpret:
        return triton.jit(do_not_specialize_on_alignment=alignment_independent)(
            function
        )

    from ._ntt_compiler import NttJITFunction

    return NttJITFunction(
        function, do_not_specialize_on_alignment=alignment_independent
    )


class StageEmitter:
    """Emit compact-table butterflies for one NTT stage interval.

    Tiled intervals exchange lanes through reshapes and permutations. Global
    intervals keep a register tuple per lane, matching grouped native stages.
    All butterflies retain the native lazy interval and operation order.
    """

    def __init__(
        self,
        *,
        n: int,
        start: int,
        end: int,
        inverse: bool,
        tiled: bool,
        radix: int | None = None,
        twiddle_encoding: str = "montgomery",
        prefix: str = "",
    ) -> None:
        self.n = n
        self.log_n = n.bit_length() - 1
        self.start = start
        self.end = end
        self.inverse = inverse
        self.tiled = tiled
        self.radix = 2 if radix is None else radix
        self.prefix = prefix
        self.twiddle_encoding = twiddle_encoding
        self.pair_aligned: str | None = None
        self.roots: dict[int, str] = {}
        self.lines: list[str] = []
        self.serial = 0

    def name(self, prefix: str) -> str:
        self.serial += 1
        return f"{self.prefix}{prefix}_{self.serial}"

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

    def _twiddle(self, index: str, *, masked: bool) -> tuple[str, ...]:
        root = self.name("twiddle")
        address = f"twiddles + row * TWIDDLE_STRIDE + ({index}) * TWIDDLE_ELEMENT_STRIDE"
        mask = ", mask=mask, other=0" if masked else ""
        if self.twiddle_encoding == "montgomery":
            self.lines.append(f"{root} = tl.load({address}{mask}).to(tl.int64)")
            return (root,)
        quotient = self.name("quotient")
        if self.pair_aligned is None:
            row_address = self.name("row_address")
            self.pair_aligned = self.name("pair_aligned")
            # Cached launchers accept bindings with the same strides and
            # different offsets. Check the actual row address at execution.
            self.lines.extend(
                (
                    f"{row_address} = (twiddles + row * TWIDDLE_STRIDE).to(tl.uint64)",
                    f"{self.pair_aligned} = (TWIDDLE_WORD_STRIDE == 1) & (TWIDDLE_ELEMENT_STRIDE % 2 == 0) & (({row_address} & 7) == 0)",
                )
            )
        packed = self.name("root_pair")
        self.lines.extend(
            (
                f"if {self.pair_aligned}:",
                # The pair starts at an 8-byte-aligned row address plus an
                # even number of int32 words, and the second word is adjacent.
                f"    {packed} = tl.load(({address}).to(tl.pointer_type(tl.uint64)){mask})",
                f"    {root} = ({packed} & 0xFFFFFFFF).to(tl.int64)",
                f"    {quotient} = ({packed} >> 32).to(tl.uint32)",
                "else:",
                f"    {root} = tl.load({address}{mask}).to(tl.int64)",
                f"    {quotient} = tl.load({address} + TWIDDLE_WORD_STRIDE{mask}).to(tl.uint32)",
            )
        )
        return root, quotient

    def _product(self, value: str, twiddle: tuple[str, ...]) -> str:
        if self.twiddle_encoding == "montgomery":
            return f"montgomery_mul({twiddle[0]}, {value}, q, k, RADIX)"
        return f"shoup32_mul({value}, {twiddle[0]}, {twiddle[1]}, q)"

    def _butterfly(
        self, u: str, v: str, twiddle: tuple[str, ...]
    ) -> tuple[str, str]:
        lo, hi = self.name("lo"), self.name("hi")
        total, difference = self.name("sum"), self.name("difference")
        if self.inverse:
            self.lines.extend(
                (
                    f"{total} = {u} + {v}",
                    f"{difference} = {u} + 2 * q - {v}",
                    f"{difference} = tl.where({difference} < 2 * q, {difference}, {difference} - 2 * q)",
                    f"{lo} = tl.where({total} < 2 * q, {total}, {total} - 2 * q)",
                    f"{hi} = {self._product(difference, twiddle)}",
                )
            )
        else:
            product = self.name("product")
            self.lines.extend(
                (
                    f"{product} = {self._product(v, twiddle)}",
                    f"{total} = {u} + {product}",
                    f"{difference} = {u} + 2 * q - {product}",
                    f"{lo} = tl.where({total} < 2 * q, {total}, {total} - 2 * q)",
                    f"{hi} = tl.where({difference} < 2 * q, {difference}, {difference} - 2 * q)",
                )
            )
        return lo, hi

    def transform(self, values: tuple[str, ...]) -> tuple[str, ...]:
        if self.radix != 2:
            return self._fixed(values)
        if self.tiled:
            return (self._tile(values[0]),)
        registers = list(values)
        group = len(registers)
        for stage in range(self.start, self.end):
            r = stage - self.start
            stride = 1 << (r if self.inverse else self.end - stage - 1)
            for base in range(0, group, 2 * stride):
                subgroup = base // (2 * stride)
                if self.inverse:
                    group_index = f"initial_group * {1 << (self.end - stage - 1)} + {subgroup}"
                    table_base = self.n >> (stage + 1)
                else:
                    group_index = f"initial_group * {1 << r} + {subgroup}"
                    table_base = 1 << stage
                twiddle = self._twiddle(
                    f"{table_base} + {group_index}", masked=True
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
            lo_index = self.name("lo_index")
            self.lines.extend(
                (
                    f"{even}, {odd} = tl.split(tl.reshape({value}, ({groups}, 2, {stride})).permute(0, 2, 1))",
                    f"{lo_index} = tile_base + tl.arange(0, {groups})[:, None] * {2 * stride} + tl.arange(0, {stride})[None, :]",
                )
            )
            table_base = self.n >> (stage + 1) if self.inverse else 1 << stage
            twiddle = self._twiddle(
                f"{table_base} + {lo_index} // {2 * stride}", masked=False
            )
            low, high = self._butterfly(even, odd, twiddle)
            value = self.name("tile")
            self.lines.append(
                f"{value} = tl.reshape(tl.join({low}, {high}).permute(0, 2, 1), ({width},))"
            )
        return value

    def _cyclic4(self, values: list[str]) -> list[str]:
        names = [self.name("radix4") for _ in range(9)]
        es, ed, os, od, rotated, y0, y1, y2, y3 = names
        self.lines.extend(
            (
                f"{es} = add_lazy({values[0]}, {values[2]}, q)",
                f"{ed} = subtract_lazy({values[0]}, {values[2]}, q)",
                f"{os} = add_lazy({values[1]}, {values[3]}, q)",
                f"{od} = subtract_lazy({values[1]}, {values[3]}, q)",
                f"{rotated} = montgomery_mul({self._root(self.radix // 4)}, {od}, q, k, RADIX)",
                f"{y0} = add_lazy({es}, {os}, q)",
                f"{y2} = subtract_lazy({es}, {os}, q)",
                f"{y1} = add_lazy({ed}, {rotated}, q)",
                f"{y3} = subtract_lazy({ed}, {rotated}, q)",
            )
        )
        return [y0, y1, y2, y3]

    def _root(self, exponent: int) -> str:
        root = self.roots.get(exponent)
        if root is None:
            root = self.name("root")
            self.roots[exponent] = root
            self.lines.append(
                f"{root} = tl.load(roots + row * ROOT_STRIDE + {exponent} * ROOT_ELEMENT_STRIDE).to(tl.int64)"
            )
        return root

    def _couple(self, value: str, exponent: int) -> str:
        if exponent == 0:
            return value
        root = self._root(exponent)
        name = self.name("coupled")
        self.lines.append(
            f"{name} = montgomery_mul({root}, {value}, q, k, RADIX)"
        )
        return name

    def _cyclic(self, values: list[str]) -> list[str]:
        if self.radix == 4:
            return self._cyclic4(values)
        if self.radix == 8:
            even = self._cyclic4(values[::2])
            odd = self._cyclic4(values[1::2])
            result = [""] * 8
            for frequency in range(4):
                coupled = self._couple(odd[frequency], frequency)
                low, high = self.name("low"), self.name("high")
                self.lines.extend(
                    (
                        f"{low} = add_lazy({even[frequency]}, {coupled}, q)",
                        f"{high} = subtract_lazy({even[frequency]}, {coupled}, q)",
                    )
                )
                result[frequency], result[frequency + 4] = low, high
            return result
        columns = [self._cyclic4(values[column::4]) for column in range(4)]
        result = [""] * 16
        for frequency in range(4):
            row = [
                self._couple(columns[column][frequency], frequency * column)
                for column in range(4)
            ]
            for high, value in enumerate(self._cyclic4(row)):
                result[frequency + 4 * high] = value
        return result

    def _digit(self, values: list[str], group: str, stage: int) -> list[str]:
        bits = self.radix.bit_length() - 1
        permutation = [
            int(f"{lane:0{bits}b}"[::-1], 2) for lane in range(self.radix)
        ]
        offset = (
            (self.n - (self.n >> stage)) if self.inverse else (1 << stage) - 1
        )
        if self.inverse:
            values = [values[index] for index in permutation]
        else:
            values = self._twist(values, group, offset)
        values = self._cyclic(values)
        return (
            self._twist(values, group, offset)
            if self.inverse
            else [values[index] for index in permutation]
        )

    def _twist(self, values: list[str], group: str, offset: int) -> list[str]:
        result = [values[0]]
        for lane in range(1, self.radix):
            twist, value = self.name("twist"), self.name("twisted")
            self.lines.extend(
                (
                    f"{twist} = tl.load(twiddles + row * TWIDDLE_STRIDE + ({offset} + ({group}) * {self.radix - 1} + {lane - 1}) * TWIDDLE_ELEMENT_STRIDE{', mask=mask, other=0' if not self.tiled else ''}).to(tl.int64)",
                    f"{value} = montgomery_mul({values[lane]}, {twist}, q, k, RADIX)",
                )
            )
            result.append(value)
        return result

    def _unpack(self, value: str, bits: int) -> list[str]:
        lanes = [""] * (1 << bits)

        def split(value, remaining, index, place):
            if remaining == 0:
                lanes[index] = value
                return
            low, high = self.name("lane"), self.name("lane")
            self.lines.append(f"{low}, {high} = tl.split({value})")
            split(low, remaining - 1, index, place * 2)
            split(high, remaining - 1, index + place, place * 2)

        split(value, bits, 0, 1)
        return lanes

    def _pack(self, values: list[str], bits: int) -> str:
        def join(remaining, index, place):
            if remaining == 0:
                return values[index]
            low = join(remaining - 1, index, place * 2)
            high = join(remaining - 1, index + place, place * 2)
            result = self.name("joined")
            self.lines.append(f"{result} = tl.join({low}, {high})")
            return result

        return join(bits, 0, 1)

    def _fixed(self, values: tuple[str, ...]) -> tuple[str, ...]:
        bits = self.radix.bit_length() - 1
        if self.tiled:
            value = values[0]
            width = 1 << (self.end - self.start)
            for stage in range(self.start, self.end, bits):
                stride = 1 << (
                    stage if self.inverse else self.log_n - stage - bits
                )
                groups = width // (self.radix * stride)
                unpacked = self.name("unpacked")
                shape = (groups, *((2,) * bits), stride)
                permutation = (0, bits + 1, *range(1, bits + 1))
                self.lines.append(
                    f"{unpacked} = tl.reshape({value}, {shape!r}).permute{permutation!r}"
                )
                lanes = self._unpack(unpacked, bits)
                group = f"tile_base // {self.radix * stride} + tl.arange(0, {groups})[:, None]"
                lanes = self._digit(lanes, group, stage)
                joined = self._pack(lanes, bits)
                value = self.name("tile")
                permutation = (0, *range(2, bits + 2), 1)
                self.lines.append(
                    f"{value} = tl.reshape({joined}.permute{permutation!r}, ({width},))"
                )
            return (value,)
        registers = list(values)
        for stage in range(self.start, self.end, bits):
            stride = 1 << (
                stage - self.start if self.inverse else self.end - stage - bits
            )
            for base in range(0, len(registers), self.radix * stride):
                subgroup = base // (self.radix * stride)
                group = f"initial_group * {1 << (self.end - stage - bits if self.inverse else stage - self.start)} + {subgroup}"
                for offset in range(stride):
                    indices = [
                        base + offset + lane * stride
                        for lane in range(self.radix)
                    ]
                    result = self._digit(
                        [registers[index] for index in indices], group, stage
                    )
                    for index, value in zip(indices, result, strict=True):
                        registers[index] = value
        return tuple(registers)
