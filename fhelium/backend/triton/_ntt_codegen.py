"""Generate compact NTT endpoint stages with fused component expressions."""

from __future__ import annotations

import hashlib
import linecache
from typing import Any, Literal


def compile_endpoint(source: str) -> Any:
    """Make generated Triton source inspectable without a source-tree file."""
    import triton
    import triton.language as tl
    from triton import knobs

    from ._arithmetic import (
        add_lazy,
        butterfly_sum,
        narrow,
        canonicalize,
        montgomery_mul,
        montgomery_reduce,
        subtract_lazy,
        shoup32_mul,
        seeded_key_residue,
        seeded_key_residue8,
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
        "butterfly_sum": butterfly_sum,
        "narrow": narrow,
        "subtract_lazy": subtract_lazy,
        "canonicalize": canonicalize,
        "shoup32_mul": shoup32_mul,
        "seeded_key_residue": seeded_key_residue,
        "seeded_key_residue8": seeded_key_residue8,
    }
    exec(compile(source, filename, "exec", dont_inherit=True), namespace)
    function = namespace.pop("endpoint")
    alignment_independent, scalars = _parameter_policy(function)
    if knobs.runtime.interpret:
        return triton.jit(
            do_not_specialize=scalars,
            do_not_specialize_on_alignment=alignment_independent,
        )(function)

    from ._ntt_compiler import NttJITFunction

    return NttJITFunction(
        function,
        do_not_specialize=scalars,
        do_not_specialize_on_alignment=alignment_independent,
    )


def _parameter_policy(function: Any) -> tuple[list[str], list[str]]:
    """Return (alignment-independent, value-independent) parameter names.

    Pointer and scalar parameters never specialize on 16-byte alignment, since
    Tensor views bound later may be unaligned. Integer scalar parameters
    (annotated ``tl.int64``) also never specialize on their value, so one
    compiled kernel serves every row count and stride passed at launch.
    """
    import triton.language as tl

    names = function.__code__.co_varnames[: function.__code__.co_argcount]
    annotations = function.__annotations__
    alignment_independent = [
        name for name in names if annotations.get(name) is not tl.constexpr
    ]
    scalars = [name for name in names if annotations.get(name) is tl.int64]
    return alignment_independent, scalars


# A full program runs on 256 threads, and each thread holds 32 bytes of
# coefficients per register round: 2^3 int32 or 2^2 int64 residues.
_THREADS = 256
# Coefficient values per program that loads may hold in registers.
_LOAD_ELEMENTS = 16384
# Smallest program size of a stage interval.
_MIN_ELEMENTS = 512
# Registers per thread for residues loaded ahead of their use: half of the
# 255 registers a thread may address.
_HELD_REGISTERS = 128


def round_bits(itemsize: int) -> int:
    """Return the butterfly stages per register round for ``itemsize``-byte residues."""
    return (32 // itemsize).bit_length() - 1


def radix2_elements(loads: int, bits: int, elements: int | None = None) -> int:
    """Return the program size, at most ``elements``, for ``loads`` coefficient loads."""
    elements = _THREADS << bits if elements is None else elements
    while elements > 128 and elements * loads > _LOAD_ELEMENTS:
        elements //= 2
    return elements


def radix2_program(stages: int, tiled: bool, bits: int) -> int:
    """Return the smallest program size for one ``stages``-stage interval.

    A thread round holds ``2^bits`` residues, or 32 bytes, which is one memory
    sector. A column interval reads ``lanes`` adjacent coefficients in each of
    its ``2^stages`` rows, so its programs keep ``lanes >= 2^bits`` to load and
    store whole sectors. Tile intervals are contiguous at every size.
    """
    ceiling = _THREADS << bits
    if tiled:
        return min(_MIN_ELEMENTS, ceiling)
    return min(ceiling, max(_MIN_ELEMENTS, 1 << (bits + stages)))


def radix2_geometry(
    n: int, start: int, end: int, elements: int
) -> tuple[int, int]:
    """Return (programs, lanes) of one radix-2 stage interval.

    A lane is one contiguous tile or one coefficient column of ``2^(end-start)``
    coefficients; each program transforms ``lanes`` of them.
    """
    stages = end - start
    groups = n >> stages
    lanes = max(1, min(elements >> stages, groups))
    return groups // lanes, lanes


def radix2_warps(
    n: int, start: int, end: int, elements: int, bits: int, loads: int
) -> int:
    """Return the warps of one program.

    Every thread holds at least one register slot of the program. A program
    may also issue all ``loads`` coefficient loads ahead of the arithmetic
    that consumes them, so enough threads share its ``elements x loads``
    residues, of ``2^(3 - bits)`` registers each, to keep each thread's
    share within ``_HELD_REGISTERS``.
    """
    stages = end - start
    lanes = radix2_geometry(n, start, end, elements)[1]
    slots = lanes << (stages - min(bits, stages))
    held = -(-(elements * loads << (3 - bits)) // _HELD_REGISTERS)
    warps = -(-max(slots, held) // 32)
    return min(8, 1 << (warps - 1).bit_length())


class StageEmitter:
    """Emit compact-table butterflies for one NTT stage interval.

    A radix-2 interval of M stages transforms lanes of 2^M coefficients, each
    a contiguous tile or a coefficient column, with the coefficient position r
    in [0, 2^M). The program holds 2^K register vectors whose element e covers
    r bits [lo, lo + K) of its slot. Rounds of K stages run in registers and
    exchange r bits between rounds through a reshape and permutation. Loaded
    and transformed values use the window lo = M - K, matching the native
    compact passes. Radix-2 twiddle loads collect in ``preloads``, once per
    index for all components, for placement ahead of the coefficient loads.
    Fixed-radix intervals exchange tiled lanes per digit and
    keep a register tuple per strided lane. All butterflies retain the native
    lazy interval and operation order.
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
        elements: int,
        bits: int,
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
        # Radix-2 twiddle loads, shared by the components of the endpoint.
        self.preloads: list[str] = []
        self.twiddles: dict[str, tuple[str, ...]] = {}
        self.serial = 0
        self.stages = end - start
        self.bits = min(bits, self.stages)
        self.programs, self.lanes = radix2_geometry(n, start, end, elements)
        self.slots = self.lanes << (self.stages - self.bits)

    def name(self, prefix: str) -> str:
        self.serial += 1
        return f"{self.prefix}{prefix}_{self.serial}"

    @property
    def segment_index(self) -> bool:
        """Whether index elements 8k..8k+7 hold coefficients 8m..8m+7."""
        if self.radix == 2:
            m, k = self.stages, self.bits
            return (1 << (m - k) if self.tiled else self.lanes) >= 8
        return self.tiled and self.end - self.start >= 3

    def indices(self) -> tuple[str, ...]:
        """Return coefficient-index vectors loaded by this endpoint."""
        if self.radix == 2:
            return self._radix2_indices()
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

    def _twiddle(self, index: str) -> tuple[str, ...]:
        cached = self.twiddles.get(index)
        if cached is not None:
            return cached
        root = self.name("twiddle")
        address = f"twiddles + row * TWIDDLE_STRIDE + ({index}) * TWIDDLE_ELEMENT_STRIDE"
        lines = self.preloads
        if self.twiddle_encoding == "montgomery":
            lines.append(f"{root} = tl.load({address}).to(tl.int64)")
            self.twiddles[index] = (root,)
            return (root,)
        quotient = self.name("quotient")
        if self.pair_aligned is None:
            row_address = self.name("row_address")
            self.pair_aligned = self.name("pair_aligned")
            # Cached launchers accept bindings with the same strides and
            # different offsets. Check the actual row address at execution.
            lines.extend(
                (
                    f"{row_address} = (twiddles + row * TWIDDLE_STRIDE).to(tl.uint64)",
                    f"{self.pair_aligned} = (TWIDDLE_WORD_STRIDE == 1) & (TWIDDLE_ELEMENT_STRIDE % 2 == 0) & (({row_address} & 7) == 0)",
                )
            )
        packed = self.name("root_pair")
        lines.extend(
            (
                f"if {self.pair_aligned}:",
                # The pair starts at an 8-byte-aligned row address plus an
                # even number of int32 words, and the second word is adjacent.
                f"    {packed} = tl.load(({address}).to(tl.pointer_type(tl.uint64)))",
                f"    {root} = ({packed} & 0xFFFFFFFF).to(tl.int64)",
                f"    {quotient} = ({packed} >> 32).to(tl.uint32)",
                "else:",
                f"    {root} = tl.load({address}).to(tl.int64)",
                f"    {quotient} = tl.load({address} + TWIDDLE_WORD_STRIDE).to(tl.uint32)",
            )
        )
        self.twiddles[index] = (root, quotient)
        return root, quotient

    def _product(self, value: str, twiddle: tuple[str, ...]) -> str:
        if self.twiddle_encoding == "montgomery":
            return f"montgomery_mul({twiddle[0]}, {value}, q, k, RADIX)"
        return f"shoup32_mul({value}, {twiddle[0]}, {twiddle[1]}, q)"

    def _butterfly(
        self, u: str, v: str, twiddle: tuple[str, ...]
    ) -> tuple[str, str]:
        lo, hi = self.name("lo"), self.name("hi")
        if self.inverse:
            difference = self.name("difference")
            self.lines.extend(
                (
                    f"{lo}, {difference} = butterfly_sum({u}, {v}, q, RADIX)",
                    f"{hi} = {self._product(difference, twiddle)}",
                )
            )
        else:
            product = self.name("product")
            self.lines.extend(
                (
                    f"{product} = {self._product(v, twiddle)}",
                    f"{lo}, {hi} = butterfly_sum({u}, {product}, q, RADIX)",
                )
            )
        return lo, hi

    def transform(self, values: tuple[str, ...]) -> tuple[str, ...]:
        if self.radix != 2:
            return self._fixed(values)
        m, k = self.stages, self.bits
        packed = self.name("registers")
        self.lines.append(
            f"{packed} = tl.permute(tl.reshape({values[0]}, {(*(2,) * k, self.slots)!r}), {(k, *range(k))!r})"
        )
        registers = self._unpack(packed, k)
        window = m - k
        for round_ in range((m + k - 1) // k):
            if self.inverse:
                low = min(round_ * k, m - k)
            else:
                low = max(m - (round_ + 1) * k, 0)
            registers = self._exchange(registers, window, low)
            window = low
            first = self._first_index(low)
            for local in range(round_ * k, min((round_ + 1) * k, m)):
                stage = self.start + local
                bit = (local if self.inverse else m - 1 - local) - low
                for e in range(1 << k):
                    if e & (1 << bit):
                        continue
                    offset = e >> (bit + 1)
                    twiddle = self._twiddle(
                        f"{self.n >> (stage + 1)} + ({first} >> {stage + 1}) + {offset}"
                        if self.inverse
                        else f"{1 << stage} + ({first} >> {self.log_n - stage}) + {offset}"
                    )
                    high = e | (1 << bit)
                    registers[e], registers[high] = self._butterfly(
                        registers[e], registers[high], twiddle
                    )
        registers = self._exchange(registers, window, m - k)
        result = self.name("transformed")
        self.lines.append(
            f"{result} = tl.reshape(tl.permute({self._pack(registers, k)}, {(*range(1, k + 1), 0)!r}), ({self.slots << k},))"
        )
        return (result,)

    def _radix2_indices(self) -> tuple[str, ...]:
        m, k = self.stages, self.bits
        rest = 1 << (m - k)
        lane = f"slot // {rest}" if self.tiled else f"slot % {self.lanes}"
        offset = f"slot % {rest}" if self.tiled else f"slot // {self.lanes}"
        step = 1 << (m - k) if self.tiled else (self.n >> m) << (m - k)
        # Element e * slots + slot holds register e of the slot, so pointwise
        # expressions are emitted once for the whole program.
        self.lines.extend(
            (
                f"element = tl.arange(0, {self.slots << k})",
                f"slot = element % {self.slots}",
                f"lane_group = tl.program_id(0).to(tl.int64) * {self.lanes} + {lane}",
                f"lane_rest = {offset}",
                f"index = {self._first_index(m - k)} + (element // {self.slots}) * {step}",
                "mask = index < N",
                f"lane_rest = tl.arange(0, {self.slots}) % {rest}"
                if self.tiled
                else f"lane_rest = tl.arange(0, {self.slots}) // {self.lanes}",
                f"lane_group = tl.program_id(0).to(tl.int64) * {self.lanes} + "
                + (
                    f"tl.arange(0, {self.slots}) // {rest}"
                    if self.tiled
                    else f"tl.arange(0, {self.slots}) % {self.lanes}"
                ),
            )
        )
        return ("index",)

    def _first_index(self, low: int) -> str:
        """Coefficient index of register 0 for the window starting at bit low."""
        m, k = self.stages, self.bits
        position = (
            "lane_rest"
            if low + k == m
            else f"((lane_rest & {(1 << low) - 1}) | ((lane_rest >> {low}) << {low + k}))"
        )
        if self.tiled:
            return f"(lane_group * {1 << m} + {position})"
        return f"({position} * {self.n >> m} + lane_group)"

    def _sequence(self, low: int) -> list[int | Literal["lane"]]:
        """Flattened axis order of packed registers, most significant first."""
        m, k = self.stages, self.bits
        rest = [*range(m - 1, low + k - 1, -1), *range(low - 1, -1, -1)]
        window = list(range(low + k - 1, low - 1, -1))
        return (
            ["lane", *rest, *window] if self.tiled else [*rest, "lane", *window]
        )

    def _exchange(
        self, registers: list[str], source: int, target: int
    ) -> list[str]:
        """Move register values from the source window to the target window."""
        if source == target:
            return registers
        m, k = self.stages, self.bits
        bounds = sorted({0, m, source, source + k, target, target + k})
        segment = {
            bit: index
            for index, (lo, hi) in enumerate(zip(bounds, bounds[1:]))
            for bit in range(lo, hi)
        }

        def axes(low: int) -> list[int | Literal["lane"]]:
            order: list[int | Literal["lane"]] = []
            for token in self._sequence(low):
                label = "lane" if token == "lane" else segment[token]
                if not order or order[-1] != label:
                    order.append(label)
            return order

        before, after = axes(source), axes(target)
        sizes = {
            label: self.lanes
            if label == "lane"
            else 1 << (bounds[label + 1] - bounds[label])
            for label in before
        }
        packed = self.name("narrow")
        self.lines.append(
            f"{packed} = narrow({self._pack(registers, k)}, RADIX)"
        )
        shaped, permuted = self.name("shaped"), self.name("permuted")
        shape = tuple(sizes[label] for label in before)
        order = tuple(before.index(label) for label in after)
        self.lines.extend(
            (
                f"{shaped} = tl.reshape({packed}, {shape!r})",
                f"{permuted} = tl.reshape(tl.permute({shaped}, {order!r}), {(self.slots, *(2,) * k)!r})",
            )
        )
        widened = self.name("widened")
        self.lines.append(f"{widened} = {permuted}.to(tl.int64)")
        return self._unpack(widened, k)

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
