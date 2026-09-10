"""Prepared Eager operations and packed matrix products for the suite."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch

from fhelium.config import CkksConfig
from fhelium.backend.ntt.context import NttContext
from fhelium.eager import Engine
from fhelium.values import Ciphertext, KeySwitchKey, RelinearizationKey, RotationKey, SecretKey

from .specification import PRIMITIVE_WORKLOADS, SEED, SuiteCell
from .polynomial import COEFFICIENTS, METHODS, POLYNOMIAL
from .primitives import analytic_ntt_rows, ntt_operands, rns_operands
from fhelium.experimental.bootstrap import BootstrapArithmetic


@dataclass
class PreparedWorkload:
    """A timed callable with its untimed decoding oracle and retained key size."""

    evaluate: Callable[[], Ciphertext | tuple[Ciphertext, ...] | torch.Tensor]
    decode: Callable[[Ciphertext | tuple[Ciphertext, ...] | torch.Tensor], torch.Tensor]
    expected: torch.Tensor
    key_bytes: int
    input_scale: float | None
    output_depth: int
    output_domain: str | None = None


class WorkloadInputs:
    """One configuration's Engine, secret and reusable evaluation keys."""

    def __init__(self, config: CkksConfig, device: torch.device) -> None:
        self.config = config
        self.device = device
        self.engine = Engine(
            config,
            rng_seed=SEED,
            rng_nonce=0,
            allow_automatic_key_generation=False,
        )
        self.secret = self.engine.create_secret_key(device=device)
        self.public = self.engine.create_public_key(self.secret, device=device)
        self._relin: RelinearizationKey | None = None
        self.rotations: dict[int, RotationKey] = {}
        self._switch_destination: SecretKey | None = None
        self._switch_key: KeySwitchKey | None = None
        self._ntt_contexts: dict[str, NttContext] = {}
        self._analytic_ntt_rows: tuple[torch.Tensor, torch.Tensor] | None = None

    @property
    def relin(self) -> RelinearizationKey:
        if self._relin is None:
            self._relin = self.engine.create_relinearization_key(
                self.secret, device=self.device
            )
        return self._relin

    def rotation(self, step: int) -> RotationKey:
        if step not in self.rotations:
            self.rotations[step] = self.engine.create_rotation_key(
                step, self.secret, device=self.device
            )
        return self.rotations[step]

    def key_bytes(self, steps: list[int], *, relin: bool = False) -> int:
        tensors = [self.rotation(step).data for step in set(steps)]
        if relin:
            tensors.append(self.relin.data)
        return sum(t.numel() * t.element_size() for t in tensors)

    def decrypt(
        self, output: Ciphertext | tuple[Ciphertext, ...] | torch.Tensor
    ) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output.cpu()
        if isinstance(output, tuple):
            return torch.stack(
                [
                    self.engine.decrypt_message(
                        c, self.secret, is_real=True
                    ).cpu()
                    for c in output
                ]
            )
        return self.engine.decrypt_message(
            output, self.secret, is_real=True
        ).cpu()

    def prepare(self, cell: SuiteCell) -> PreparedWorkload:
        if cell.workload in PRIMITIVE_WORKLOADS:
            return self.primitive(cell)
        if cell.workload == "polynomial":
            return self.polynomial(cell)
        return (
            self.matrix(cell) if cell.size is not None else self.operation(cell)
        )

    def primitive(self, cell: SuiteCell) -> PreparedWorkload:
        if cell.workload == "key_switch":
            return self.key_switch(cell)
        if cell.workload.startswith("ntt_"):
            if self.device.type == "cpu" and cell.plan != "radix2_indexed":
                raise NotImplementedError("Grouped-stage NTT is implemented for CUDA only")
            return self.ntt(cell)
        rns = self.engine._rns_context_for(self.device)
        basis = rns.basis_parameters(cell.depth)
        multiply = cell.workload == "rns_multiply"
        lhs, rhs, expected = rns_operands(
            basis.moduli, 1 << self.config.logN, cell.batch,
            rns.montgomery_parameters.R, multiply=multiply,
        )
        lhs = lhs.to(device=self.device, dtype=rns.dtype)
        rhs = rhs.to(device=self.device, dtype=rns.dtype)
        moduli = torch.tensor(basis.moduli, dtype=torch.int64).view(1, -1, 1)
        operation = rns.montgomery_mul if multiply else rns.add_standard
        return PreparedWorkload(
            lambda: operation(lhs, rhs),
            lambda output: cast(torch.Tensor, output).cpu().remainder(moduli),
            expected, 0, None, cell.depth, "coefficient",
        )

    def ntt(self, cell: SuiteCell) -> PreparedWorkload:
        rns = self.engine._rns_context_for(self.device)
        if cell.plan not in self._ntt_contexts:
            self._ntt_contexts[cell.plan] = NttContext(rns, cell.plan)
        context = self._ntt_contexts[cell.plan]
        if self._analytic_ntt_rows is None:
            self._analytic_ntt_rows = analytic_ntt_rows(
                self.config, rns.montgomery_parameters.R
            )
        basis = rns.basis_parameters(cell.depth)
        rows = slice(basis.parameter_row_start, basis.parameter_row_stop)
        coefficients, transformed = ntt_operands(
            basis.moduli, rns.montgomery_parameters.R,
            self._analytic_ntt_rows[0][rows], self._analytic_ntt_rows[1][rows],
            cell.batch,
        )
        forward = cell.workload == "ntt_forward"
        source = (coefficients if forward else transformed).to(
            device=self.device, dtype=rns.dtype
        )
        expected = transformed if forward else coefficients
        moduli = torch.tensor(basis.moduli, dtype=torch.int64).view(1, -1, 1)

        def evaluate() -> torch.Tensor:
            if forward:
                return context.forward_to_montgomery(source)
            output = source.clone()
            context.inverse_to_standard_(output)
            return output

        return PreparedWorkload(
            evaluate,
            lambda output: cast(torch.Tensor, output).cpu().remainder(moduli),
            expected, 0, None, cell.depth, "ntt" if forward else "coefficient",
        )

    def key_switch(self, cell: SuiteCell) -> PreparedWorkload:
        e = self.engine
        if self._switch_destination is None:
            self._switch_destination = e.create_secret_key(device=self.device)
            self._switch_key = e.create_key_switch_key(
                self.secret, self._switch_destination, device=self.device
            )
        destination = self._switch_destination
        key = cast(KeySwitchKey, self._switch_key)
        index = torch.arange(e.num_slots, dtype=torch.float64)
        members = torch.arange(cell.batch, dtype=torch.float64)[:, None]
        message = 0.08 * torch.sin(index[None, :] * 0.017 + members * 0.13) + 0.02
        source = e.encrypt_message(message, self.public, depth=cell.depth, device=self.device)
        return PreparedWorkload(
            lambda: e.switch_key(source, key, output_domain="ntt"),
            lambda output: e.decrypt_message(
                cast(Ciphertext, output), destination, is_real=True
            ).cpu(),
            message, key.data.numel() * key.data.element_size(), source.scale,
            cell.depth,
        )

    def operation(self, cell: SuiteCell) -> PreparedWorkload:
        e = self.engine
        index = torch.arange(e.num_slots, dtype=torch.float64)
        members = torch.arange(cell.batch, dtype=torch.float64)[:, None]
        message = (
            0.08 * torch.sin(index[None, :] * 0.017 + members * 0.13) + 0.02
        )
        source = e.encrypt_message(
            message, self.public, depth=cell.depth, device=self.device
        )
        ntt = e.coefficient_domain_to_ntt_domain(source)
        expected = message
        steps: list[int] = []
        use_relin = False
        output_depth = cell.depth
        input_scale = source.scale
        if cell.workload == "rotate":
            steps = [1]
            key = self.rotation(1)
            evaluate = lambda: e.rotate_with_key(ntt, key, output_domain="ntt")
            expected = torch.roll(message, 1, -1)
        elif cell.workload in ("rotate_many", "rotate_many_independent"):
            steps = list(range(1, 8))
            keys = [self.rotation(step) for step in steps]
            evaluate = lambda: tuple(
                e.rotate_many_with_keys(
                    source, keys, use_hoisting=cell.workload == "rotate_many", output_domain="ntt"
                )
            )
            expected = torch.stack(
                [torch.roll(message, step, -1) for step in steps]
            )
        elif cell.workload == "relinearize":
            product = e.multiply(ntt, ntt)
            input_scale = product.scale
            evaluate = lambda: e.relinearize(
                product, self.relin, output_domain="ntt"
            )
            use_relin = True
            expected = message.square()
        elif cell.workload == "rescale_ct2":
            product = e.relinearize(e.multiply(ntt, ntt), self.relin, output_domain="ntt")
            input_scale = product.scale
            evaluate = lambda: e.rescale_to_next_depth(product)
            expected = message.square()
            output_depth += 1
        elif cell.workload == "multiply_plaintext":
            plaintext = e.prepare_plaintext_for_multiplication(
                e.encode(message, depth=cell.depth, device=self.device)
            )
            evaluate = lambda: e.multiply_plaintext(ntt, plaintext)
            expected = message.square()
        elif cell.workload == "square":
            evaluate = lambda: e.multiply(ntt, ntt)
            expected = message.square()
        elif cell.workload in ("multiply_ciphertexts", "add"):
            other_message = 0.07 * torch.cos(index[None, :] * 0.019 + members * 0.11) - 0.01
            other = e.encrypt_message(other_message, self.public, depth=cell.depth, device=self.device, output_domain="ntt")
            if cell.workload == "add":
                evaluate = lambda: e.add(ntt, other)
                expected = message + other_message
            else:
                evaluate = lambda: e.multiply(ntt, other)
                expected = message * other_message
        elif cell.workload == "sum_plaintext_products":
            factors = tuple((i + 1) / 16 for i in range(8))
            plains = tuple(e.prepare_plaintext_for_multiplication(
                e.encode(message * factor, depth=cell.depth, device=self.device)
            ) for factor in factors)
            evaluate = lambda: e.sum_plaintext_products((ntt,) * len(plains), plains)
            expected = message.square() * sum(factors)
        elif cell.workload in ("multiply_coefficient", "advance_depth"):
            arithmetic = BootstrapArithmetic(e, constant_cache={}, retain_ntt=True)
            if cell.workload == "multiply_coefficient":
                evaluate = lambda: arithmetic.multiply_scalar(ntt, -0.3)
                expected = message * -0.3
            else:
                evaluate = lambda: arithmetic.advance_depth(ntt)
            output_depth += 1
        else:
            raise ValueError(f"Unknown suite operation {cell.workload}")
        return PreparedWorkload(
            evaluate,
            self.decrypt,
            expected,
            self.key_bytes(steps, relin=use_relin),
            input_scale,
            output_depth,
        )

    def polynomial(self, cell: SuiteCell) -> PreparedWorkload:
        e = self.engine
        evaluator = METHODS[cell.plan]
        grid = torch.linspace(-1.0, 1.0, e.num_slots, dtype=torch.float64)
        message = torch.stack([torch.roll(grid, shifts=i * 17) for i in range(cell.batch)])
        source = e.encrypt_message(message, self.public, depth=cell.depth, device=self.device)
        key = self.relin
        arithmetic = BootstrapArithmetic(e, constant_cache={}, retain_ntt=True)
        coordinates = message.numpy().astype(np.longdouble)
        reference = np.zeros_like(coordinates)
        for coefficient in reversed(COEFFICIENTS):
            reference = reference * coordinates + np.longdouble(coefficient)
        expected = torch.from_numpy(reference.astype(np.float64))
        return PreparedWorkload(
            lambda: evaluator.evaluate(arithmetic, source, POLYNOMIAL, relinearization_key=key),
            self.decrypt, expected, self.key_bytes([], relin=True), source.scale,
            cell.depth + evaluator.required_depths(POLYNOMIAL),
        )

    def matrix(self, cell: SuiteCell) -> PreparedWorkload:
        e = self.engine
        assert cell.size is not None
        size = cell.size
        baby = 2 ** int(math.log2(math.sqrt(size)))
        slots = e.num_slots
        columns_per_ciphertext = slots // (2 * size)
        ciphertext_groups = math.ceil(size / columns_per_ciphertext)
        row = torch.arange(size, dtype=torch.float64)[None, :, None]
        column = torch.arange(size, dtype=torch.float64)[None, None, :]
        member = torch.arange(cell.batch, dtype=torch.float64)[:, None, None]
        left = 0.01 * torch.sin((row + 1) * (column + 2) * 0.17 + member * 0.13)
        left += 0.005 * torch.cos((row + column + 1) * 0.23)
        right = 0.03 * torch.sin(
            (row + 3) * (column + 1) * 0.11 + member * 0.19
        )
        padded = torch.zeros(
            (cell.batch, ciphertext_groups * columns_per_ciphertext, size),
            dtype=torch.float64,
        )
        padded[:, :size] = right.transpose(-1, -2)
        packed = (
            padded.reshape(
                cell.batch, ciphertext_groups, columns_per_ciphertext, size
            )
            .unsqueeze(-2)
            .repeat(1, 1, 1, 2, 1)
            .reshape(cell.batch, ciphertext_groups, slots)
        )
        source = e.encrypt_message(
            packed, self.public, depth=cell.depth, device=self.device
        )
        ntt = e.coefficient_domain_to_ntt_domain(source)
        steps = sorted(set(range(1, baby)) | set(range(baby, size, baby)))
        keys = {step: self.rotation(step) for step in steps}
        slot_rows = torch.arange(slots) % size
        plaintexts = []
        ciphertexts = []
        for offset in range(size):
            giant = (offset // baby) * baby
            values = left[:, slot_rows, (slot_rows - offset) % size].unsqueeze(
                1
            )
            values = torch.roll(values, -giant, -1)
            values = values.expand(-1, ciphertext_groups, -1)
            if cell.workload == "matrix_ptct":
                plaintexts.append(
                    e.prepare_plaintext_for_multiplication(
                        e.encode(values, depth=cell.depth, device=self.device)
                    )
                )
            else:
                ciphertexts.append(
                    e.encrypt_message(
                        values,
                        self.public,
                        depth=cell.depth,
                        device=self.device,
                        output_domain="ntt",
                    )
                )

        def evaluate() -> Ciphertext:
            if cell.plan == "shared":
                rotations = e.rotate_many_with_keys(
                    source,
                    [keys[k] for k in range(1, baby)],
                    use_hoisting=True,
                    output_domain="ntt",
                )
            else:
                rotations = [
                    e.rotate_with_key(source, keys[k], output_domain="ntt")
                    for k in range(1, baby)
                ]
            babies = (ntt, *rotations)
            result = None
            for giant in range(0, size, baby):
                if cell.workload == "matrix_ptct":
                    inner = e.sum_plaintext_products(
                        babies, plaintexts[giant : giant + baby]
                    )
                else:
                    inner = e.multiply(babies[0], ciphertexts[giant])
                    for b in range(1, baby):
                        inner = e.add(
                            inner, e.multiply(babies[b], ciphertexts[giant + b])
                        )
                    inner = e.relinearize(
                        inner, self.relin, output_domain="ntt"
                    )
                inner = e.rescale_to_next_depth(inner)
                part = (
                    inner
                    if giant == 0
                    else e.rotate_with_key(
                        inner, keys[giant], output_domain="ntt"
                    )
                )
                result = part if result is None else e.add(result, part)
            assert result is not None
            return result

        def decode(output: Ciphertext | tuple[Ciphertext, ...] | torch.Tensor) -> torch.Tensor:
            values = self.decrypt(output).reshape(
                cell.batch, ciphertext_groups, columns_per_ciphertext, 2, size
            )
            return (
                values[:, :, :, 1, :]
                .reshape(cell.batch, -1, size)[:, :size]
                .transpose(-1, -2)
            )

        return PreparedWorkload(
            evaluate,
            decode,
            left @ right,
            self.key_bytes(steps, relin=cell.workload == "matrix_ctct"),
            source.scale,
            cell.depth + 1,
        )
