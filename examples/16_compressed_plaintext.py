"""Compress a periodic operation-ready plaintext without changing Plaintext.

The example starts from a standard periodic CKKS slot message, encodes it with
the ordinary codec, verifies the exact repeated NTT structure, and converts it
to a separate CompressedPlaintext. The evaluator kernel reads the compact
operand directly; it does not materialize a dense plaintext during multiply.
"""

from __future__ import annotations

import argparse
import time

import torch
from common import add_engine_args, make_engine

import fhelium as fh


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _median_ms(
    operation,
    *,
    iterations: int,
    device: torch.device,
) -> float:
    for _ in range(3):
        operation()
    _synchronize(device)
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        operation()
        _synchronize(device)
        samples.append((time.perf_counter() - start) * 1e3)
    return float(torch.tensor(samples).median())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_engine_args(
        parser,
        default_preset=fh.Preset.slots8192_scale40_levels7_int64.value,
    )
    parser.add_argument("--period", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    engine = make_engine(args)
    period = args.period
    if period <= 0 or period & (period - 1):
        raise ValueError("--period must be a positive power of two")
    if engine.num_slots % period:
        raise ValueError("--period must divide the CKKS slot count")

    # Repetition modes describe the encoded tensor's last axis. Their compact
    # payload is identical; only index expansion differs.
    compact_example = torch.tensor([1, 2])
    print("encoded cyclic:    ", compact_example.repeat(4).tolist())
    print(
        "encoded contiguous:",
        compact_example.repeat_interleave(4).tolist(),
    )

    unique_index = torch.arange(period, dtype=torch.float64)
    unique_slots = torch.complex(
        0.03 * torch.cos(unique_index * 0.07) + 0.001 * unique_index / period,
        0.02 * torch.sin(unique_index * 0.05),
    )
    factor = unique_slots.repeat(engine.num_slots // period)

    # A period-r semantic slot vector yields 2r exact encoded NTT values,
    # stored as contiguous repeated blocks for the current CKKS codec.
    dense = engine.prepare_plaintext_for_multiplication(engine.encode(factor))
    compressed = fh.CompressedPlaintext.from_plaintext(
        dense,
        unique_count=2 * period,
        compression_layout="contiguous",
    )
    dense_addend = engine.prepare_plaintext_for_addition(engine.encode(factor))
    sparse_addend = fh.CompressedPlaintext.from_plaintext(
        dense_addend,
        unique_count=2 * period,
        compression_layout="strided_sparse",
    )
    dense_data = dense.data
    compressed_data = compressed.data
    if dense_data is None or compressed_data is None:
        raise RuntimeError("prepared plaintext payloads must be materialized")
    dense_bytes = dense_data.numel() * dense_data.element_size()
    compressed_bytes = compressed_data.numel() * compressed_data.element_size()
    print(f"ring dimension:       {engine.config.N}")
    print(f"periodic slots:       {period}")
    print(f"encoded unique count: {compressed.unique_count}")
    print(f"dense bytes:          {dense_bytes:,}")
    print(f"compressed bytes:     {compressed_bytes:,}")
    print(f"storage reduction:    {dense_bytes / compressed_bytes:.1f}x")
    print(f"sparse-add bytes:     {sparse_addend.nbytes:,}")

    slot_index = torch.arange(engine.num_slots, dtype=torch.float64)
    message = torch.complex(
        0.01 * torch.sin(slot_index * 0.013),
        0.008 * torch.cos(slot_index * 0.011),
    )
    ciphertext = engine.encrypt_message(message)
    ciphertext_ntt = engine.coefficient_domain_to_ntt_domain(ciphertext)
    dense_result = engine.multiply_plaintext(ciphertext_ntt, dense)
    compressed_result = engine.multiply_plaintext(ciphertext_ntt, compressed)
    if not torch.equal(compressed_result.data, dense_result.data):
        raise AssertionError("Compressed and dense ciphertexts differ")
    decoded = engine.decrypt_message(
        engine.ntt_domain_to_coefficient_domain(compressed_result)
    ).cpu()
    max_error = torch.max(torch.abs(decoded - message * factor)).item()
    print(f"maximum cleartext error: {max_error:.3e}")

    dense_sum = engine.add_plaintext(ciphertext, dense_addend)
    sparse_sum = engine.add_plaintext(ciphertext, sparse_addend)
    if not torch.equal(sparse_sum.data, dense_sum.data):
        raise AssertionError("Sparse and dense addition ciphertexts differ")

    dense_ms = _median_ms(
        lambda: engine.multiply_plaintext(ciphertext_ntt, dense),
        iterations=args.iterations,
        device=engine.device,
    )
    compressed_ms = _median_ms(
        lambda: engine.multiply_plaintext(ciphertext_ntt, compressed),
        iterations=args.iterations,
        device=engine.device,
    )
    print(f"dense evaluator median:      {dense_ms:.3f} ms")
    print(f"compressed evaluator median: {compressed_ms:.3f} ms")
    print(f"evaluator speedup:           {dense_ms / compressed_ms:.2f}x")

    dense_add_work = ciphertext.clone()
    sparse_add_work = ciphertext.clone()
    dense_add_ms = _median_ms(
        lambda: engine.add_plaintext_(dense_add_work, dense_addend),
        iterations=args.iterations,
        device=engine.device,
    )
    sparse_add_ms = _median_ms(
        lambda: engine.add_plaintext_(sparse_add_work, sparse_addend),
        iterations=args.iterations,
        device=engine.device,
    )
    print(f"dense in-place addition median:  {dense_add_ms:.3f} ms")
    print(f"sparse in-place addition median: {sparse_add_ms:.3f} ms")
    print(
        f"in-place addition speedup:       {dense_add_ms / sparse_add_ms:.2f}x"
    )


if __name__ == "__main__":
    main()
