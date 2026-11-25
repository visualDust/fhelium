r"""Refresh a full-slot CKKS ciphertext with :mod:`fhelium.bootstrap`.

The default chain is the built-in preset ``slots32768-scale40-depth29-boot13``:
$N=2^{16}$, int32 residues, Galois generator 5, and a 16-depth bootstrap segment
followed by 13 application depths. ``bootstrap.plan(engine,
intermediate_weight=48)`` selects the ``"3/38"`` profile and an encapsulated
modulus raise through an intermediate secret of Hamming weight 48. A value at
the maximum depth 29 is refreshed to depth 16 through Eager execution,
``fhelium.compile``, or a captured CUDA Graph. ``--construction dense`` instead
uses a 50-bit catalog suffix chain with the ``"7/44"`` profile and a centered
raise, which publishes no encapsulation keys. The message range is an empirical
end-to-end input; it does not establish the EvalMod input bound for an
application.
"""

from __future__ import annotations

import argparse
import time

import torch

import fhelium as fh
from fhelium import bootstrap
from fhelium.eager import Engine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--construction",
        choices=("fast", "dense"),
        default="fast",
        help=(
            "fast: slots32768-scale40-depth29-boot13 with the 3/38 profile and "
            "an encapsulated raise; dense: 50-bit suffix chain with the 7/44 "
            "profile and a centered raise"
        ),
    )
    parser.add_argument(
        "--execution", choices=("compile", "graph", "eager"), default="compile"
    )
    parser.add_argument(
        "--intermediate-secret-weight",
        type=int,
        default=48,
        help="Hamming weight of the intermediate secret of the encapsulated raise",
    )
    return parser.parse_args()


def refresh(engine, plan, keys, value, execution):
    """Return the refreshed value and the steady-state seconds of one call."""
    if execution == "eager":
        bootstrap.bootstrap(engine, value, plan, keys=keys)
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = bootstrap.bootstrap(engine, value, plan, keys=keys)
        torch.cuda.synchronize()
        return result, time.perf_counter() - started
    from fhelium import compile as fc

    prepared = bootstrap.prepare(engine, plan, value, keys=keys)
    compiled = fc.compile(prepared, graph=execution == "graph")
    try:
        # The first call runs the prepared computation while compilation
        # proceeds; with a Graph, the second call captures it.
        for _ in range(2):
            compiled(value)
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = compiled(value)
        torch.cuda.synchronize()
        return result, time.perf_counter() - started
    finally:
        compiled.clear()
        prepared.close()


def main() -> None:
    args = parse_args()
    torch.set_default_device("cuda:0")
    if args.construction == "fast":
        config = fh.CkksConfig.parse(fh.Preset.slots32768_scale40_depth29_boot13)
    else:
        config = fh.CkksConfig.catalog_suffix_chain(
            logN=16, scale_bits=50, depth=27, p_count=4, galois_generator=5
        )
    engine = Engine(config, allow_automatic_key_generation=False)
    secret_key = engine.create_secret_key()
    public_key = engine.create_public_key(secret_key)

    if args.construction == "fast":
        # The encapsulated raise bounds the overflow polynomial through the
        # intermediate secret weight. The security of publishing its two
        # encapsulation keys with the remaining evaluation keys depends on
        # that weight and the parameters, and is analyzed by the application.
        plan = bootstrap.plan(
            engine, intermediate_weight=args.intermediate_secret_weight
        )
        print(f"worst-case |I|+1: {plan.modulus_raise_bound}")
    else:
        plan = bootstrap.plan(engine, "7/44")
    for stage in plan.stages():
        print(f"{stage.name}: depth {stage.input_depth} -> {stage.output_depth}")
    print(f"refresh window: {plan.refresh_window}")
    keys = bootstrap.create_evaluation_keys(engine, plan, secret_key)
    key_bytes = sum(key.data.numel() * key.data.element_size() for key in keys)
    print(f"evaluation keys: {len(keys)} ({key_bytes / 2**30:.2f} GiB)")

    generator = torch.Generator(device="cpu").manual_seed(0)
    real, imag = torch.rand(
        2, engine.num_slots, generator=generator, dtype=torch.float64, device="cpu"
    )
    values = (2 * real - 1) + 1j * (2 * imag - 1)
    ciphertext = engine.encrypt_message(values, public_key, depth=plan.source_depth)

    torch.cuda.reset_peak_memory_stats()
    refreshed, seconds = refresh(engine, plan, keys, ciphertext, args.execution)
    error = (engine.decrypt_message(refreshed, secret_key).cpu() - values).abs()
    print(f"depth {ciphertext.depth} -> {refreshed.depth}")
    print(f"bootstrap seconds ({args.execution}): {seconds:.4f}")
    print(f"max error: {error.max().item():.3e}")
    print(f"peak allocated GPU GiB: {torch.cuda.max_memory_allocated() / 2**30:.3f}")


if __name__ == "__main__":
    main()
