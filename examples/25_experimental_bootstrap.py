r"""Deplete and refresh one full-slot CKKS ciphertext at $\mathtt{logN}=16$.

The measured profile uses 50-bit scale and structural-base primes, generator 5,
and a periodic-reduction raw `input_bound` of 1024. The example message range is
an empirical end-to-end input, not a proof of the encrypted branch bound.
"""

from __future__ import annotations

import argparse
import time

import torch

import fhelium as fh
from fhelium.eager import Engine
from fhelium.runtime import CudaGraphProgram
from fhelium.experimental.bootstrap import (
    cosine_bootstrap,
    BinaryDecompositionChebyshevEvaluator,
    BlockedChebyshevEvaluator,
    EncapsulatedModRaise,
)


def deplete_to_bootstrap_input(
    engine: Engine,
    ciphertext: fh.Ciphertext,
    target_depth: int,
) -> fh.Ciphertext:
    r"""Consume public Q depths with multiplication by semantic $1$.

    Each iteration multiplies a two-component coefficient-domain standard-RNS
    Q ciphertext by an unbatched NTT/Montgomery plaintext at scale $\Delta_0$,
    then drops the complete leading Q depth group. The output remains coefficient-domain
    standard RNS with axes `[component, *batch, limb, coefficient]`, unchanged
    batch shape and component count, next-depth `prime_ids`, and actual
    scale $\Delta_{\rm out}=\Delta_{\rm in}\Delta_0/M_{\rm drop}$. The
    returned ciphertext is functional; the argument's storage is not mutated.
    """

    ones = torch.ones(
        engine.num_slots,
        dtype=torch.float64,
        device=torch.get_default_device(),
    )
    while ciphertext.depth < target_depth:
        identity = engine.prepare_plaintext_for_multiplication(
            engine.encode(
                ones, depth=ciphertext.depth, scale=engine.config.default_scale
            )
        )
        ciphertext = engine.rescale_to_next_depth(
            engine.ntt_domain_to_coefficient_domain(
                engine.multiply_plaintext(
                    engine.coefficient_domain_to_ntt_domain(ciphertext),
                    identity,
                )
            )
        )
    return ciphertext


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument(
        "--compile",
        dest="execution",
        action="store_const",
        const="compile",
        help="compile the complete prepared refresh (default)",
    )
    execution.add_argument(
        "--eager",
        dest="execution",
        action="store_const",
        const="eager",
        help="execute the refresh through Eager",
    )
    execution.add_argument(
        "--core-graph",
        dest="execution",
        action="store_const",
        const="core-graph",
        help="prepare and capture the normalized core",
    )
    parser.set_defaults(execution="compile")
    parser.add_argument(
        "--message-domain", choices=("complex", "real"), default="complex"
    )
    parser.add_argument(
        "--product-implementation",
        choices=("native", "auto", "triton"),
        default="auto",
    )
    parser.add_argument("--product-chunk-size", type=int, default=16)
    parser.add_argument(
        "--profile", choices=("7/44", "8/28", "8/28-s2c3", "3/30"), default=None
    )
    parser.add_argument(
        "--giant-step-reduction",
        choices=("aggregate", "independent"),
        default=None,
    )
    parser.add_argument(
        "--polynomial-evaluator",
        choices=("profile", "binary", "blocked"),
        default="profile",
    )
    parser.add_argument(
        "--rotation-key-strategy",
        choices=(
            "signed_power_of_two",
            "signed_power_of_two_with_baby_steps",
            "all_required",
        ),
        default="all_required",
    )
    parser.add_argument(
        "--encapsulated",
        action="store_true",
        help="use the explicit int32 low-basis encapsulation construction",
    )
    parser.add_argument("--intermediate-secret-weight", type=int, default=32)
    parser.add_argument("--no-retain-diagonals", action="store_true")
    args = parser.parse_args()
    torch.set_default_device("cuda:0")
    auxiliary = None
    target_scales = None
    if args.encapsulated:
        from _bootstrap_encapsulation_parameters import parameters

        config, auxiliary, target_scales = parameters()
        if args.profile not in {None, "3/30"}:
            parser.error(
                "the encapsulation parameter example uses profile 3/30"
            )
        profile = "3/30"
    else:
        config = fh.CkksConfig.parse(
            fh.Preset.slots32768_scale50_depth27_int64, galois_generator=5
        )
        profile = args.profile or "7/44"
    engine = Engine(
        config,
        allow_automatic_key_generation=False,
    )
    implementation = {
        "native": None,
        "auto": "auto",
        "triton": "triton-montgomery-weighted-sum",
    }[args.product_implementation]
    polynomial_evaluator = {
        "profile": None,
        "binary": BinaryDecompositionChebyshevEvaluator(skip_near_zero=1e-15),
        "blocked": BlockedChebyshevEvaluator(baby_step=8),
    }[args.polynomial_evaluator]
    secret_key = engine.create_secret_key(
        hamming_weight=config.N // 2 if args.encapsulated else None
    )
    modulus_raising = None
    if auxiliary is not None:
        intermediate = engine.create_secret_key(
            hamming_weight=args.intermediate_secret_weight
        )
        low_key, return_key = engine.create_encapsulation_keys(
            secret_key, intermediate, **auxiliary
        )
        modulus_raising = EncapsulatedModRaise(low_key, return_key, **auxiliary)
        del intermediate
    bootstrap = cosine_bootstrap(
        engine,
        profile=profile,
        modulus_raising=modulus_raising,
        input_depth=config.max_depth if args.encapsulated else None,
        target_scales=target_scales,
        giant_step_reduction=args.giant_step_reduction,
        polynomial_evaluator=polynomial_evaluator,
        retain_diagonals=not args.no_retain_diagonals,
        message_domain=args.message_domain,
        product_implementation=implementation,
        product_chunk_size=args.product_chunk_size,
    )

    for stages, evaluator in (
        (bootstrap.coeffs_to_slots, bootstrap.coeffs_to_slots_evaluator),
        (bootstrap.slots_to_coeffs, bootstrap.slots_to_coeffs_evaluator),
    ):
        for stage in stages:
            print(f"{stage.name}: {evaluator.schedule_for(stage)}")

    public_key = engine.create_public_key(secret_key)
    evaluation_keys = bootstrap.create_evaluation_keys(
        secret_key,
        rotation_key_strategy=args.rotation_key_strategy,
    )

    values = torch.linspace(
        -0.1,
        0.1,
        engine.num_slots,
        dtype=torch.float64,
    )
    depleted = (
        engine.encrypt_message(
            values,
            public_key,
            depth=bootstrap.input_depth,
            scale=bootstrap.input_scale,
        )
        if args.encapsulated
        else deplete_to_bootstrap_input(
            engine,
            engine.encrypt_message(values, public_key),
            bootstrap.input_depth,
        )
    )

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(torch.get_default_device())
    started = time.perf_counter()
    if args.execution == "compile":
        from fhelium import compile as fc

        prepared = bootstrap.prepare(depleted, evaluation_keys=evaluation_keys)
        try:
            optimized = fc.compile(prepared)
            refreshed = optimized(depleted)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            print(f"prepared complete Compile seconds: {elapsed:.6f}")
        finally:
            prepared.close()
    elif args.execution == "core-graph":
        coordinate, ratio = bootstrap.entry(depleted)
        prepared = bootstrap.prepare_core(
            coordinate, evaluation_keys=evaluation_keys
        )
        torch.cuda.synchronize()
        print(
            f"entry and core preparation seconds: {time.perf_counter() - started:.6f}"
        )
        print(f"prepared diagonal bytes: {prepared.cached_diagonal_bytes}")
        program = None
        try:
            capture_started = time.perf_counter()
            program = CudaGraphProgram.capture(
                prepared, example_inputs=(coordinate,), warmup=1
            )
            torch.cuda.synchronize()
            print(
                f"core capture seconds: {time.perf_counter() - capture_started:.6f}"
            )
            started = time.perf_counter()
            refreshed = bootstrap.finish(
                program.replay(coordinate, copy_output=True, synchronize=True),
                ratio,
            )
            elapsed = time.perf_counter() - started
        finally:
            if program is not None:
                program.close()
            prepared.close()
        print(f"core replay with input/output copies seconds: {elapsed:.6f}")
    else:
        refreshed = bootstrap(depleted, evaluation_keys=evaluation_keys)
    torch.cuda.synchronize()
    if args.execution == "eager":
        elapsed = time.perf_counter() - started

    decoded = engine.decrypt_message(refreshed, secret_key, is_real=True).cpu()
    error = (decoded - values.cpu()).abs()
    print(f'input depth: {depleted.depth}')
    print(f'output depth: {refreshed.depth}')
    print(f'remaining depth: {bootstrap.remaining_depth}')
    print(f'refresh window: {bootstrap.refresh_window}')
    print(
        'pipeline depth: '
        f'{bootstrap.output_depth - bootstrap.modulus_raise_target_depth}'
    )
    print(
        f'periodic raw input bound: {bootstrap.modular_reduction.input_bound}'
    )
    print(f'output actual scale: {refreshed.scale:.6g}')
    print(f'rotation keys: {len(evaluation_keys.rotation_keys)}')
    if not args.core_graph:
        print(f'bootstrap seconds: {elapsed:.3f}')
    print(f'max error: {error.max().item():.6g}')
    print(f'mean error: {error.mean().item():.6g}')
    print(
        'peak allocated GPU GiB: '
        f'{torch.cuda.max_memory_allocated(torch.get_default_device()) / 2**30:.3f}'
    )


if __name__ == '__main__':
    main()
