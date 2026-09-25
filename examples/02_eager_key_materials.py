#!/usr/bin/env python3

"""Create, inspect, and install process-local CKKS keys.

Each ``RotationKey`` records its normalized signed ``rotation_step``;
``RotationKeySet`` validates the same identity when constructing or updating the mapping.
"""

from __future__ import annotations

import argparse

from common import add_engine_args, format_bytes, make_engine, print_table


def _size(value) -> int:
    return value.data.nbytes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_engine_args(parser)
    parser.add_argument("--rotations", default="1,2,4")
    args = parser.parse_args()

    engine = make_engine(args)
    rotation_steps = [
        int(item) for item in args.rotations.split(",") if item.strip()
    ]
    secret_key = engine.create_secret_key()
    public_key = engine.create_public_key(secret_key)
    relinearization_key = engine.create_relinearization_key(secret_key)
    engine.set_secret_key(secret_key)
    engine.set_public_key(public_key)
    engine.set_relinearization_key(relinearization_key)
    for rotation_step in rotation_steps:
        engine.set_rotation_key(
            engine.create_rotation_key(rotation_step, secret_key)
        )

    rows = [
        [
            "secret",
            "[limb, coeff]",
            tuple(secret_key.data.shape),
            format_bytes(_size(secret_key)),
        ],
        [
            "public",
            "[key_component, limb, coeff]",
            tuple(public_key.data.shape),
            format_bytes(_size(public_key)),
        ],
        [
            "relinearization",
            "[digit, key_component, limb, coeff]",
            tuple(relinearization_key.data.shape),
            format_bytes(_size(relinearization_key)),
        ],
    ]
    for rotation_step in rotation_steps:
        key = engine.rotation_keys[rotation_step]
        rows.append(
            [
                f"rotation[{rotation_step}]",
                "[digit, key_component, limb, coeff]",
                tuple(key.data.shape),
                format_bytes(_size(key)),
            ]
        )
    print_table(["material", "axes", "local shape", "local bytes"], rows)
    print(f"RotationKeySet normalized steps: {list(engine.rotation_keys)}")


if __name__ == "__main__":
    main()
