#!/usr/bin/env python3

"""Inspect and optionally persist dense process-local CKKS key layouts.

Each ``RotationKey`` records its canonical signed ``rotation_step``;
``RotationKeySet`` validates the same identity when constructing or updating the mapping.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from common import add_engine_args, format_bytes, make_engine, print_table

import fhelium as fh
from fhelium.artifacts import ArtifactStore


def _size(value) -> int:
    return value.data.nbytes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_engine_args(parser)
    parser.add_argument("--rotations", default="1,2,4")
    parser.add_argument(
        "--store",
        type=Path,
        help="Persist public/relinearization/rotation keys under this store root.",
    )
    parser.add_argument(
        "--persist-secret",
        action="store_true",
        help="Explicitly opt in to an unencrypted SecretKey artifact.",
    )
    args = parser.parse_args()

    engine = make_engine(args)
    rotation_steps = [
        int(item) for item in args.rotations.split(",") if item.strip()
    ]
    secret_key = engine.secret_key
    public_key = engine.public_key
    relinearization_key = engine.relinearization_key
    for rotation_step in rotation_steps:
        engine.rotation_key(rotation_step)

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
    print(f"RotationKeySet canonical steps: {list(engine.rotation_keys)}")
    print(
        "RotationKey tensor canonical step: "
        f"{engine.rotation_keys[rotation_steps[0]].rotation_step}"
    )

    if args.store is not None:
        store = ArtifactStore(args.store)
        store.put("keys/public", public_key, overwrite=True)
        relinearization_ref = store.put(
            "keys/relinearization", relinearization_key, overwrite=True
        )
        rotation_keys = store.collection("keys/rotation")
        for rotation_step in rotation_steps:
            rotation_keys.put(
                str(rotation_step),
                engine.rotation_keys[rotation_step],
                overwrite=True,
            )
        if args.persist_secret:
            store.put(
                "keys/secret",
                secret_key,
                allow_secret=True,
                overwrite=True,
            )

        restored = store.get(relinearization_ref, device=engine.device)
        assert type(restored) is fh.RelinearizationKey
        torch.testing.assert_close(restored.data, relinearization_key.data)
        print(
            f"Persisted {len(store.list(prefix='keys'))} key artifacts "
            f"under {args.store.resolve()}"
        )


if __name__ == "__main__":
    main()
