#!/usr/bin/env python3

"""Store named CKKS values, replace a generation, and reject a stale reference.

The application owns logical names. ArtifactStore owns durable payloads and the
current generation; get(name) loads that generation, while get(ref) requires the
specific generation previously returned by put.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from common import (
    add_engine_args,
    make_engine,
    print_table,
    small_complex_vector,
)

import fhelium as fh
from fhelium.artifacts import ArtifactStore
from fhelium.errors import StaleArtifactReferenceError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_engine_args(parser, default_preset="slots8192-scale40-depth7-int64")
    parser.add_argument(
        "--store",
        type=Path,
        help="Keep the repository here; otherwise use a temporary directory.",
    )
    args = parser.parse_args()
    engine = make_engine(args)
    message = small_complex_vector(engine.num_slots, seed=10)
    source = engine.encrypt_message(message)
    replacement = engine.negate(source)
    context = (
        TemporaryDirectory(prefix="fhelium-artifacts-")
        if args.store is None
        else nullcontext(args.store)
    )
    with context as root:
        store = ArtifactStore(root)
        requests = store.collection("requests")
        first = requests.put("activation", source, overwrite=True)
        restored = store.get(
            first, device=source.device, expected_type=fh.Ciphertext
        )
        torch.testing.assert_close(restored.data, source.data, rtol=0, atol=0)

        current = requests.put("activation", replacement, overwrite=True)
        try:
            store.get(first)
        except StaleArtifactReferenceError:
            print(
                "The previous generation's reference is stale after replacement."
            )
        else:
            raise AssertionError(
                "A replaced generation must not resolve to the new payload"
            )

        loaded = requests.get(
            "activation", device=source.device, expected_type=fh.Ciphertext
        )
        assert loaded is not None
        torch.testing.assert_close(
            loaded.data, replacement.data, rtol=0, atol=0
        )
        assert store.inspect(current).ref == current
        print_table(
            ["name", "type", "logical bytes"],
            [
                [ref.name, ref.value_type, ref.nbytes]
                for ref in store.list(prefix="requests")
            ],
        )
        print(f"Store: {Path(root).resolve()}")


if __name__ == "__main__":
    main()
