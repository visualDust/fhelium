r"""Tensor operations for collective CKKS key and output protocols.

The package exposes share-generation and aggregation functions for collective
key generation, evaluation-key generation, collective decryption arithmetic,
and public-key switching arithmetic.  Its supported scope is arithmetic
correctness for compatible FHElium values and engine tensors. The implementation
accepts a local CPU or CUDA ``CkksEngine``. The package provides no
authentication, transcript binding, malicious-party security, reviewed
output-error sampler, supported smudging/useful-precision parameter profile,
or privacy guarantee.  Functions whose names begin with
``unsafe_``
require caller-supplied output randomness and errors.
"""

from fhelium.experimental.mpc._ops import (
    aggregate_ckg,
    aggregate_conjugation_key,
    aggregate_rkg_round1,
    aggregate_rkg_round2,
    aggregate_rotation_key,
    ckg_share,
    conjugation_key_share,
    rkg_round1_share,
    rkg_round2_share,
    rotation_key_share,
    sample_common_uniform,
    sample_secret_share,
    unsafe_collective_decryption_share,
    unsafe_fuse_collective_decryption,
    unsafe_fuse_public_key_switch,
    unsafe_public_key_switch_share,
)

__all__ = [
    "aggregate_ckg",
    "aggregate_conjugation_key",
    "aggregate_rkg_round1",
    "aggregate_rkg_round2",
    "aggregate_rotation_key",
    "ckg_share",
    "conjugation_key_share",
    "rkg_round1_share",
    "rkg_round2_share",
    "rotation_key_share",
    "sample_common_uniform",
    "sample_secret_share",
    "unsafe_collective_decryption_share",
    "unsafe_fuse_collective_decryption",
    "unsafe_fuse_public_key_switch",
    "unsafe_public_key_switch_share",
]
