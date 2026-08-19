from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Any

import torch


def _load_triton_csprng():
    try:
        from triton_csprng import (  # type: ignore[import-untyped]
            ChaCha20Rng,
            RnsRandomStreams,
        )

        return ChaCha20Rng, RnsRandomStreams
    except ModuleNotFoundError as error:
        # Do not mask an import failure inside an installed triton-csprng
        # package by silently switching to a source submodule.
        if error.name != "triton_csprng":
            raise
        raise ModuleNotFoundError(
            "FHElium requires its declared `triton-csprng` package dependency. "
            "Reinstall FHElium dependencies in the target environment."
        ) from error


_UINT32_MASK = 0xFFFFFFFF


def _words_from_digest(data: bytes, *, n_words: int) -> list[int]:
    digest = hashlib.sha256(data).digest()
    while len(digest) < n_words * 4:
        digest += hashlib.sha256(digest).digest()
    return [
        int.from_bytes(digest[i : i + 4], "little") & _UINT32_MASK
        for i in range(0, n_words * 4, 4)
    ]


def _normalize_key(
    seed: Any,
) -> Sequence[int] | bytes | bytearray | torch.Tensor | None:
    if seed is None:
        return None
    if isinstance(seed, int):
        return _words_from_digest(
            seed.to_bytes(32, "little", signed=True), n_words=8
        )
    return seed


def _normalize_nonce(
    nonce: Any,
) -> Sequence[int] | bytes | bytearray | torch.Tensor | None:
    if nonce is None:
        return None
    if isinstance(nonce, int):
        low = nonce & _UINT32_MASK
        high = (nonce >> 32) & _UINT32_MASK
        return [low, high]
    return nonce


def _derive_nonce(base: Sequence[int], stream_id: int) -> list[int]:
    return [
        int(base[0]) & _UINT32_MASK,
        (int(base[1]) + stream_id) & _UINT32_MASK,
    ]


class Csprng:
    """FHElium adapter around the standalone :mod:`triton_csprng` package.

    The adapter supplies CKKS-specific channel and sampling configuration
    while delegating ChaCha20 stream generation to the standalone package.

    Args:
        num_coefs: Coefficients generated per channel.
        num_channels: Independent channel count per device, or one count to
            repeat for every device.
        num_repeating_channels: Channel count reproduced across devices.
        sigma: Standard deviation for discrete-Gaussian sampling.
        devices: CPU or CUDA devices that own generator streams. Defaults to
            every visible CUDA device.
        torch_dtype: Integral dtype returned to the engine.
        seed: Optional key material accepted by ``triton-csprng``. Fixed seed
            material is appropriate for controlled tests, not an application
            production-entropy policy.
        nonce: Optional base nonce accepted by ``triton-csprng``. The caller
            owns nonce uniqueness across independent generator instances.
    """

    def __init__(
        self,
        num_coefs: int = 2**15,
        num_channels: Sequence[int] = (8,),
        num_repeating_channels: int = 2,
        sigma: float = 3.19,
        devices: Sequence[torch.device | str] | None = None,
        torch_dtype: torch.dtype = torch.int64,
        seed: Any = None,
        nonce: Any = None,
    ) -> None:
        self.num_coefs = int(num_coefs)
        self.num_channels = list(num_channels)
        self.num_repeating_channels = int(num_repeating_channels)
        if torch_dtype not in {torch.int32, torch.int64}:
            raise ValueError("torch_dtype must be torch.int32 or torch.int64")
        self.torch_dtype = torch_dtype
        self._sigma = float(sigma)
        if not math.isfinite(self._sigma) or self._sigma <= 0:
            raise ValueError("sigma must be positive and finite")

        if devices is None:
            devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        normalized_devices = [torch.device(device) for device in devices]
        device_types = {device.type for device in normalized_devices}
        if len(device_types) != 1 or not device_types <= {"cpu", "cuda"}:
            raise ValueError("Csprng devices must be all CPU or all CUDA")
        self.devices = [str(device) for device in normalized_devices]
        self.num_devices = len(self.devices)
        if self.num_devices == 0:
            raise ValueError("Csprng requires at least one CPU or CUDA device")

        if len(self.num_channels) == 1:
            self.shares = [int(self.num_channels[0])] * self.num_devices
        elif len(self.num_channels) == self.num_devices:
            self.shares = [int(v) for v in self.num_channels]
        else:
            raise ValueError("Mismatch between num_channels and devices.")
        if self.num_coefs <= 0 or self.num_coefs % 4 != 0:
            raise ValueError("num_coefs must be a positive multiple of 4")
        if any(share < 0 for share in self.shares):
            raise ValueError("num_channels must be non-negative")
        if self.num_repeating_channels < 0:
            raise ValueError("num_repeating_channels must be non-negative")

        self.total_num_channels = sum(self.shares)
        self.L = self.num_coefs // 4
        ChaCha20Rng, RnsRandomStreams = _load_triton_csprng()
        self.streams = RnsRandomStreams(
            num_coeffs=self.num_coefs,
            channel_counts=self.shares,
            repeated_channels=self.num_repeating_channels,
            devices=self.devices,
            key=_normalize_key(seed),
            nonce=_normalize_nonce(nonce),
        )
        self._round_stream = ChaCha20Rng(
            key=self.streams.key_words,
            nonce=_derive_nonce(self.streams.nonce_words, 10_000),
            device=self.devices[0],
        )

    @property
    def sigma(self) -> float:
        """Discrete-Gaussian standard deviation fixed at construction."""

        return self._sigma

    def randbytes(
        self,
        shares: list[int] | None = None,
        repeats: int = 0,
        reshape: bool = False,
    ) -> list[torch.Tensor]:
        if shares is None:
            shares = self.shares
        if len(shares) != self.num_devices:
            raise ValueError("shares length must match devices")

        outputs = []
        for dev_id, share in enumerate(shares):
            parts = []
            share = int(share)
            if share > 0:
                parts.append(
                    self.streams._device_streams[dev_id].uint32(
                        (share, self.L, 16)
                    )
                )
            if repeats > 0:
                parts.append(
                    self.streams._repeat_streams[dev_id].uint32(
                        (int(repeats), self.L, 16)
                    )
                )
            if len(parts) == 1:
                out = parts[0]
            elif parts:
                out = torch.cat(parts, dim=0)
            else:
                out = torch.empty(
                    (0, self.L, 16),
                    dtype=torch.uint32,
                    device=self.devices[dev_id],
                )
            out = out.to(self.torch_dtype)
            if not reshape:
                out = out.reshape(out.shape[0], -1)
            outputs.append(out)
        return outputs

    def randint(
        self,
        amax: int | list[list[int]],
        shift: int = 0,
        repeats: int = 0,
    ) -> list[torch.Tensor]:
        if not isinstance(amax, (list, tuple)):
            channel_count = max(1, int(repeats))
            amax = [[int(amax)] * channel_count for _ in self.shares]
        result = self.streams.randint_channels(amax, repeated_channels=repeats)
        result = [sample.to(self.torch_dtype) for sample in result]
        if shift:
            result = [ri + int(shift) for ri in result]
        return result

    def discrete_gaussian(
        self, non_repeats: int | list[int] = 0, repeats: int = 1
    ) -> list[torch.Tensor]:
        if isinstance(non_repeats, int):
            shares = [int(non_repeats)] * self.num_devices
        else:
            shares = [int(v) for v in non_repeats]
        return [
            sample.to(self.torch_dtype)
            for sample in self.streams.discrete_gaussian_channels(
                shares, repeated_channels=repeats, sigma=self.sigma
            )
        ]

    def randround(self, coef: torch.Tensor) -> torch.Tensor:
        return self._round_stream.stochastic_round(coef).to(self.torch_dtype)
