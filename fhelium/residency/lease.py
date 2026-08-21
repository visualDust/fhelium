"""Borrowed value, hold, and accounting-reservation lifetimes."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from threading import RLock
from types import TracebackType
from typing import TYPE_CHECKING, Any, TypeVar, cast
from warnings import warn
from weakref import finalize

import torch

from fhelium.core import TensorResident
from fhelium.errors import ResidencyLifetimeClosedError
from fhelium.residency.location import ResidencyLocation
from fhelium.residency.model import ResidencyHandle

if TYPE_CHECKING:
    from fhelium.residency.manager import ResidencyManager


ValueT = TypeVar("ValueT", bound=TensorResident)
_ORPHANED_MANAGERS: list[Any] = []


def _abandon_lifetime(
    manager: Any,
    token: object,
    kind: str,
    diagnostic: str,
    streams: list[torch.cuda.Stream] | None = None,
) -> None:
    retained = False
    try:
        if kind == "lease":
            manager._release_lease(
                token,
                consumer_streams=tuple(streams or ()),
                wait=True,
            )
        elif kind == "hold":
            manager._release_hold(token, missing_ok=True)
        else:
            manager._release_reservation(token, missing_ok=True)
    except BaseException:
        # Retain a real strong root if synchronization/release is impossible.
        # This is preferable to returning storage to the CUDA allocator while
        # an unobserved kernel may still reference it.
        _ORPHANED_MANAGERS.append(manager)
        retained = True
    warn(
        f"An active {kind} for {diagnostic} was abandoned. "
        + (
            "Its manager remains conservatively retained because safe "
            "release was not possible. Close every lifetime."
            if retained
            else "Its protection was released safely. Close it."
        ),
        ResourceWarning,
        stacklevel=2,
    )


class BorrowedValues(Mapping[ResidencyHandle[Any], TensorResident]):
    """Concrete immutable materializations borrowed through one active lease.

    The mapping validates its lease on every lookup and iteration. Returned
    values remain ordinary Python objects; callers must not mutate them or
    retain aliases beyond the lease. Managed memory accounting is strong only
    for manager-owned aliases that obey these borrowing rules.
    """

    def __init__(self, lease: ResidencyLease) -> None:
        self._lease = lease

    def __getitem__(
        self,
        handle: ResidencyHandle[ValueT],
    ) -> ValueT:
        return self._lease._value(handle)

    def __iter__(self) -> Iterator[ResidencyHandle[Any]]:
        for handle in self._lease.handles:
            self._lease._require_active()
            yield handle

    def __len__(self) -> int:
        self._lease._require_active()
        return len(self._lease.handles)


class ResidencyLease:
    """Short read lifetime protected through CUDA consumer completion.

    A lease is created only by :meth:`ResidencyManager.acquire`. CUDA consumer
    streams are registered rather than synchronized at Python scope exit. On
    release, the manager records one completion event per registered stream and
    retains every materialization until those events complete. CUDA acquisition
    requires one consumer stream; CPU leases release immediately.

    Consumers using additional CUDA streams must call
    :meth:`add_consumer_stream` before release. Abandoning a lease synchronizes
    its registered streams before release; if that fails, a process-global
    strong root conservatively retains the manager and its storage.
    """

    def __init__(
        self,
        *,
        manager: ResidencyManager,
        token: object,
        handles: tuple[ResidencyHandle[Any], ...],
        location: ResidencyLocation,
        consumer_streams: Sequence[torch.cuda.Stream],
    ) -> None:
        self._manager = manager
        self._token = token
        self._handles = handles
        self._location = location
        self._streams = list(consumer_streams)
        self._active = True
        self._entered = False
        self._state_lock = RLock()
        self._finalizer = finalize(
            self,
            _abandon_lifetime,
            manager,
            token,
            "lease",
            f"manager {manager.manager_id!r} at {location}",
            self._streams,
        )

    @property
    def active(self) -> bool:
        """Whether the Python lease scope remains open."""

        with self._state_lock:
            return self._active

    @property
    def handles(self) -> tuple[ResidencyHandle[Any], ...]:
        """Managed values protected by this lease."""

        return self._handles

    @property
    def location(self) -> ResidencyLocation:
        """Location of every borrowed materialization."""

        return self._location

    @property
    def values(self) -> BorrowedValues:
        """Borrowed mapping valid until this lease is released."""

        self._require_active()
        return BorrowedValues(self)

    def add_consumer_stream(self, stream: torch.cuda.Stream) -> None:
        """Register another CUDA stream whose prior reads this lease protects."""

        with self._state_lock:
            self._require_active_locked()
            if self._location.kind != "cuda":
                raise ValueError("CPU residency leases do not use CUDA streams")
            if not isinstance(stream, torch.cuda.Stream):
                raise TypeError("Consumer stream must be a torch.cuda.Stream")
            if torch.device(stream.device) != self._location.device:
                raise ValueError(
                    "Consumer stream is on the wrong CUDA device: "
                    f"expected={self._location.device}, actual={stream.device}"
                )
            if all(stream is not existing for existing in self._streams):
                self._streams.append(stream)

    def release(self, *, wait: bool = False) -> None:
        """Close the borrow and protect CUDA reads until completion.

        Args:
            wait: Synchronize recorded completion events before returning.
                With ``False``, the manager retains a pending protection and
                reaps it after the events report completion.
        """

        with self._state_lock:
            if not self._active:
                return
            self._manager._release_lease(
                self._token,
                consumer_streams=tuple(self._streams),
                wait=wait,
            )
            self._active = False
            self._finalizer.detach()

    close = release

    def __enter__(self) -> BorrowedValues:
        with self._state_lock:
            self._require_active_locked()
            if self._entered:
                raise RuntimeError("ResidencyLease cannot be entered twice")
            self._entered = True
        return BorrowedValues(self)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()

    def _require_active(self) -> None:
        with self._state_lock:
            self._require_active_locked()

    def _require_active_locked(self) -> None:
        if not self._active:
            raise ResidencyLifetimeClosedError(
                "Borrowed residency values cannot be accessed after lease release"
            )

    def _value(self, handle: ResidencyHandle[ValueT]) -> ValueT:
        with self._state_lock:
            self._require_active_locked()
            if handle not in self._handles:
                raise KeyError(handle)
            return cast(
                ValueT, self._manager._leased_value(self._token, handle)
            )


class ResidencyHold:
    """Long-lived retention protection that exposes no concrete value."""

    def __init__(
        self,
        *,
        manager: ResidencyManager,
        token: object,
        handles: tuple[ResidencyHandle[Any], ...],
        location: ResidencyLocation,
    ) -> None:
        self._manager = manager
        self._token = token
        self._handles = handles
        self._location = location
        self._active = True
        self._entered = False
        self._state_lock = RLock()
        self._finalizer = finalize(
            self,
            _abandon_lifetime,
            manager,
            token,
            "hold",
            f"manager {manager.manager_id!r} at {location}",
        )

    @property
    def active(self) -> bool:
        """Whether this hold still protects its materializations."""

        with self._state_lock:
            return self._active

    @property
    def handles(self) -> tuple[ResidencyHandle[Any], ...]:
        """Managed values retained by this hold."""

        return self._handles

    @property
    def location(self) -> ResidencyLocation:
        """Location at which the materializations are retained."""

        return self._location

    def release(self) -> None:
        """Release exactly this retention protection idempotently."""

        with self._state_lock:
            if not self._active:
                return
            self._manager._release_hold(self._token)
            self._active = False
            self._finalizer.detach()

    close = release

    def __enter__(self) -> ResidencyHold:
        with self._state_lock:
            if not self._active:
                raise ResidencyLifetimeClosedError(
                    "ResidencyHold cannot be entered after release"
                )
            if self._entered:
                raise RuntimeError("ResidencyHold cannot be entered twice")
            self._entered = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()


class ResidencyReservation:
    """Idempotently releasable accounting reservation at one location."""

    def __init__(
        self,
        *,
        manager: ResidencyManager,
        token: object,
        location: ResidencyLocation,
        nbytes: int,
        label: str,
    ) -> None:
        self._manager = manager
        self._token = token
        self.location = location
        self.nbytes = nbytes
        self.label = label
        self._active = True
        self._entered = False
        self._state_lock = RLock()
        self._finalizer = finalize(
            self,
            _abandon_lifetime,
            manager,
            token,
            "reservation",
            f"manager {manager.manager_id!r} at {location}",
        )

    @property
    def active(self) -> bool:
        """Whether this reservation still charges its location budget."""

        with self._state_lock:
            return self._active

    def release(self) -> None:
        """Return the reserved headroom to its location budget."""

        with self._state_lock:
            if not self._active:
                return
            self._manager._release_reservation(self._token)
            self._active = False
            self._finalizer.detach()

    close = release

    def __enter__(self) -> ResidencyReservation:
        with self._state_lock:
            if not self._active:
                raise ResidencyLifetimeClosedError(
                    "ResidencyReservation cannot be entered after release"
                )
            if self._entered:
                raise RuntimeError(
                    "ResidencyReservation cannot be entered twice"
                )
            self._entered = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()


__all__ = [
    "BorrowedValues",
    "ResidencyHold",
    "ResidencyLease",
    "ResidencyReservation",
]
