"""Page-addressed local transfers, independent of a native transport or store."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class DevicePageSpan:
    """Pages in one component pool's rank-local address space."""

    dp_rank: int
    pages: tuple[int, ...]


@dataclass(frozen=True)
class FullKVTransfer:
    device: DevicePageSpan
    device_tokens: tuple[int, ...]
    host_handles: tuple[int, ...] = ()


class CopyFuture(Protocol):
    def IsReady(self) -> bool: ...

    def Await(self) -> None: ...


class TransferOperation:
    """Native completion with scheduler-thread callbacks and no implicit worker.

    A failed native call does not prove that every shard has stopped accessing
    memory. The backend's failure callback must quarantine the resources.
    """

    def __init__(
        self,
        future: CopyFuture | None,
        on_success: Callable[[], None] = lambda: None,
        on_failure: Callable[[Exception], None] = lambda _: None,
    ):
        self._future = future
        self._on_success = on_success
        self._on_failure = on_failure
        self._finished = False
        self._error: Exception | None = None

    def done(self) -> bool:
        if not self._finished:
            try:
                if self._future is not None and not self._future.IsReady():
                    return False
            except Exception as exc:
                self._fail(exc)
                return True
            self._finish()
        return True

    def _fail(self, exc: Exception) -> None:
        self._error = exc
        self._finished = True
        self._on_failure(exc)

    def _finish(self) -> None:
        if self._finished:
            return
        try:
            if self._future is not None:
                self._future.Await()
            self._on_success()
        except Exception as exc:
            self._fail(exc)
        else:
            self._finished = True

    def wait(self) -> None:
        self._finish()
        if self._error is not None:
            raise self._error
