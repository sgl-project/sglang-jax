"""Pinned-host KV pool for PD disaggregation.

The pool predefines ``pool_size`` independent host arrays (``memory_kind=
"pinned_host"`` on TPU; on CPU the kind falls back to the default since
``pinned_host`` is TPU-only) and hands them out / takes them back via a
FIFO queue. There is no LRU, no lock_ref, no retention — every borrow is
intended to live for one transfer.

The :class:`HostKVPool` ABC is the surface that HiCache's
``LRUHostKVPool`` will also implement in its own RFC; keeping the ABC
backend-agnostic now means HiCache can drop in without re-shaping the
contract.
"""

from __future__ import annotations

import abc
import logging
import threading
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

logger = logging.getLogger(__name__)


def _make_host_sharding(mesh: Mesh, partition_spec: PartitionSpec) -> NamedSharding:
    """Build a host-side sharding.

    Tries ``memory_kind="pinned_host"`` first (TPU-only); falls back to
    the default memory kind so unit tests can construct the pool on
    CPU. The fallback path is purely a CPU-test affordance — production
    PD always runs on TPU and uses pinned_host.
    """

    try:
        return NamedSharding(mesh, partition_spec, memory_kind="pinned_host")
    except (TypeError, ValueError):
        logger.warning(
            "pinned_host memory_kind unavailable on this platform; "
            "falling back to default. This is expected on CPU-only "
            "jaxlib and will hurt H2D throughput on TPU."
        )
        return NamedSharding(mesh, partition_spec)


@dataclass
class HostBufferHandle:
    """Handle for an in-use host buffer.

    ``buffer`` is the current ``jax.Array``; it may be replaced after
    each :meth:`HostKVPool.copy_from_device` because ``.at[].set()`` is
    a functional update. **Do not cache** ``handle.buffer`` across
    multiple ``copy_from_device`` calls on the same handle.
    """

    buffer_id: int
    num_tokens: int
    buffer: jax.Array


@dataclass
class StagedData:
    """Result of a successful D2H staging copy.

    ``array`` is the host-side jax.Array ready to hand to
    ``JaxTransferWrapper.register_pull``. ``buffer_id`` lets the
    transfer-completion callback return the buffer to the pool via
    :meth:`HostKVPool.put_buffer`.
    """

    buffer_id: int
    array: jax.Array


class HostKVPool(abc.ABC):
    """Backend-agnostic pinned-host KV pool contract.

    All sizing is in tokens. Concrete shapes (per-layer K/V split, head
    count, head dim) live on the implementing class. The ABC keeps the
    surface small so HiCache's LRU variant can implement the same
    contract without inheriting Queue semantics.
    """

    @abc.abstractmethod
    def alloc(self, num_tokens: int) -> HostBufferHandle | None:
        """Reserve a buffer big enough for ``num_tokens`` tokens.

        Returns ``None`` if the pool is empty or if ``num_tokens``
        exceeds the per-buffer capacity. The returned handle's
        ``buffer`` field is the un-modified pre-allocated array; use
        :meth:`copy_from_device` if you want it filled.
        """

    @abc.abstractmethod
    def free(self, handle: HostBufferHandle) -> None:
        """Return ``handle``'s buffer to the pool."""

    @abc.abstractmethod
    def get_buffer(self) -> tuple[int, HostBufferHandle]:
        """Low-level: pull one buffer regardless of token count."""

    @abc.abstractmethod
    def put_buffer(self, buffer_id: int) -> None:
        """Low-level: return a buffer by id."""

    @abc.abstractmethod
    def copy_from_device(self, device_kv: jax.Array) -> StagedData:
        """D2H staging primitive used by PD ``producer_handoff``.

        Allocates one buffer, copies ``device_kv`` (which lives on
        TPU/device memory) into the host-side buffer, and returns a
        :class:`StagedData` carrying the staged array + the buffer id
        for later release. ``device_kv.shape[0]`` is treated as the
        token count.
        """

    @abc.abstractmethod
    def available_size(self) -> int:
        """Buffers currently free."""

    @abc.abstractmethod
    def total_size(self) -> int:
        """Total buffers in the pool (free + in-use)."""


class QueueHostKVPool(HostKVPool):
    """FIFO-queue implementation of :class:`HostKVPool`.

    Short-lived borrows only — no LRU, no eviction, no lock_ref. Borrow
    via :meth:`alloc` or :meth:`copy_from_device`, return via
    :meth:`free` or :meth:`put_buffer`. Concurrent access is allowed
    (the queue is protected by a lock); the typical caller is the PD
    producer-handoff path which alternates main-thread alloc with
    background-thread free triggered by the ZMQ ack listener.
    """

    def __init__(
        self,
        pool_size: int,
        max_tokens_per_buffer: int,
        layer_num: int,
        kv_head_per_rank: int,
        head_dim: int,
        dtype: Any,
        mesh: Mesh,
        partition_spec: PartitionSpec,
        *,
        pool_name: str = "default",
    ) -> None:
        if pool_size <= 0:
            raise ValueError(f"pool_size must be positive, got {pool_size}")
        if max_tokens_per_buffer <= 0:
            raise ValueError(
                f"max_tokens_per_buffer must be positive, " f"got {max_tokens_per_buffer}"
            )
        self._pool_size = pool_size
        self._max_tokens_per_buffer = max_tokens_per_buffer
        self._layer_num = layer_num
        self._kv_head_per_rank = kv_head_per_rank
        self._head_dim = head_dim
        self._dtype = dtype
        self._mesh = mesh
        self._partition_spec = partition_spec
        self._pool_name = pool_name
        self._host_sharding = _make_host_sharding(mesh, partition_spec)

        self._buffer_shape = (
            max_tokens_per_buffer,
            layer_num,
            kv_head_per_rank,
            head_dim,
        )

        self._lock = threading.Lock()
        self._buffers: list[jax.Array] = self._allocate_buffers()
        self._free_ids: list[int] = list(range(pool_size))

    def _allocate_buffers(self) -> list[jax.Array]:
        buffers: list[jax.Array] = []
        zeros = jnp.zeros(self._buffer_shape, dtype=self._dtype)
        for _ in range(self._pool_size):
            buffers.append(jax.device_put(zeros, self._host_sharding))
        for b in buffers:
            b.block_until_ready()
        return buffers

    # ------------------------------------------------------------------
    # HostKVPool ABC
    # ------------------------------------------------------------------

    def alloc(self, num_tokens: int) -> HostBufferHandle | None:
        if num_tokens > self._max_tokens_per_buffer:
            return None
        if num_tokens <= 0:
            raise ValueError(f"num_tokens must be positive, got {num_tokens}")
        with self._lock:
            if not self._free_ids:
                return None
            buffer_id = self._free_ids.pop(0)
        try:
            from sgl_jax.srt.disaggregation.common.metrics import host_pool_alloc

            host_pool_alloc(self._pool_name, 1)
        except Exception:  # noqa: BLE001
            pass
        return HostBufferHandle(
            buffer_id=buffer_id,
            num_tokens=num_tokens,
            buffer=self._buffers[buffer_id],
        )

    def free(self, handle: HostBufferHandle) -> None:
        self._release(handle.buffer_id)

    def get_buffer(self) -> tuple[int, HostBufferHandle]:
        with self._lock:
            if not self._free_ids:
                raise RuntimeError(
                    "QueueHostKVPool is empty; caller should have " "checked available_size() first"
                )
            buffer_id = self._free_ids.pop(0)
        try:
            from sgl_jax.srt.disaggregation.common.metrics import host_pool_alloc

            host_pool_alloc(self._pool_name, 1)
        except Exception:  # noqa: BLE001
            pass
        return buffer_id, HostBufferHandle(
            buffer_id=buffer_id,
            num_tokens=self._max_tokens_per_buffer,
            buffer=self._buffers[buffer_id],
        )

    def put_buffer(self, buffer_id: int) -> None:
        self._release(buffer_id)

    def copy_from_device(self, device_kv: jax.Array) -> StagedData:
        num_tokens = device_kv.shape[0]
        if num_tokens > self._max_tokens_per_buffer:
            raise ValueError(
                f"device_kv has {num_tokens} tokens > pool's "
                f"max_tokens_per_buffer={self._max_tokens_per_buffer}"
            )
        handle = self.alloc(num_tokens)
        if handle is None:
            raise RuntimeError("QueueHostKVPool exhausted; alloc returned None")
        staged_device = jax.device_put(device_kv, self._host_sharding)
        updated = handle.buffer.at[:num_tokens].set(staged_device)
        updated.block_until_ready()
        # Replace the pool's slot so subsequent reads of self._buffers
        # see the latest content; .at[].set() returns a new array.
        self._buffers[handle.buffer_id] = updated
        try:
            from sgl_jax.srt.disaggregation.common.metrics import PD_TRANSFER_BYTES_TOTAL

            PD_TRANSFER_BYTES_TOTAL.labels(direction="d2h", role="prefill").inc(
                int(device_kv.nbytes)
            )
        except Exception:  # noqa: BLE001
            pass
        return StagedData(buffer_id=handle.buffer_id, array=updated)

    def available_size(self) -> int:
        with self._lock:
            return len(self._free_ids)

    def total_size(self) -> int:
        return self._pool_size

    # ------------------------------------------------------------------

    def _release(self, buffer_id: int) -> None:
        with self._lock:
            if buffer_id in self._free_ids:
                raise RuntimeError(f"double free of buffer_id={buffer_id}")
            if not (0 <= buffer_id < self._pool_size):
                raise ValueError(
                    f"buffer_id={buffer_id} outside pool range " f"[0, {self._pool_size})"
                )
            self._free_ids.append(buffer_id)
        try:
            from sgl_jax.srt.disaggregation.common.metrics import host_pool_free

            host_pool_free(self._pool_name, 1)
        except Exception:  # noqa: BLE001
            pass


def make_unit_mesh() -> Mesh:
    """Convenience for tests: a single-device mesh with axis ``x``."""

    devices = jax.local_devices()
    return Mesh(np.asarray(devices[:1]).reshape(1), axis_names=("x",))
