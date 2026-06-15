"""Host-staging KV pool for PD disaggregation.

The pool predefines ``pool_size`` independent host arrays (pinned host
memory, via an explicit ``memory_kind="pinned_host"`` — see
``_make_host_sharding``) and hands them out / takes them back via a
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
import time
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

logger = logging.getLogger(__name__)


def _make_host_sharding(mesh: Mesh, partition_spec: PartitionSpec) -> NamedSharding:
    """Build a pinned-host sharding.

    Requests host memory explicitly via ``memory_kind="pinned_host"``. A plain
    ``NamedSharding(mesh, spec)`` with no ``memory_kind`` resolves to ``device``
    (HBM) on TPU, which would keep the "staged" KV on the prefill HBM and defeat
    the whole point of staging — so the kind must be stated explicitly.

    ``pinned_host`` (not ``unpinned_host``): ``jax.experimental.transfer`` pulls
    a bf16 array registered from an unpinned-host source with a 4-byte stride,
    reading every other element (``got[i] == src[2i]``) and corrupting the KV.
    pinned_host transfers bit-exact; it is also what tpu-inference stages
    through (tpu_connector.py). Falls back to a plain (device) sharding if the
    platform (e.g. CPU jaxlib) does not support host memory kinds.
    """

    try:
        return NamedSharding(mesh, partition_spec, memory_kind="pinned_host")
    except (TypeError, ValueError):
        logger.warning(
            "pinned_host memory_kind unavailable on this platform; "
            "falling back to default sharding."
        )
        return NamedSharding(mesh, partition_spec)


@dataclass
class StagedData:
    """Result of a successful D2H staging copy.

    ``array_pytree`` is a per-layer list of host-side jax.Arrays, each
    sized to the request's padded page count, ready to hand to
    ``JaxTransferWrapper.register_pull``. ``buffer_id`` identifies the
    reserved pool slot; release is owned by the prefill terminal callback.
    """

    buffer_id: int
    array_pytree: list[jax.Array]


class HostKVPool(abc.ABC):
    """Backend-agnostic host-staging KV pool contract.

    Sizing is per-request entry: each entry is a list of ``layer_num``
    host arrays, each large enough to hold one request's padded pages.
    Concrete shapes (per-layer K/V split, head count, head dim) live on
    the implementing class. The ABC keeps the surface small so HiCache's
    LRU variant can implement the same contract without inheriting Queue
    semantics.
    """

    @abc.abstractmethod
    def reserve(self) -> int | None:
        """Pop a free slot id, or ``None`` if the pool is exhausted.

        Admission reserves a slot up front; the reserved slot is later
        filled via :meth:`copy_from_device` and returned via
        :meth:`release`.
        """

    @abc.abstractmethod
    def release(self, buffer_id: int) -> None:
        """Return a reserved slot to the pool by id."""

    @abc.abstractmethod
    def copy_from_device(self, layers: list[jax.Array], buffer_id: int) -> StagedData:
        """D2H staging primitive used by PD ``producer_handoff``.

        Writes each per-layer device array in ``layers`` into the
        PRE-RESERVED slot ``buffer_id`` and returns a :class:`StagedData`
        carrying a per-layer list of right-sized host slices. Release is
        owned by the caller (prefill terminal callback), not by this
        method.
        """

    @abc.abstractmethod
    def available_size(self) -> int:
        """Entries currently free."""

    @abc.abstractmethod
    def total_size(self) -> int:
        """Total entries in the pool (free + in-use)."""


class QueueHostKVPool(HostKVPool):
    """FIFO-queue implementation of :class:`HostKVPool`.

    Short-lived borrows only — no LRU, no eviction, no lock_ref. Borrow
    via :meth:`reserve`, fill via :meth:`copy_from_device`, return via
    :meth:`release`. ``self._lock`` protects only the
    ``_free_ids`` free-list (reserve/release); the per-slot buffer writes
    in :meth:`copy_from_device` are NOT lock-protected and rely instead on
    the caller holding an exclusive reservation of that ``buffer_id``. The
    typical caller is the PD producer-handoff path which alternates
    main-thread reserve with background-thread release triggered by the
    ZMQ ack listener.
    """

    def __init__(
        self,
        pool_size: int,
        max_padded_pages: int,
        layer_num: int,
        per_layer_shape: tuple[int, ...],
        dtype: Any,
        mesh: Mesh,
        partition_spec: PartitionSpec,
        *,
        pool_name: str = "default",
    ) -> None:
        if pool_size <= 0:
            raise ValueError(f"pool_size must be positive, got {pool_size}")
        if max_padded_pages <= 0:
            raise ValueError(f"max_padded_pages must be positive, got {max_padded_pages}")
        self._pool_size = pool_size
        self._max_padded_pages = max_padded_pages
        self._layer_num = layer_num
        self._per_layer_shape = tuple(per_layer_shape)
        self._dtype = dtype
        self._mesh = mesh
        self._partition_spec = partition_spec
        self._pool_name = pool_name
        self._host_sharding = _make_host_sharding(mesh, partition_spec)
        self._layer_shape = (max_padded_pages, *self._per_layer_shape)

        self._lock = threading.Lock()
        # Each entry is a list of ``layer_num`` host arrays.
        self._buffers: list[list[jax.Array]] = self._allocate_buffers()
        self._free_ids: list[int] = list(range(pool_size))
        # Backpressure observability: peak concurrent occupancy and the number
        # of reserve() calls that hit an empty pool (each is an admission
        # deferral upstream). The exhaustion log is throttled to ~1/s so a
        # stress run does not flood the server log.
        self._peak_used = 0
        self._exhaust_count = 0
        self._last_exhaust_log = 0.0

    def _allocate_buffers(self) -> list[list[jax.Array]]:
        zeros = jnp.zeros(self._layer_shape, dtype=self._dtype)
        host_zero = jax.device_put(zeros, self._host_sharding)
        host_zero.block_until_ready()
        entries: list[list[jax.Array]] = []
        for _ in range(self._pool_size):
            entries.append([host_zero for _ in range(self._layer_num)])
        return entries

    # ------------------------------------------------------------------
    # HostKVPool ABC
    # ------------------------------------------------------------------

    def reserve(self) -> int | None:
        with self._lock:
            if not self._free_ids:
                self._exhaust_count += 1
                now = time.time()
                if now - self._last_exhaust_log >= 1.0:
                    self._last_exhaust_log = now
                    logger.info(
                        "host pool %s exhausted (size=%d, peak_used=%d, "
                        "exhaust_count=%d); deferring PD admission (backpressure)",
                        self._pool_name,
                        self._pool_size,
                        self._peak_used,
                        self._exhaust_count,
                    )
                return None
            buffer_id = self._free_ids.pop(0)
            used = self._pool_size - len(self._free_ids)
            if used > self._peak_used:
                self._peak_used = used
        self._inc_alloc_metric()
        return buffer_id

    def release(self, buffer_id: int) -> None:
        self._release(buffer_id)

    def copy_from_device(self, layers: list[jax.Array], buffer_id: int) -> StagedData:
        # Caller must hold exclusive ownership of ``buffer_id`` (obtained via
        # reserve() and not yet release()d). The per-slot buffer mutation below
        # is intentionally done outside ``self._lock`` and is safe only under
        # this single-owner discipline.
        if not (0 <= buffer_id < self._pool_size):
            raise ValueError(f"buffer_id={buffer_id} outside pool range [0, {self._pool_size})")
        if len(layers) != self._layer_num:
            raise ValueError(f"expected {self._layer_num} layers, got {len(layers)}")
        padded_pages = layers[0].shape[0]
        if padded_pages > self._max_padded_pages:
            raise ValueError(
                f"layer has {padded_pages} pages > max_padded_pages={self._max_padded_pages}"
            )
        array_pytree: list[jax.Array] = []
        total_bytes = 0
        # Right-sized staging: device_put each per-layer array straight to the
        # host sharding and hand it back directly. Scattering into a fixed
        # max-page pool buffer (``entry[i].at[:n].set(...)``) would copy the
        # whole buffer every call regardless of payload size; donation is not
        # viable because the pre-allocated slots share one buffer object. The
        # pool's reserve/release still bounds concurrency by slot COUNT;
        # ``register_pull`` keeps each staged array alive until release. Issue
        # all per-layer copies before blocking so they pipeline.
        for layer in layers:
            host_layer = jax.device_put(layer, self._host_sharding)
            array_pytree.append(host_layer)
            total_bytes += int(layer.nbytes)
        jax.block_until_ready(array_pytree)
        self._record_d2h_bytes(total_bytes)
        return StagedData(buffer_id=buffer_id, array_pytree=array_pytree)

    def available_size(self) -> int:
        with self._lock:
            return len(self._free_ids)

    def total_size(self) -> int:
        return self._pool_size

    # ------------------------------------------------------------------

    def _inc_alloc_metric(self) -> None:
        try:
            from sgl_jax.srt.disaggregation.common.metrics import host_pool_alloc

            host_pool_alloc(self._pool_name, 1)
        except Exception:  # noqa: BLE001
            pass

    def _record_d2h_bytes(self, nbytes: int) -> None:
        try:
            from sgl_jax.srt.disaggregation.common.metrics import (
                PD_TRANSFER_BYTES_TOTAL,
            )

            PD_TRANSFER_BYTES_TOTAL.labels(direction="d2h", role="prefill").inc(nbytes)
        except Exception:  # noqa: BLE001
            pass

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
