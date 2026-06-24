"""Host-staging KV pools (pinned host memory).

    HostKVPool (ABC)              backend-agnostic contract; page-id boundary
    │                            (every int crossing in/out is a page id,
    │                             raiden-block-switchable; no jax.Array crosses)
    ├── QueueHostKVPool          PD disaggregation: bounded FIFO, one-shot slots
    │                            (reserve -> copy_from_device -> release; no LRU)
    └── LRUHostKVPool            HiCache L2: slots RETAIN data until released;
                                 page-addressed, lock_ref-protected

LRUHostKVPool transfers are two-phase, split by which thread may touch the live
KV buffer (the forward donates it every step):

    D2H backup:    stage_backup (sync gather, KV-owning thread)
                   -> flush_backup (async host put, D2H worker)
    H2D load-back: stage_load   (async device put, H2D worker)
                   -> flush_load (sync scatter, KV-owning thread, donation-safe)
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
    """Pinned-host sharding (falls back to default on platforms without it).

    ``memory_kind="pinned_host"`` is required: a plain ``NamedSharding`` resolves
    to ``device`` (HBM) on TPU, which would keep the "staged" KV in HBM. Must be
    ``pinned_host``, not ``unpinned_host`` — an unpinned bf16 source transfers
    with a 4-byte stride (``got[i] == src[2i]``), silently corrupting the KV.
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
    """Result of a D2H staging copy: a per-layer list of host arrays
    (``array_pytree``) sized to the request's padded pages, plus the reserved
    ``buffer_id``. Release is owned by the caller, not by the staging copy."""

    buffer_id: int
    array_pytree: list[jax.Array]


class HostKVPool(abc.ABC):
    """Backend-agnostic host-staging KV pool contract.

    Each entry is a list of ``layer_num`` host arrays, each holding one
    request's padded pages. Concrete shapes live on the implementing class.
    """

    @abc.abstractmethod
    def reserve(self) -> int | None:
        """Pop a free slot id, or ``None`` if exhausted."""

    @abc.abstractmethod
    def release(self, buffer_id: int) -> None:
        """Return a reserved slot to the pool by id."""

    @abc.abstractmethod
    def copy_from_device(self, layers: list[jax.Array], buffer_id: int) -> StagedData:
        """D2H staging for PD ``producer_handoff``: write ``layers`` into the
        pre-reserved ``buffer_id`` and return a :class:`StagedData`. Release is
        owned by the caller."""

    @abc.abstractmethod
    def available_size(self) -> int:
        """Entries currently free."""

    @abc.abstractmethod
    def total_size(self) -> int:
        """Total entries in the pool (free + in-use)."""

    # Retaining transfer primitives (HiCache). Defaulted (not abstract) so
    # QueueHostKVPool stays concrete and the PD path is untouched — only
    # LRUHostKVPool overrides them. Pure integer index pairs, isomorphic to a
    # future raiden ``d2h(src_offsets, dst_offsets)`` / ``h2d(...)``.

    def copy_into(self, device_indices: list[int], host_buffer_ids: list[int]) -> None:
        """Retaining D2H: copy device page(s) into reserved slot(s), pairwise.
        Data STAYS in the slot until :meth:`release`; no ``jax.Array`` crosses
        the boundary."""
        raise NotImplementedError

    def copy_to_device(self, host_buffer_ids: list[int], device_indices: list[int]) -> None:
        """H2D: scatter slot(s) back into the device pool by index, pairwise."""
        raise NotImplementedError


class QueueHostKVPool(HostKVPool):
    """FIFO one-shot pool for PD: reserve -> copy_from_device -> release.

    No LRU, no eviction, no lock_ref. ``self._lock`` guards only the
    ``_free_ids`` free-list; per-slot buffer writes in :meth:`copy_from_device`
    are unlocked and rely on the caller holding an exclusive reservation of that
    ``buffer_id`` (the PD producer-handoff path: main-thread reserve, background
    ZMQ-ack release).
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

        self._lock = threading.Lock()
        self._free_ids: list[int] = list(range(pool_size))
        # Backpressure observability: peak concurrent occupancy and the number
        # of reserve() calls that hit an empty pool (each is an admission
        # deferral upstream). The exhaustion log is throttled to ~1/s so a
        # stress run does not flood the server log.
        self._peak_used = 0
        self._exhaust_count = 0
        self._last_exhaust_log = 0.0

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
        expected_shape = (padded_pages, *self._per_layer_shape)
        expected_dtype = np.dtype(self._dtype)
        for i, layer in enumerate(layers):
            # Validate every layer, not just layers[0]: a ragged list (e.g. a
            # later layer shaped differently) would otherwise stage mismatched
            # arrays and silently corrupt the pulled KV on the decode side.
            if layer.shape != expected_shape:
                raise ValueError(f"layer {i} shape {layer.shape} != expected {expected_shape}")
            if np.dtype(layer.dtype) != expected_dtype:
                raise ValueError(f"layer {i} dtype {layer.dtype} != expected {expected_dtype}")
        array_pytree: list[jax.Array] = []
        total_bytes = 0
        # device_put each per-layer array straight to the right-sized host
        # sharding. Issue all per-layer copies before blocking so they pipeline.
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


class LRUHostKVPool(HostKVPool):
    """Retaining host pool for HiCache L2.

    Each slot holds one KV page packed over all layers as a single pinned-host
    ``jax.Array`` ``(layer_num, *per_layer_shape)``; retention = keeping the
    reference, avoiding ``dynamic_update_slice`` on sharded host (multi-chip
    crash, the MegaScale issue from the HiCache+PD spike). Page-addressed: every
    ``int`` crossing the boundary is a page id, and ``available_size`` /
    ``total_size`` count pages.

    ``self._lock`` guards only the free-list and ``_lock_ref``; per-slot
    reads/writes rely on the caller holding an exclusive reservation.
    """

    def __init__(
        self,
        device_pool: Any,
        pool_size: int,
        page_size: int,
        layer_num: int,
        per_layer_shape: tuple[int, ...],
        dtype: Any,
        mesh: Mesh,
        partition_spec: PartitionSpec,
        *,
        pool_name: str = "hicache",
    ) -> None:
        if pool_size <= 0:
            raise ValueError(f"pool_size must be positive, got {pool_size}")
        self._device_pool = device_pool
        self._pool_size = pool_size
        self._page_size = page_size
        self._layer_num = layer_num
        self._per_layer_shape = tuple(per_layer_shape)
        self._dtype = dtype
        self._mesh = mesh
        self._partition_spec = partition_spec
        self._pool_name = pool_name

        # A slot is the page packed over layers: prepend a replicated layer
        # axis and drop the device pool's leading page/DP axis (a single
        # gathered page is not DP-sharded). The TP axis on heads is preserved.
        self._slot_spec = PartitionSpec(None, *tuple(partition_spec)[1:])
        self._host_sharding = _make_host_sharding(mesh, self._slot_spec)
        self._device_packed_sharding = NamedSharding(mesh, self._slot_spec)
        # Batched gather output (one layer, MANY pages): page axis is back and
        # replicated (gathered pages are not DP-distributed); same form PD uses.
        self._batched_layer_sharding = NamedSharding(
            mesh, PartitionSpec(None, *tuple(partition_spec)[1:])
        )
        # PD's bucketed batched-gather helpers, reused to bound compiled shapes
        # (one kernel per page bucket). Imported at runtime to avoid a
        # mem_cache <- disaggregation import cycle.
        from sgl_jax.srt.disaggregation.prefill import (
            _jit_gather_one_layer,
            _pad_to_page_bucket,
        )

        self._jit_gather_one_layer = _jit_gather_one_layer
        self._pad_to_page_bucket = _pad_to_page_bucket

        self._lock = threading.Lock()
        self._free_ids: list[int] = list(range(pool_size))
        self._slots: list[jax.Array | None] = [None] * pool_size
        self._lock_ref: list[int] = [0] * pool_size
        # Allocated-state mirror of the free-list (O(1) membership for the
        # page-id boundary checks below; a bare ``in self._free_ids`` is O(n)).
        self._allocated: list[bool] = [False] * pool_size
        self._peak_used = 0
        self._exhaust_count = 0
        self._last_exhaust_log = 0.0
        # D2H: pages gathered by stage_backup (buffer-owning thread, sync) await
        # their async host transfer in flush_backup (D2H worker), keyed by buffer_id.
        self._pending_lock = threading.Lock()
        self._pending_gather: dict[int, jax.Array] = {}
        # H2D (mirror): pages staged by stage_load (worker, slow device_put, never
        # touches kv_buffer) await their cheap in-place scatter in flush_load
        # (buffer-owning thread, donation-safe window), keyed by buffer_id.
        self._pending_load_lock = threading.Lock()
        self._pending_load: dict[int, jax.Array] = {}

    # ------------------------------------------------------------------
    # HostKVPool ABC
    # ------------------------------------------------------------------

    def reserve(self) -> int | None:
        # Free-list pop, no auto-evict: returns None when full. The pool can't
        # evict (only the tree cache knows which buffer_id a node points at), so
        # the tree cache must release an LRU slot before retrying.
        with self._lock:
            if not self._free_ids:
                self._exhaust_count += 1
                now = time.time()
                if now - self._last_exhaust_log >= 1.0:
                    self._last_exhaust_log = now
                    logger.info(
                        "host pool %s exhausted (size=%d, peak_used=%d, "
                        "exhaust_count=%d); tree cache must evict before retry",
                        self._pool_name,
                        self._pool_size,
                        self._peak_used,
                        self._exhaust_count,
                    )
                return None
            buffer_id = self._free_ids.pop(0)
            self._allocated[buffer_id] = True
            used = self._pool_size - len(self._free_ids)
            if used > self._peak_used:
                self._peak_used = used
            return buffer_id

    def release(self, buffer_id: int) -> None:
        with self._lock:
            if not (0 <= buffer_id < self._pool_size):
                raise ValueError(f"buffer_id={buffer_id} outside pool range [0, {self._pool_size})")
            if buffer_id in self._free_ids:
                raise RuntimeError(f"double free of buffer_id={buffer_id}")
            if self._lock_ref[buffer_id] != 0:
                raise RuntimeError(
                    f"release of locked buffer_id={buffer_id} "
                    f"(lock_ref={self._lock_ref[buffer_id]})"
                )
            self._drop_pending(buffer_id)
            self._slots[buffer_id] = None
            self._allocated[buffer_id] = False
            self._free_ids.append(buffer_id)

    # Page-batch alloc/free: the batched, page-addressed counterparts of
    # reserve/release. The control plane speaks PAGE ids; each slot holds exactly
    # one device page. These map 1:1 onto a future raiden d2h/h2d block interface.

    def alloc(self, need_pages: int) -> np.ndarray | None:
        """Pop ``need_pages`` free page slots, returning their ids.

        All-or-nothing: returns ``None`` if the request can't be satisfied in
        full (no partial alloc, no auto-evict — the tree cache evicts an LRU
        node and retries).
        """
        if need_pages <= 0:
            raise ValueError(f"need_pages must be positive, got {need_pages}")
        with self._lock:
            if len(self._free_ids) < need_pages:
                self._exhaust_count += 1
                now = time.time()
                if now - self._last_exhaust_log >= 1.0:
                    self._last_exhaust_log = now
                    logger.info(
                        "host pool %s short on alloc (need=%d, free=%d, size=%d, "
                        "exhaust_count=%d); tree cache must evict before retry",
                        self._pool_name,
                        need_pages,
                        len(self._free_ids),
                        self._pool_size,
                        self._exhaust_count,
                    )
                return None
            pages = [self._free_ids.pop(0) for _ in range(need_pages)]
            for pid in pages:
                self._allocated[pid] = True
            used = self._pool_size - len(self._free_ids)
            if used > self._peak_used:
                self._peak_used = used
        return np.asarray(pages, dtype=np.int64)

    def free(self, host_page_ids) -> None:
        """Release page slot(s) by id, deduplicated. Rejects locked
        (``lock_ref != 0``) or already-free pages."""
        seen: set[int] = set()
        unique: list[int] = []
        for pid in host_page_ids:
            pid = int(pid)
            if pid not in seen:
                seen.add(pid)
                unique.append(pid)
        with self._lock:
            for pid in unique:
                if not (0 <= pid < self._pool_size):
                    raise ValueError(f"page id={pid} outside pool range [0, {self._pool_size})")
                if pid in self._free_ids:
                    raise RuntimeError(f"double free of page id={pid}")
                if self._lock_ref[pid] != 0:
                    raise RuntimeError(
                        f"free of locked page id={pid} (lock_ref={self._lock_ref[pid]})"
                    )
                self._drop_pending(pid)
                self._slots[pid] = None
                self._allocated[pid] = False
                self._free_ids.append(pid)

    def stage_backup(self, device_indices: list[int], host_buffer_ids: list[int]) -> None:
        """D2H phase 1: gather live device pages into per-buffer staging arrays.

        Sync + on the KV-owning thread: the forward donates ``kv_buffer`` every
        step, so an off-thread gather would race that deletion. ``block_until_ready``
        materializes the pages before return; the slow host put is deferred to
        :meth:`flush_backup`.
        """
        if len(device_indices) != len(host_buffer_ids):
            raise ValueError(
                f"device_indices ({len(device_indices)}) and host_buffer_ids "
                f"({len(host_buffer_ids)}) length mismatch"
            )
        if not host_buffer_ids:
            return
        for buffer_id in host_buffer_ids:
            self._require_allocated(buffer_id)
        self._require_device_pages(device_indices)
        buffers = self._device_pool.kv_buffer
        n = len(device_indices)
        # Bucket the page count so each layer's gather compiles once per bucket
        # (PD pattern), not once per distinct n. Pad with page 0; the padding
        # rows are gathered then discarded by the per-buffer slice below.
        n_b = self._pad_to_page_bucket(n)
        dev_idx_np = np.asarray(device_indices, dtype=np.int32)
        if n_b > n:
            dev_idx_np = np.concatenate([dev_idx_np, np.zeros(n_b - n, dtype=dev_idx_np.dtype)])
        page_indices = jax.device_put(dev_idx_np, NamedSharding(self._mesh, PartitionSpec(None)))
        # One batched gather per layer (page axis back); keep the per-layer loop
        # so peak HBM stays at one layer's bucket (PD caps it this way too).
        per_layer = [
            self._jit_gather_one_layer(buffers[layer], page_indices, self._batched_layer_sharding)
            for layer in range(self._layer_num)
        ]
        packed = jnp.stack(per_layer, axis=0)  # (layer_num, n_b, *per_layer_shape)
        jax.block_until_ready(packed)
        # Slice the real pages back per buffer_id (cheap views on the now-
        # materialized array); padding rows [n:] are dropped.
        gathered = {bid: packed[:, i] for i, bid in enumerate(host_buffer_ids)}
        with self._pending_lock:
            self._pending_gather.update(gathered)

    def flush_backup(self, host_buffer_ids: list[int]) -> None:
        """D2H phase 2: ``device_put`` the staged pages into their host slots.

        Touches only ``_pending_gather`` / ``_slots`` (never the KV buffer), so
        it runs on the D2H worker concurrently with forward.
        """
        if not host_buffer_ids:
            return
        # Per-slot host transfer: on the worker thread (off the hot path), never
        # the profiled bottleneck. Batching would need to slice a pinned-host
        # array, which has no CPU implementation.
        staged: list[jax.Array] = []
        for buffer_id in host_buffer_ids:
            self._require_allocated(buffer_id)
            with self._pending_lock:
                packed = self._pending_gather.pop(buffer_id, None)
            if packed is None:
                raise RuntimeError(
                    f"flush_backup with no staged gather for buffer_id={buffer_id}; "
                    f"stage_backup() must run (synchronously) first"
                )
            host_packed = jax.device_put(packed, self._host_sharding)
            self._slots[buffer_id] = host_packed
            staged.append(host_packed)
        jax.block_until_ready(staged)

    def copy_into(self, device_indices: list[int], host_buffer_ids: list[int]) -> None:
        # Sync convenience (gather + host transfer on one thread): tests and
        # single-threaded callers. The async HiCache path calls stage_backup +
        # flush_backup separately so the device read can't race KV-buffer donation.
        self.stage_backup(device_indices, host_buffer_ids)
        self.flush_backup(host_buffer_ids)

    def copy_to_device(self, host_buffer_ids: list[int], device_indices: list[int]) -> None:
        # Sync convenience (stage + scatter on one thread): tests and
        # single-threaded callers. The async HiCache path calls stage_load +
        # flush_load separately so the slow transfer overlaps forward and only
        # the cheap scatter serializes with KV-buffer donation.
        host_buffer_ids = list(host_buffer_ids)
        self.stage_load(host_buffer_ids)
        self.flush_load(host_buffer_ids, device_indices)

    def stage_load(self, host_buffer_ids: list[int]) -> None:
        """H2D phase 1: ``device_put`` host slots onto the device into per-buffer
        staging arrays.

        Touches only ``_slots`` / ``_pending_load`` (never the KV buffer), so it
        runs on the H2D worker concurrently with forward; the cheap scatter is
        deferred to :meth:`flush_load`.
        """
        if not host_buffer_ids:
            return
        # Per-slot transfer: worker thread, off the hot path, not the bottleneck.
        # Batching would need to stack pinned-host arrays (no CPU implementation).
        staged: list[jax.Array] = []
        loaded: dict[int, jax.Array] = {}
        for buffer_id in host_buffer_ids:
            self._require_allocated(buffer_id)
            slot = self._slots[buffer_id]
            if slot is None:
                raise RuntimeError(f"stage_load from empty buffer_id={buffer_id}")
            packed_dev = jax.device_put(slot, self._device_packed_sharding)
            loaded[buffer_id] = packed_dev
            staged.append(packed_dev)
        jax.block_until_ready(staged)
        with self._pending_load_lock:
            self._pending_load.update(loaded)

    def flush_load(self, host_buffer_ids: list[int], device_indices: list[int]) -> None:
        """H2D phase 2: scatter the staged pages into the KV buffer via the
        in-place aliased Pallas kernel (``write_kv_layer``).

        Sync + on the KV-owning thread, in a donation-safe window: it reads +
        reassigns ``kv_buffer[layer]``, which the forward donates every step.
        ``device_indices`` are GLOBAL device page ids, expanded to ``page_size``
        token slots for the kernel ``loc``.
        """
        if len(host_buffer_ids) != len(device_indices):
            raise ValueError(
                f"host_buffer_ids ({len(host_buffer_ids)}) and device_indices "
                f"({len(device_indices)}) length mismatch"
            )
        if not host_buffer_ids:
            return
        self._require_device_pages(device_indices)
        from sgl_jax.srt.mem_cache.memory_pool import write_kv_layer

        PS = self._page_size
        staged_pages: list[jax.Array] = []
        for buffer_id in host_buffer_ids:
            self._require_allocated(buffer_id)
            with self._pending_load_lock:
                packed = self._pending_load.pop(buffer_id, None)
            if packed is None:
                raise RuntimeError(
                    f"flush_load with no staged page for buffer_id={buffer_id}; "
                    f"stage_load() must run first"
                )
            staged_pages.append(packed)

        n = len(host_buffer_ids)
        # Bucket the page count so write_kv_layer compiles once per bucket, not
        # per distinct n. Pad pages by duplicating page 0; the padding tokens get
        # loc=-1, which the kernel skips, so the duplicate data is never written.
        n_b = self._pad_to_page_bucket(n)
        sel = list(range(n)) + [0] * (n_b - n)
        stack = jnp.stack(
            [staged_pages[i] for i in sel], axis=0
        )  # (n_b, layer_num, *per_layer_shape)

        # Expand global page ids -> absolute device token slots (page*PS + offset),
        # then pad to n_b pages with -1 (skipped by the kernel).
        dev_pages = np.asarray(device_indices, dtype=np.int64)
        loc_np = (dev_pages[:, None] * PS + np.arange(PS, dtype=np.int64)).reshape(-1)
        if n_b > n:
            loc_np = np.concatenate([loc_np, -np.ones((n_b - n) * PS, dtype=loc_np.dtype)])
        dp = self._device_pool
        loc = jax.device_put(
            jnp.asarray(loc_np, dtype=jnp.int32),
            NamedSharding(dp.mesh, PartitionSpec(dp.attention_data_partition_axis)),
        )
        total_tokens = n_b * PS
        # layer_kv fed to write_kv_layer is [total_tokens, 1, *per_head_tail]. The
        # fold of (n_b, page_size) -> total_tokens crosses replicated leading axes
        # but keeps the tensor-sharded head axis, so the reshape needs an explicit
        # out_sharding (matches write_kv_layer's own fused_sharding).
        folded_sharding = NamedSharding(
            dp.mesh,
            PartitionSpec(
                dp.attention_data_partition_axis,
                None,
                dp.kv_partition_axis,
                None,
                None,
            ),
        )
        for layer in range(self._layer_num):
            # One slice for the whole batch's layer (page axis kept), folded to
            # tokens — no per-page gather/concatenate.
            layer_kv = stack[:, layer]  # (n_b, page_size, *per_head_tail)
            layer_kv = jax.lax.reshape(
                layer_kv,
                (total_tokens, 1) + tuple(layer_kv.shape[2:]),
                out_sharding=folded_sharding,
            )
            dp.kv_buffer[layer] = write_kv_layer(
                layer_kv,
                loc,
                dp.kv_buffer[layer],
                PS,
                dp.kv_partition_axis,
                dp.attention_data_partition_axis,
                dp.mesh,
            )
        jax.block_until_ready(dp.kv_buffer)

    def precompile_transfers(self, max_pages: int | None = None) -> None:
        """Warm the JIT/Pallas compile of the four transfer kernels for every
        page bucket serving can hit, so compilation never lands on the scheduler
        thread mid-serving (the dominant ON-vs-OFF latency gap). Device pages
        ``[0, r)`` are scratch: safe because serving starts with an empty cache
        and overwrites every page before any read.
        """
        from sgl_jax.srt.disaggregation.prefill import _KV_GATHER_PAGE_BUCKETS

        # Largest single transfer serving can do = min(host slots, device pages).
        # ``device_pool.size`` is token capacity, so // page_size. A transfer of
        # ``r`` real pages compiles shape ``_pad_to_page_bucket(r)``; warm by REAL
        # count (never more than ``cap_real`` slots, so alloc always fits) and let
        # the transfer pad internally — covering the top partial bucket too.
        dev_pages_total = int(self._device_pool.size) // self._page_size
        cap_real = min(self._pool_size, dev_pages_total)
        if max_pages is not None:
            cap_real = min(cap_real, int(max_pages))
        if cap_real <= 0:
            return
        real_counts = [b for b in _KV_GATHER_PAGE_BUCKETS if b <= cap_real]
        if not real_counts or real_counts[-1] < cap_real:
            real_counts.append(cap_real)
        t0 = time.perf_counter()
        warmed = 0
        for r in real_counts:
            slots = self.alloc(r)
            if slots is None:
                logger.warning(
                    "hicache precompile: host pool too small for %d pages "
                    "(free=%d); stopping warmup early",
                    r,
                    self.available_size(),
                )
                break
            host_ids = [int(x) for x in slots]
            device_idx = list(range(r))
            try:
                self.stage_backup(device_idx, host_ids)
                self.flush_backup(host_ids)
                self.stage_load(host_ids)
                self.flush_load(host_ids, device_idx)
            except Exception as e:
                # A page count too large for the write_kv_layer Pallas kernel
                # (SMEM overflow) must never crash the server: serving never
                # drives a transfer that big either. Log and stop.
                self.free(slots)
                logger.warning(
                    "hicache precompile: %d pages failed to warm (%s); stopping "
                    "at the largest kernel-safe size",
                    r,
                    type(e).__name__,
                )
                break
            self.free(slots)
            warmed += 1
        logger.info(
            "hicache precompile: warmed %d shapes (max=%d pages) in %.1fs",
            warmed,
            real_counts[warmed - 1] if warmed else 0,
            time.perf_counter() - t0,
        )

    def available_size(self) -> int:
        """Free page slots (HiCache control plane counts in pages)."""
        with self._lock:
            return len(self._free_ids)

    def total_size(self) -> int:
        """Total page slots in the pool (free + in-use)."""
        return self._pool_size

    def copy_from_device(self, layers: list[jax.Array], buffer_id: int) -> StagedData:
        # ABC contract only — the HiCache upper layer never calls this (it uses
        # copy_into so no jax.Array crosses the boundary). Present so the class
        # is not abstract and so a non-retaining caller could still borrow.
        self._require_allocated(buffer_id)
        if len(layers) != self._layer_num:
            raise ValueError(f"expected {self._layer_num} layers, got {len(layers)}")
        packed = jnp.stack(list(layers), axis=0)
        host_packed = jax.device_put(packed, self._host_sharding)
        jax.block_until_ready(host_packed)
        self._slots[buffer_id] = host_packed
        return StagedData(buffer_id=buffer_id, array_pytree=[host_packed])

    # ------------------------------------------------------------------
    # LRU / lock_ref mechanism (private to this class, not in the ABC).
    # The pool only provides the mechanism; victim selection lives in the
    # tree cache.
    # ------------------------------------------------------------------

    def _drop_pending(self, buffer_id: int) -> None:
        # Drop any orphaned staged transfer for a slot being freed, so a later
        # reuse of this id can't have flush_backup/flush_load pop stale data into
        # the reused slot. Called under self._lock (lock order: _lock -> pending).
        with self._pending_lock:
            self._pending_gather.pop(buffer_id, None)
        with self._pending_load_lock:
            self._pending_load.pop(buffer_id, None)

    def _require_allocated(self, buffer_id: int) -> None:
        # Page-id boundary invariant: an id handed to a transfer/lock op must be
        # currently allocated. A free-list id would otherwise get _slots/_lock_ref
        # written while still claimable by the next alloc() -> double ownership.
        if not (0 <= buffer_id < self._pool_size):
            raise ValueError(f"buffer_id={buffer_id} outside pool range [0, {self._pool_size})")
        if not self._allocated[buffer_id]:
            raise RuntimeError(f"buffer_id={buffer_id} is not allocated (in free-list)")

    def _require_device_pages(self, device_indices) -> None:
        # Reject out-of-range device page ids: a negative id would wrap (JAX
        # negative indexing on D2H gather) or expand to negative loc on H2D
        # (only -1 is treated as padding by the kernel; other negatives become
        # real DMA targets and corrupt pages).
        n_dev = int(self._device_pool.size) // self._page_size
        for idx in device_indices:
            if not (0 <= int(idx) < n_dev):
                raise ValueError(f"device page id={idx} outside range [0, {n_dev})")

    def inc_lock_ref(self, buffer_id: int) -> None:
        with self._lock:
            self._require_allocated(buffer_id)
            self._lock_ref[buffer_id] += 1

    def dec_lock_ref(self, buffer_id: int) -> None:
        with self._lock:
            self._require_allocated(buffer_id)
            if self._lock_ref[buffer_id] <= 0:
                raise RuntimeError(f"dec_lock_ref underflow on buffer_id={buffer_id}")
            self._lock_ref[buffer_id] -= 1


def make_unit_mesh() -> Mesh:
    """Convenience for tests: a single-device mesh with axis ``x``."""

    devices = jax.local_devices()
    return Mesh(np.asarray(devices[:1]).reshape(1), axis_names=("x",))
