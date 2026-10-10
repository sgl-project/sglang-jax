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
from functools import partial
from typing import Any, Callable

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


def _host_to_device_array(x: np.ndarray, sharding: NamedSharding) -> jax.Array:
    """Place a host ``numpy`` array on the mesh WITHOUT a cross-host collective.

    ``jax.device_put(np_array, <non-fully-addressable NamedSharding>)`` runs
    ``multihost_utils.assert_equal`` -- an all-gather across every process -- to
    check all hosts passed the same value. HiCache's index arrays are tiny,
    derived from scheduler state that is identical on every host by
    construction, and are built on threads other than the forward launcher, so
    that implicit collective is both wasted and a launch-order hazard on
    multi-host meshes. Build the array per addressable device instead.
    """
    x = np.ascontiguousarray(x)
    return jax.make_array_from_callback(x.shape, sharding, lambda idx: x[idx])


@dataclass(frozen=True)
class _BufferGroup:
    """One list of per-layer device buffers that a host slot must carry.

    A host slot packs one page of every buffer in the group over its layers as
    ``(layer_num, *per_layer_shape)``. MHA pools have a single group
    (``kv_buffer``); the MLA pool under ``dsa_sparse`` has two (latent
    ``kv_buffer`` + ``indexer_key_buffer``) with different trailing shapes, so
    they cannot share one array and are staged/transferred side by side.
    """

    name: str
    get_buffers: Callable[[], list[jax.Array]]
    set_buffer: Callable[[int, jax.Array], None]
    layer_num: int
    per_layer_shape: tuple[int, ...]
    dtype: Any
    partition_spec: PartitionSpec
    # Derived shardings (see LRUHostKVPool.__init__ for the rationale).
    slot_spec: PartitionSpec
    host_sharding: NamedSharding
    device_packed_sharding: NamedSharding
    batched_layer_sharding: NamedSharding

    @staticmethod
    def build(
        name: str,
        *,
        get_buffers: Callable[[], list[jax.Array]],
        set_buffer: Callable[[int, jax.Array], None],
        layer_num: int,
        per_layer_shape: tuple[int, ...],
        dtype: Any,
        mesh: Mesh,
        partition_spec: PartitionSpec,
    ) -> _BufferGroup:
        # A slot is the page packed over layers: prepend a replicated layer
        # axis and drop the device pool's leading page/DP axis (a single
        # gathered page is not DP-sharded). Any TP axis on trailing dims is kept.
        slot_spec = PartitionSpec(None, *tuple(partition_spec)[1:])
        return _BufferGroup(
            name=name,
            get_buffers=get_buffers,
            set_buffer=set_buffer,
            layer_num=layer_num,
            per_layer_shape=tuple(per_layer_shape),
            dtype=dtype,
            partition_spec=partition_spec,
            slot_spec=slot_spec,
            host_sharding=_make_host_sharding(mesh, slot_spec),
            device_packed_sharding=NamedSharding(mesh, slot_spec),
            # Batched gather output (one layer, MANY pages): page axis is back and
            # replicated (gathered pages are not DP-distributed); same form PD uses.
            batched_layer_sharding=NamedSharding(mesh, PartitionSpec(None, *tuple(partition_spec)[1:])),
        )


@partial(jax.jit, static_argnames=("mesh", "cache_spec"), donate_argnames=("kv_cache",))
def write_pages_layer(pages, local_page_ids, kv_cache, mesh, cache_spec):
    """One-layer, page-granular in-place write: ``kv_cache[local_page_ids[i]] = pages[i]``.

    Layout-agnostic counterpart of :func:`memory_pool.write_kv_layer`. The MHA
    path expands pages into per-token slots and drives a 5-D fused Pallas
    kernel that assumes ``kv_cache.shape[1] == page_size`` and a head axis; the
    MLA pool stores a page as a ``(page_size // packing, packing, kv_dim)`` token
    grid with no head axis, so neither assumption holds. HiCache only ever moves
    whole pages, so a page-level ``dynamic-update-slice`` is sufficient — and it
    is exactly the device layout already, so no reshape is needed.

    ``pages`` is ``[n_b, *per_layer_shape]`` laid out rank-major (``n_b ==
    per_shard * dp_size``); ``local_page_ids`` are SHARD-LOCAL page ids, ``-1``
    for padding (dropped). Under ``shard_map`` each data-shard receives its own
    contiguous chunk of both, matching the shard-local page indexing of its
    ``kv_cache`` slice.
    """
    data_axis = tuple(cache_spec)[0]
    pages_spec = PartitionSpec(data_axis, *tuple(cache_spec)[1:])
    pages = jax.sharding.reshard(pages, NamedSharding(mesh, pages_spec))
    local_page_ids = jax.sharding.reshard(
        local_page_ids.astype(jnp.int32), NamedSharding(mesh, PartitionSpec(data_axis))
    )

    @jax.shard_map(
        mesh=mesh,
        in_specs=(pages_spec, PartitionSpec(data_axis), cache_spec),
        out_specs=cache_spec,
        check_vma=False,
    )
    def _update(local_pages, local_ids, local_cache):
        # Out-of-range index + mode="drop" skips the padding rows.
        safe = jnp.where(local_ids < 0, local_cache.shape[0], local_ids)
        return local_cache.at[safe].set(local_pages, mode="drop")

    return _update(pages, local_page_ids, kv_cache)


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
    def available_size(self) -> int:
        """Entries currently free."""

    @abc.abstractmethod
    def total_size(self) -> int:
        """Total entries in the pool (free + in-use)."""

    # Role-specific transfer primitives, all defaulted (not abstract) so each
    # concrete pool implements only the ones its role needs and stays concrete:
    # PD overrides copy_from_device; HiCache overrides copy_into/copy_to_device.

    def copy_from_device(self, layers: list[jax.Array], buffer_id: int) -> StagedData:
        """D2H staging for PD ``producer_handoff``: write ``layers`` into the
        pre-reserved ``buffer_id`` and return a :class:`StagedData`. Release is
        owned by the caller. PD-only; HiCache uses :meth:`copy_into`."""
        raise NotImplementedError

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

    ``self._lock`` guards the free-list, ``_lock_ref`` and ``_generation``. Per-slot
    reads rely on the caller holding an exclusive reservation, but the transfer
    write/publish points (``_slots`` in flush_backup, ``_pending_*`` in
    stage_backup/stage_load, the scatter in flush_load) capture the slot's epoch
    (``_generation``) at entry and re-check it under ``_lock`` before publishing.
    A free()+realloc() that reused the slot mid-transfer leaves ``_allocated`` True
    but bumps the epoch, so the stale publish is dropped (the ABA a bare
    ``_allocated`` re-check missed). Lock order is
    ``_lock -> _pending_lock``/``_pending_load_lock``.
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

        # Buffer groups: every list of per-layer device buffers a page lives in.
        # Group 0 is the pool's ``kv_buffer`` (the only group for MHA pools).
        # The MLA pool under ``dsa_sparse`` carries a second paged buffer list,
        # ``indexer_key_buffer`` (lightning-indexer keys, different kv_dim), that
        # must be offloaded/restored alongside the latent KV or a restored
        # prefix silently attends with zeroed indexer keys.
        groups = [
            _BufferGroup.build(
                "kv",
                get_buffers=lambda dp=device_pool: dp.kv_buffer,
                set_buffer=lambda layer, buf, dp=device_pool: dp.kv_buffer.__setitem__(layer, buf),
                layer_num=layer_num,
                per_layer_shape=self._per_layer_shape,
                dtype=dtype,
                mesh=mesh,
                partition_spec=partition_spec,
            )
        ]
        indexer_bufs = getattr(device_pool, "indexer_key_buffer", None)
        if indexer_bufs:
            idx0 = indexer_bufs[0]
            idx_spec = getattr(getattr(idx0, "sharding", None), "spec", partition_spec)
            groups.append(
                _BufferGroup.build(
                    "indexer",
                    get_buffers=lambda dp=device_pool: dp.indexer_key_buffer,
                    set_buffer=lambda slot, buf, dp=device_pool: dp.indexer_key_buffer.__setitem__(
                        slot, buf
                    ),
                    layer_num=len(indexer_bufs),
                    per_layer_shape=tuple(int(d) for d in idx0.shape[1:]),
                    dtype=idx0.dtype,
                    mesh=mesh,
                    partition_spec=idx_spec,
                )
            )
            logger.info(
                "host pool %s: offloading DSA indexer-key buffers too (%d slots, page shape %s)",
                pool_name,
                len(indexer_bufs),
                tuple(int(d) for d in idx0.shape[1:]),
            )
        self._groups: tuple[_BufferGroup, ...] = tuple(groups)

        # Legacy single-group aliases (group 0), kept for readers of these names.
        g0 = self._groups[0]
        self._slot_spec = g0.slot_spec
        self._host_sharding = g0.host_sharding
        self._device_packed_sharding = g0.device_packed_sharding
        self._batched_layer_sharding = g0.batched_layer_sharding

        # H2D scatter flavour. The MHA fused pool exposes
        # ``attention_data_partition_axis`` and is written through the
        # token-granular Pallas kernel (``write_kv_layer``); any other pool
        # (MLA latent/indexer: no token axis, no head axis) goes through the
        # page-granular ``write_pages_layer``.
        self._token_scatter = hasattr(device_pool, "attention_data_partition_axis")
        if not self._token_scatter:
            logger.info(
                "host pool %s: device pool %s has no fused MHA layout; using page-granular H2D scatter",
                pool_name,
                type(device_pool).__name__,
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
        # One host array per buffer group, or None while the slot is empty.
        self._slots: list[list[jax.Array] | None] = [None] * pool_size
        self._lock_ref: list[int] = [0] * pool_size
        # Allocated-state mirror of the free-list (O(1) membership for the
        # page-id boundary checks below; a bare ``in self._free_ids`` is O(n)).
        self._allocated: list[bool] = [False] * pool_size
        # Per-slot allocation epoch, bumped on every alloc (reserve/alloc). A
        # transfer captures the epoch at entry and re-checks it at publish: a
        # free()+realloc() that reused the slot mid-transfer leaves _allocated
        # True again but bumps the epoch, so the stale publish is dropped instead
        # of resurrecting the old allocation's data into the new owner's slot
        # (the ABA a bare _allocated re-check can't catch).
        self._generation: list[int] = [0] * pool_size
        self._peak_used = 0
        self._exhaust_count = 0
        self._last_exhaust_log = 0.0
        # D2H: pages gathered by stage_backup (buffer-owning thread, sync) await
        # their async host transfer in flush_backup (D2H worker), keyed by
        # buffer_id; each value is ``(epoch, [array per group])`` so flush can
        # reject a stale publish from a slot reused since staging.
        self._pending_lock = threading.Lock()
        self._pending_gather: dict[int, tuple[int, list[jax.Array]]] = {}
        # H2D (mirror): pages staged by stage_load (worker, slow device_put, never
        # touches kv_buffer) await their cheap in-place scatter in flush_load
        # (buffer-owning thread, donation-safe window), keyed by buffer_id; value
        # is ``(epoch, [array per group])`` for the same staleness guard.
        self._pending_load_lock = threading.Lock()
        self._pending_load: dict[int, tuple[int, list[jax.Array]]] = {}

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
            self._generation[buffer_id] += 1
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
                self._generation[pid] += 1
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
        # Capture each slot's epoch under _lock alongside the allocated check, so
        # the publish below can tell whether a free()+realloc() reused the slot
        # while this (slow) gather was in flight.
        with self._lock:
            for buffer_id in host_buffer_ids:
                self._require_allocated(buffer_id)
            gens = {bid: self._generation[bid] for bid in host_buffer_ids}
        self._require_device_pages(device_indices)
        n = len(device_indices)
        # Bucket the page count so each layer's gather compiles once per bucket
        # (PD pattern), not once per distinct n. Pad with page 0; the padding
        # rows are gathered then discarded by the per-buffer slice below.
        n_b = self._pad_to_page_bucket(n)
        dev_idx_np = np.asarray(device_indices, dtype=np.int32)
        if n_b > n:
            dev_idx_np = np.concatenate([dev_idx_np, np.zeros(n_b - n, dtype=dev_idx_np.dtype)])
        page_indices = _host_to_device_array(
            dev_idx_np, NamedSharding(self._mesh, PartitionSpec(None))
        )
        # One batched gather per layer per buffer group (page axis back); keep
        # the per-layer loop so peak HBM stays at one layer's bucket (PD caps it
        # this way too). The gather indexes axis 0 only, so it is layout-agnostic
        # (MHA token rows and MLA token grids alike).
        packed_groups: list[jax.Array] = []
        for group in self._groups:
            buffers = group.get_buffers()
            per_layer = [
                self._jit_gather_one_layer(
                    buffers[layer], page_indices, group.batched_layer_sharding
                )
                for layer in range(group.layer_num)
            ]
            packed_groups.append(jnp.stack(per_layer, axis=0))  # (layer_num, n_b, *shape)
        jax.block_until_ready(packed_groups)
        # Slice the real pages back per buffer_id (cheap views on the now-
        # materialized arrays); padding rows [n:] are dropped.
        gathered = {
            bid: [packed[:, i] for packed in packed_groups] for i, bid in enumerate(host_buffer_ids)
        }
        # Publish under _lock, tagging each entry with the epoch captured at
        # entry. A free()+realloc() since then leaves _allocated True but bumps
        # the epoch, so the stale gather is dropped rather than landing in the
        # reused slot (a bare _allocated re-check can't tell the new owner apart).
        # Lock order is _lock -> _pending_lock, matching free()/release().
        with self._lock, self._pending_lock:
            for buffer_id in host_buffer_ids:
                if self._allocated[buffer_id] and self._generation[buffer_id] == gens[buffer_id]:
                    self._pending_gather[buffer_id] = (
                        gens[buffer_id],
                        gathered[buffer_id],
                    )

    def flush_backup(self, host_buffer_ids: list[int], *, block: bool = True) -> list[jax.Array]:
        """D2H phase 2: ``device_put`` the staged pages into their host slots.

        Touches only ``_pending_gather`` / ``_slots`` (never the KV buffer).
        With ``block=False`` the host copies are only *issued*; the returned
        arrays can be waited on from any thread (waiting is not a launch).
        """
        if not host_buffer_ids:
            return []
        # Per-slot host transfer. Batching would need to slice a pinned-host
        # array, which has no CPU implementation.
        staged: list[jax.Array] = []
        for buffer_id in host_buffer_ids:
            with self._pending_lock:
                entry = self._pending_gather.pop(buffer_id, None)
            if entry is None:
                raise RuntimeError(
                    f"flush_backup with no staged gather for buffer_id={buffer_id}; "
                    f"stage_backup() must run (synchronously) first"
                )
            gen, packed_list = entry
            host_packed = [
                jax.device_put(packed, group.host_sharding)
                for packed, group in zip(packed_list, self._groups)
            ]
            # Publish into _slots only if the slot is still the same allocation
            # that staged this gather (epoch unchanged): a release()/free()
            # (+realloc) after the pop above would otherwise resurrect stale data
            # into the reused slot.
            with self._lock:
                if self._allocated[buffer_id] and self._generation[buffer_id] == gen:
                    self._slots[buffer_id] = host_packed
            staged.extend(host_packed)
        if block:
            jax.block_until_ready(staged)
        return staged

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

    def stage_load(self, host_buffer_ids: list[int], *, block: bool = True) -> list[jax.Array]:
        """H2D phase 1: ``device_put`` host slots onto the device into per-buffer
        staging arrays.

        Touches only ``_slots`` / ``_pending_load`` (never the KV buffer); the
        cheap scatter is deferred to :meth:`flush_load`. With ``block=False`` the
        copies are only issued and the returned arrays can be awaited elsewhere.
        """
        if not host_buffer_ids:
            return []
        # Per-slot transfer. Batching would need to stack pinned-host arrays
        # (no CPU implementation).
        staged: list[jax.Array] = []
        loaded: dict[int, list[jax.Array]] = {}
        # Read the slot refs + capture epochs under _lock, so the publish can
        # reject a slot reused since this (slow) device_put started.
        with self._lock:
            for buffer_id in host_buffer_ids:
                self._require_allocated(buffer_id)
            gens = {bid: self._generation[bid] for bid in host_buffer_ids}
            slots = {bid: self._slots[bid] for bid in host_buffer_ids}
        for buffer_id in host_buffer_ids:
            slot = slots[buffer_id]
            if slot is None:
                raise RuntimeError(f"stage_load from empty buffer_id={buffer_id}")
            packed_dev = [
                jax.device_put(arr, group.device_packed_sharding)
                for arr, group in zip(slot, self._groups)
            ]
            loaded[buffer_id] = packed_dev
            staged.extend(packed_dev)
        if block:
            jax.block_until_ready(staged)
        # Publish tagged with the entry epoch (same ABA guard as stage_backup): a
        # free()+realloc() mid-stage bumps the epoch, so the stale device copy is
        # dropped instead of landing in the reused slot's pending load. Lock order
        # _lock -> _pending_load_lock.
        with self._lock, self._pending_load_lock:
            for buffer_id in host_buffer_ids:
                if self._allocated[buffer_id] and self._generation[buffer_id] == gens[buffer_id]:
                    self._pending_load[buffer_id] = (
                        gens[buffer_id],
                        loaded[buffer_id],
                    )

    def flush_load(
        self, host_buffer_ids: list[int], device_indices: list[int], *, block: bool = True
    ) -> None:
        """H2D phase 2: scatter the staged pages into the device buffers in place.

        Sync + on the KV-owning thread, in a donation-safe window: it reads +
        reassigns ``kv_buffer[layer]`` (and the indexer buffers for DSA), which
        the forward donates every step. ``device_indices`` are GLOBAL device page
        ids.

        Two scatter flavours, picked at construction (``self._token_scatter``):

        * MHA fused pool ``[pages, page_size, heads*2/pack, pack, hd]``: pages are
          expanded to per-token ``loc`` and written by the in-place Pallas kernel
          (``write_kv_layer``), as before.
        * Any other layout (MLA ``[pages, page_size/pack, pack, kv_dim]``, DSA
          indexer keys): whole pages are written by :func:`write_pages_layer`
          with shard-local page ids. No token axis is assumed.
        """
        if len(host_buffer_ids) != len(device_indices):
            raise ValueError(
                f"host_buffer_ids ({len(host_buffer_ids)}) and device_indices "
                f"({len(device_indices)}) length mismatch"
            )
        if not host_buffer_ids:
            return
        self._require_device_pages(device_indices)

        # Capture each slot's current epoch under _lock; a page whose epoch moved
        # since stage_load (free()+realloc()) is stale and must not be scattered
        # into the caller's device pages.
        with self._lock:
            cur_gen: dict[int, int | None] = {}
            for buffer_id in host_buffer_ids:
                if not (0 <= buffer_id < self._pool_size):
                    raise ValueError(
                        f"buffer_id={buffer_id} outside pool range [0, {self._pool_size})"
                    )
                cur_gen[buffer_id] = (
                    self._generation[buffer_id] if self._allocated[buffer_id] else None
                )
        staged_pages: list[list[jax.Array]] = []  # per slot: one array per group
        fresh: list[bool] = []
        for buffer_id in host_buffer_ids:
            with self._pending_load_lock:
                entry = self._pending_load.pop(buffer_id, None)
            if entry is None:
                raise RuntimeError(
                    f"flush_load with no staged page for buffer_id={buffer_id}; "
                    f"stage_load() must run first"
                )
            gen, packed_list = entry
            staged_pages.append(packed_list)
            fresh.append(cur_gen[buffer_id] == gen)

        n = len(host_buffer_ids)
        dp = self._device_pool
        dp_size = int(dp.dp_size)
        num_dev_pages = int(dp.kv_buffer[0].shape[0])
        pages_per_shard = num_dev_pages // dp_size

        # The KV buffer's page axis is sharded contiguously across the data axis:
        # shard d owns global pages [d*pages_per_shard, (d+1)*pages_per_shard).
        # Both scatter kernels shard_map their inputs into dp_size equal
        # contiguous chunks and index each shard's LOCAL slice -- so a global page
        # must land in its owning shard's chunk, addressed as a shard-local offset
        # (passing global ids writes to the wrong shard under dp>1). Give every
        # shard a fixed per_shard bucket (pad_to_page_bucket(n) == the worst case
        # where all n pages fall on one shard) so the compiled shape depends only
        # on (n bucket, dp_size). For dp_size==1 this collapses to one
        # global==local chunk -- identical to the single-shard layout.
        per_shard = self._pad_to_page_bucket(n)
        n_b = per_shard * dp_size
        # Bucket each transfer page into its owning shard's slot list; padding
        # slots keep staged index 0 (data ignored) and page id -1 (skipped).
        shard_sel = [[0] * per_shard for _ in range(dp_size)]
        shard_loc = [[-1] * per_shard for _ in range(dp_size)]
        fill_pos = [0] * dp_size
        for k, gp in enumerate(int(p) for p in device_indices):
            if not fresh[k]:
                continue  # slot freed/realloc'd since stage_load: drop stale page
            d = gp // pages_per_shard
            pos = fill_pos[d]
            shard_sel[d][pos] = k
            shard_loc[d][pos] = gp - d * pages_per_shard
            fill_pos[d] = pos + 1
        sel = [k for chunk in shard_sel for k in chunk]
        # Shard-local page ids in rank-major order, -1 = padding.
        loc_pages = np.asarray([lp for chunk in shard_loc for lp in chunk], dtype=np.int64)

        if self._token_scatter:
            self._flush_load_tokens(staged_pages, sel, loc_pages, n_b, block)
        else:
            self._flush_load_pages(staged_pages, sel, loc_pages, block)

    def _flush_load_tokens(self, staged_pages, sel, loc_pages, n_b, block) -> None:
        """MHA fused layout: expand pages to token slots, Pallas in-place write."""
        from sgl_jax.srt.mem_cache.memory_pool import write_kv_layer

        dp = self._device_pool
        PS = self._page_size
        group = self._groups[0]  # MHA pools have exactly one buffer group
        stack = jnp.stack(
            [staged_pages[i][0] for i in sel], axis=0
        )  # (n_b, layer_num, *per_layer_shape)

        # Expand shard-local page ids -> shard-local token slots (page*PS + offset),
        # -1 for padding. The per-shard chunk order matches the loc/data data-axis
        # split so each shard receives exactly the pages it owns.
        loc_np = np.where(
            loc_pages[:, None] < 0,
            -1,
            loc_pages[:, None] * PS + np.arange(PS, dtype=np.int64),
        ).reshape(-1)
        loc = _host_to_device_array(
            loc_np.astype(np.int32),
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
        buffers = group.get_buffers()
        for layer in range(group.layer_num):
            # One slice for the whole batch's layer (page axis kept), folded to
            # tokens — no per-page gather/concatenate.
            layer_kv = stack[:, layer]  # (n_b, page_size, *per_head_tail)
            layer_kv = jax.lax.reshape(
                layer_kv,
                (total_tokens, 1) + tuple(layer_kv.shape[2:]),
                out_sharding=folded_sharding,
            )
            group.set_buffer(
                layer,
                write_kv_layer(
                    layer_kv,
                    loc,
                    buffers[layer],
                    PS,
                    dp.kv_partition_axis,
                    dp.attention_data_partition_axis,
                    dp.mesh,
                ),
            )
        if block:
            jax.block_until_ready(group.get_buffers())

    def _flush_load_pages(self, staged_pages, sel, loc_pages, block) -> None:
        """Layout-agnostic: whole-page in-place write per layer per buffer group.

        The staged slot already has the device page layout (whatever it is), so
        the only work is placing page ``i`` at shard-local page ``loc_pages[i]``
        of the owning data-shard. No token expansion, no reshape.
        """
        dp = self._device_pool
        local_ids = jnp.asarray(loc_pages, dtype=jnp.int32)
        for gi, group in enumerate(self._groups):
            stack = jnp.stack(
                [staged_pages[i][gi] for i in sel], axis=0
            )  # (n_b, layer_num, *per_layer_shape)
            buffers = group.get_buffers()
            cache_spec = buffers[0].sharding.spec
            for layer in range(group.layer_num):
                group.set_buffer(
                    layer,
                    write_pages_layer(
                        stack[:, layer],  # (n_b, *per_layer_shape), page axis replicated
                        local_ids,
                        buffers[layer],
                        dp.mesh,
                        cache_spec,
                    ),
                )
            if block:
                jax.block_until_ready(group.get_buffers())

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
                self.free(slots)
                if warmed == 0:
                    # The smallest transfer failing is not a kernel-size limit,
                    # it means this device pool / layout is not supported by the
                    # transfer path. Serving would come up "healthy" and crash the
                    # scheduler on the first host-cache hit; fail at startup instead.
                    raise RuntimeError(
                        f"hicache precompile: the smallest transfer ({r} page(s)) failed "
                        f"for device pool {type(self._device_pool).__name__}; HiCache is "
                        f"not functional for this configuration"
                    ) from e
                # A page count too large for the write_kv_layer Pallas kernel
                # (SMEM overflow) must never crash the server: serving never
                # drives a transfer that big either. Log and stop.
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
        #
        # Bound is the kv_buffer's page-axis length, NOT size // page_size: the
        # fused buffer carries an extra sentinel page (memory_pool packs
        # ``(size + page_size*dp_size) // page_size`` rows) and the allocator
        # hands out 1-based page ids up to that top row, so ``size // page_size``
        # would wrongly reject the highest legitimately-allocated page.
        n_dev = int(self._device_pool.kv_buffer[0].shape[0])
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
