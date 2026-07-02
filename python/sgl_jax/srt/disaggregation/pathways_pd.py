"""Pathways single-controller cross-slice P/D disaggregation.

Architecture: one Python process (Pathways head) sees all devices across
N slices via IFRT proxy. Build prefill_mesh on slice 0, decode_mesh on
slice 1, run two ModelRunners. KV transfer = jax.device_put across meshes
(IFRT runtime handles per-host DCN fan-out, ~14 GB/s/host on v7x vs ~4.5
for jax.experimental.transfer on the same hardware).
"""

from __future__ import annotations

import logging
from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

__all__ = [
    "group_by_slice",
    "make_slice_meshes",
    "PathwaysPDKVTransfer",
    "migrate_reqs_p_to_d",
    "slots_to_ordered_pages",
]

logger = logging.getLogger(__name__)

_NPG_BUCKETS = (8, 16, 32, 64)
# Upper bound on pages per stacked transfer. 64pg x 78L x 80KB ~= 0.39 GB
# transient HBM (x2 with the recv-side donate buffer) — fits the smallest
# headroom we measured (v6e fp8: ~1.3 GB free after weights + KV + jit exec).
_NPG_BATCH_MAX = 64


def _bucket_npg(n: int) -> int:
    for b in _NPG_BUCKETS:
        if n <= b:
            return b
    return _NPG_BATCH_MAX


def slots_to_ordered_pages(slots: np.ndarray, page_size: int) -> np.ndarray:
    """Token slots -> page indices, deduped, preserving first-seen order.

    PD forces disable_radix so a single req's slots are page-block contiguous
    (== slots[::page_size]//page_size), but a chunked req can resume mid-page.
    The generic dedup handles both.
    """
    pages = np.asarray(slots, np.int64) // page_size
    _, first_idx = np.unique(pages, return_index=True)
    return pages[np.sort(first_idx)].astype(np.int32)


def migrate_reqs_p_to_d(
    reqs: list,
    page_size: int,
    p_r2t,
    p_alloc,
    d_r2t,
    d_alloc,
    kv_transfer,
) -> None:
    """Migrate reqs whose prefill finished from P pool into D pool.

    Per req: read P slots -> alloc page-aligned D slots + req_pool_idx ->
    rewrite req.{req_pool_idx, prefix_indices, kv_committed_len} -> free P.
    Page pairs are accumulated then transferred in one batched kv_transfer.
    """
    if not reqs:
        return
    p_pages_all, d_pages_all = [], []
    for r in reqs:
        seq_len = len(r.fill_ids)
        p_idx_old = r.req_pool_idx
        p_slots = p_r2t.req_to_token[p_idx_old, :seq_len].copy()
        p_pages = slots_to_ordered_pages(p_slots, page_size)
        n_pages = len(p_pages)

        d_slots = d_alloc.alloc(n_pages * page_size, dp_rank=r.dp_rank or 0)
        if d_slots is None:
            raise RuntimeError(
                f"[pathways_pd] D pool OOM during migrate: need {n_pages} pages, "
                f"avail {d_alloc.available_size()}"
            )
        d_pages = slots_to_ordered_pages(d_slots, page_size)
        # Remap by in-page offset: token at P-page k offset o -> D-page k offset o.
        p_slots64 = np.asarray(p_slots, np.int64)
        offsets = p_slots64 % page_size
        page_pos = {int(pg): k for k, pg in enumerate(p_pages)}
        tok_page_k = np.array([page_pos[int(s)] for s in p_slots64 // page_size], np.int64)
        d_slot_per_tok = (d_pages[tok_page_k].astype(np.int64) * page_size + offsets).astype(
            np.int32
        )

        p_alloc.free(p_slots, dp_rank=r.dp_rank or 0)
        p_r2t.free_slots.append(p_idx_old)
        r.req_pool_idx = None
        d_r2t.alloc([r])
        d_r2t.req_to_token[r.req_pool_idx, :seq_len] = d_slot_per_tok
        r.prefix_indices = d_slot_per_tok
        r.last_node = None
        r.cache_protected_len = 0
        r.kv_committed_len = seq_len
        r.kv_allocated_len = seq_len

        p_pages_all.append(p_pages)
        d_pages_all.append(d_pages)

    kv_transfer.transfer(np.concatenate(p_pages_all), np.concatenate(d_pages_all))


def group_by_slice(devs: list) -> dict[int, list]:
    """Group devices by slice_index. Fallback: split in half by id order."""
    groups: dict[int, list] = defaultdict(list)
    for d in devs:
        sid = getattr(d, "slice_index", None)
        if sid is not None:
            groups[sid].append(d)
    if len(groups) >= 2:
        return dict(groups)
    ordered = sorted(devs, key=lambda d: d.id)
    half = len(ordered) // 2
    return {0: ordered[:half], 1: ordered[half:]}


def make_slice_meshes(
    dp_size: int, tp_per_side: int, n_prefill: int = 1
) -> tuple[list[Mesh], Mesh]:
    """Build ([prefill_mesh_0..n_prefill-1], decode_mesh) from the first
    n_prefill+1 slices. Multi-P: n_prefill>1 fans prefill across
    slices to lift R_prefill (the bottleneck on device-bound models)."""
    devs = jax.devices()
    groups = group_by_slice(devs)
    sids = sorted(groups)
    need = n_prefill + 1
    if len(sids) < need:
        raise RuntimeError(
            f"pathways_pd n_prefill={n_prefill} requires >={need} slices, got {len(sids)} "
            f"(devices={len(devs)}, backend={jax.default_backend()})"
        )
    tp_axis = tp_per_side // dp_size
    et = (jax.sharding.AxisType.Explicit,) * 2

    def _mk(ds: list) -> Mesh:
        if len(ds) != tp_per_side:
            raise RuntimeError(f"slice size {len(ds)} != tp_per_side {tp_per_side}")
        return Mesh(
            np.asarray(ds).reshape(dp_size, tp_axis),
            axis_names=("data", "tensor"),
            axis_types=et,
        )

    p_meshes = [_mk(groups[sids[i]]) for i in range(n_prefill)]
    d_mesh = _mk(groups[sids[n_prefill]])
    logger.info(
        "[pathways_pd] slices=%d n_prefill=%d p_slices=%s d_slice=%d shape=%s",
        len(sids),
        n_prefill,
        sids[:n_prefill],
        sids[n_prefill],
        d_mesh.shape,
    )
    return p_meshes, d_mesh


def _make_pool_jits(p_mesh: Mesh, d_mesh: Mesh, p_sub, d_sub):
    """Build (gather_jit, scatter_jit, d_stack_shard) for one MHA-style sub-pool."""
    kv_spec = d_sub.kv_sharding.spec
    page_spec = P(None, *kv_spec[1:])
    stack_spec = P(None, *kv_spec)
    gather_jit = jax.jit(
        lambda bufs, idx: jnp.stack([b.at[idx].get(out_sharding=page_spec) for b in bufs])
    )
    scatter_jit = jax.jit(
        lambda bufs, idx, stacked: tuple(
            b.at[idx].set(stacked[i], out_sharding=kv_spec) for i, b in enumerate(bufs)
        ),
        donate_argnums=(0,),
    )
    return gather_jit, scatter_jit, NamedSharding(d_mesh, stack_spec)


class PathwaysPDKVTransfer:
    """Paged KV pool P-mesh -> D-mesh via cross-slice device_put.

    gather(P pages) -> stack L layers -> device_put to d_mesh sharding
    -> scatter into D pool (donate). Single transfer per batch maximizes
    payload (P0 measured 28.9 GB/s aggregate at 2048 MiB on 2x v7x-8).

    SWAKVPool: full + swa sub-pools transferred independently. Input/output
    pages are in the *full* index space; swa pages are derived per side via
    each allocator's full_to_swa_index_mapping (page-aligned 1:1 because PD
    forces disable_radix + no chunked prefill, so every alloc is page-head).
    """

    def __init__(self, p_mesh: Mesh, d_mesh: Mesh, p_pool, d_pool, **kw) -> None:
        self.p_mesh = p_mesh
        self.d_mesh = d_mesh
        self.d_pool = d_pool
        self.is_swa = hasattr(p_pool, "swa_kv_pool")
        if self.is_swa:
            self.page_size = kw["page_size"]
            self.p_mapping = kw["p_alloc"].full_to_swa_index_mapping
            self.d_mapping = kw["d_alloc"].full_to_swa_index_mapping
            sub_pools = [
                (p_pool.full_kv_pool, d_pool.full_kv_pool),
                (p_pool.swa_kv_pool, d_pool.swa_kv_pool),
            ]
        else:
            sub_pools = [(p_pool, d_pool)]
        # _jits[k] = (gather, scatter, d_stack_shard, p_sub_pool, d_sub_pool)
        self._jits = [
            (*_make_pool_jits(p_mesh, d_mesh, p_sub, d_sub), p_sub, d_sub)
            for p_sub, d_sub in sub_pools
        ]
        self._p_idx_shard = NamedSharding(p_mesh, P(None))
        self._d_idx_shard = NamedSharding(d_mesh, P(None))
        logger.info(
            "[pathways_pd] kv_transfer is_swa=%s sub_pools=%d layers=%s",
            self.is_swa,
            len(sub_pools),
            [p.layer_num for p, _ in sub_pools],
        )

    def _swa_pages(self, full_pages: np.ndarray, mapping) -> np.ndarray:
        # PD currently runs dp=1 only; mapping is a single np.array.
        m = mapping[0] if isinstance(mapping, list) else mapping
        return (m[full_pages.astype(np.int64) * self.page_size] // self.page_size).astype(np.int32)

    def gather_to_dmesh(self, p_pages: np.ndarray) -> tuple[jax.Array | tuple, int]:
        """gather P pool pages -> stack -> device_put to D mesh.

        Runs in the prefill thread; the returned d_stacked is a future on the
        D mesh — main-thread scatter consumes it via data-dep so the cross-slice
        RecvRefs interleaves with decode instead of stalling here for ~250ms.
        """
        p_pages = np.asarray(p_pages, np.int32)
        npg = len(p_pages)
        bucket = _bucket_npg(npg)
        if bucket > npg:
            p_pages = np.concatenate([p_pages, np.full(bucket - npg, p_pages[-1], np.int32)])
        out = []
        for k, (gather_jit, _, d_stack_shard, p_sub, _) in enumerate(self._jits):
            pp = p_pages if k == 0 else self._swa_pages(p_pages, self.p_mapping)
            p_idx = jax.device_put(pp, self._p_idx_shard)
            with jax.set_mesh(self.p_mesh):
                p_stacked = gather_jit(tuple(p_sub.kv_buffer), p_idx)
            out.append(jax.device_put(p_stacked, d_stack_shard))
        return (out[0] if len(out) == 1 else tuple(out), bucket)

    def scatter_from_dmesh(self, d_pages: np.ndarray, d_stacked, bucket: int) -> None:
        """scatter d_stacked into D pool. Called on the main thread so the
        donated kv_buffer reassignment is ordered before the next decode
        forward dispatch (data-dependency guarantees device-side ordering)."""
        d_pages = np.asarray(d_pages, np.int32)
        npg = len(d_pages)
        if bucket > npg:
            d_pages = np.concatenate([d_pages, np.full(bucket - npg, d_pages[-1], np.int32)])
        stacked = d_stacked if isinstance(d_stacked, tuple) else (d_stacked,)
        with self.d_pool._donate_lock, jax.set_mesh(self.d_mesh):
            for k, (_, scatter_jit, _, _, d_sub) in enumerate(self._jits):
                dp = d_pages if k == 0 else self._swa_pages(d_pages, self.d_mapping)
                d_idx = jax.device_put(dp, self._d_idx_shard)
                new_bufs = scatter_jit(tuple(d_sub.kv_buffer), d_idx, stacked[k])
                d_sub.kv_buffer = list(new_bufs)

    def transfer(self, p_page_indices: np.ndarray, d_page_indices: np.ndarray) -> None:
        p_page_indices = np.asarray(p_page_indices, np.int32)
        d_page_indices = np.asarray(d_page_indices, np.int32)
        n = len(p_page_indices)
        assert n == len(d_page_indices)
        for i in range(0, n, _NPG_BATCH_MAX):
            p = p_page_indices[i : i + _NPG_BATCH_MAX]
            d = d_page_indices[i : i + _NPG_BATCH_MAX]
            if len(p) == 0:
                continue
            d_stacked, bucket = self.gather_to_dmesh(p)
            self.scatter_from_dmesh(d, d_stacked, bucket)
