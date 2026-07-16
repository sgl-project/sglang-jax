"""Pathways single-controller cross-slice P/D disaggregation.

Architecture: one Python process (Pathways head) sees all devices across
N slices via IFRT proxy. Build prefill_mesh on slice 0, decode_mesh on
slice 1, run two ModelRunners. KV transfer = jax.device_put across meshes
(IFRT runtime handles per-host DCN fan-out, ~14 GB/s/host measured vs ~4.5
for jax.experimental.transfer on the same hardware).
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

__all__ = [
    "group_by_slice",
    "group_pages_by_dp_rank",
    "make_slice_meshes",
    "PathwaysPDKVTransfer",
    "migrate_reqs_p_to_d",
    "slots_to_ordered_pages",
]

logger = logging.getLogger(__name__)
_PD_DBG = bool(os.environ.get("SGLANG_PD_DBG"))

_NPG_BUCKETS = (8, 16, 32, 64, 128, 256)
# Upper bound on pages per stacked transfer. 256pg x 78L x 80KB ~= 1.6 GB
# transient on each side; mf=0.84 on 96GB devices leaves >8 GB headroom.
# 16K/ps256 = 64pg fits one shot, 16K/ps128 = 128pg also one shot.
_NPG_BATCH_MAX = 256


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


def group_pages_by_dp_rank(
    pages_by_req: list[tuple[int, np.ndarray]], dp_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Group per-req (dp_rank, local_pages) pairs into the [dp_size, max_n]
    layout the dp>1 shard_map kernels (_make_pool_jits' gather_jit /
    scatter_one_jit) require -- each row must hold ONLY that rank's own
    LOCAL page indices (allocator.py: indices aren't comparable across
    ranks), never a flat cross-rank concat.

    Returns (pages, valid) both shaped [dp_size, max_n]: `pages[i, j]` for
    j >= real-count is padded by repeating rank i's own last real page
    (harmless self-overwrite/re-gather, same scheme _bucket_npg already
    uses within a single rank). `valid[i, j]` is False for those padding
    slots -- callers MUST check it before trusting/using the corresponding
    output.

    KNOWN LIMITATION (not yet handled by any caller): a rank with ZERO real
    pages this round has no "own last page" to repeat, so its row is left
    as all-zero-index/all-invalid. Naively feeding that row into
    scatter_one_jit would overwrite page 0 of THAT rank's D pool with
    whatever stale P-side data rides along -- a real corruption, not a
    harmless no-op (unlike the same-rank padding case). Callers must NOT
    invoke the scatter kernel across a batch where any rank's row is
    all-invalid; either defer that rank's migration to a round where it
    has >=1 real req, or split into per-rank-populated sub-batches.
    """
    per_rank: list[list[int]] = [[] for _ in range(dp_size)]
    for dp_rank, pages in pages_by_req:
        per_rank[dp_rank].extend(int(p) for p in np.asarray(pages, np.int64))
    max_n = max((len(p) for p in per_rank), default=0)
    if max_n == 0:
        return np.zeros((dp_size, 0), np.int32), np.zeros((dp_size, 0), bool)
    pages_out = np.zeros((dp_size, max_n), np.int32)
    valid_out = np.zeros((dp_size, max_n), bool)
    for i, p in enumerate(per_rank):
        if not p:
            continue  # no real pages for this rank -- see KNOWN LIMITATION above
        arr = np.asarray(p, np.int32)
        valid_out[i, : len(arr)] = True
        if len(arr) < max_n:
            arr = np.concatenate([arr, np.full(max_n - len(arr), arr[-1], np.int32)])
        pages_out[i] = arr
    return pages_out, valid_out


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
    dp_size: int,
    tp_per_side: int,
    n_prefill: int = 1,
    n_decode: int = 1,
    tp_prefill: int = 0,
) -> tuple[list[Mesh], list[Mesh]]:
    """Build ([prefill_mesh_0..n_prefill-1], [decode_mesh_0..n_decode-1]) from
    the first n_prefill+n_decode slices.

    Multi-P (n_prefill>1) fans prefill across slices to lift R_prefill (the
    bottleneck on P-bound workloads, short OL). Multi-D (n_decode>1) fans
    decode across slices for D-bound workloads (long OL, e.g. 4K/16K where
    R_p/R_d ~= 26x on MiMo-V2 tp16 -- one P slice can feed >>1 D slice).

    Hetero-TP (Stage 6 P-D-different-tp): tp_prefill!=0 builds each P mesh
    with tp_prefill devices instead of tp_per_side (D always tp_per_side).
    Requires the underlying PathwaysJob to expose slices of BOTH sizes
    (workers[] is a list -- e.g. one 2x2x4 entry for P + one 2x2x2 for D);
    slices are then role-assigned by device count, not by slice_index order,
    so tp_p<tp_d and tp_p>tp_d both work regardless of which topology the
    RM enumerates first. dp_size is shared (attention_tp differs per side).
    """
    tp_d = tp_per_side
    tp_p = tp_prefill or tp_per_side
    devs = jax.devices()
    groups = group_by_slice(devs)
    sids = sorted(groups)
    need = n_prefill + n_decode
    if len(sids) < need:
        raise RuntimeError(
            f"pathways_pd n_prefill={n_prefill} n_decode={n_decode} requires >={need} "
            f"slices, got {len(sids)} (devices={len(devs)}, backend={jax.default_backend()})"
        )
    if tp_p != tp_d:
        # Assign by slice size: n_prefill slices of size tp_p go to P,
        # n_decode of size tp_d go to D. Fail loud if the PathwaysJob
        # workers[] topologies don't match -- don't silently truncate a
        # bigger slice (wastes chips + adds an untested code path).
        p_sids = [s for s in sids if len(groups[s]) == tp_p][:n_prefill]
        d_sids = [s for s in sids if len(groups[s]) == tp_d and s not in p_sids][:n_decode]
        if len(p_sids) < n_prefill or len(d_sids) < n_decode:
            raise RuntimeError(
                f"[pathways_pd] hetero-tp needs {n_prefill} slice(s) of size {tp_p} (P) "
                f"+ {n_decode} of size {tp_d} (D); got sizes={[len(groups[s]) for s in sids]}"
            )
    else:
        p_sids = sids[:n_prefill]
        d_sids = sids[n_prefill : n_prefill + n_decode]
    et = (jax.sharding.AxisType.Explicit,) * 2

    def _mk(ds: list, tp: int) -> Mesh:
        if len(ds) != tp:
            raise RuntimeError(f"slice size {len(ds)} != tp {tp}")
        return Mesh(
            np.asarray(ds).reshape(dp_size, tp // dp_size),
            axis_names=("data", "tensor"),
            axis_types=et,
        )

    p_meshes = [_mk(groups[s], tp_p) for s in p_sids]
    d_meshes = [_mk(groups[s], tp_d) for s in d_sids]
    logger.info(
        "[pathways_pd] slices=%d n_prefill=%d n_decode=%d tp_p=%d tp_d=%d "
        "p_slices=%s d_slices=%s p_shape=%s d_shape=%s",
        len(sids),
        n_prefill,
        n_decode,
        tp_p,
        tp_d,
        p_sids,
        d_sids,
        p_meshes[0].shape,
        d_meshes[0].shape,
    )
    return p_meshes, d_meshes


def _make_pool_jits(p_mesh: Mesh, d_mesh: Mesh, p_sub, d_sub):
    """Build (gather_jit, scatter_one_jit, d_stack_shard) for one MHA-style sub-pool.

    scatter is per-layer: one jit call per kv_buffer entry. Batching all
    layers into a single jit made the XLA program's temp reservation scale
    with the whole D pool (n_layers x pool_bytes), which OOMs the bottom-
    of-HBM region once the pool is large (V2-Flash tp16 DPOOL=1310720 ->
    32G reserve vs 15G free). Per-layer keeps the reservation at O(1 layer).

    dp>1: kv_buffer's leading (token/page) axis is sharded by the mesh's
    'data' axis, and page/token indices from the allocator are PER-RANK
    LOCAL (allocator.py: "Each rank has independent [1, size_per_rank]
    indices"). A plain jax.jit `.at[idx].get/set()` with a replicated idx
    array would misinterpret rank>0's local index as a GLOBAL index (landing
    on the wrong rank's physical data) -- this is exactly the pattern the
    base (non-PD) KV-cache write path avoids by running under `jax.shard_map`
    (see memory_pool.py update_fused_kv_cache_vectorized). gather_jit/
    scatter_one_jit below do the same: `idx`/`page` carry an explicit
    dp_size-sized leading axis sharded on 'data', so each rank's shard_map
    instance only ever sees and applies its OWN local indices.

    dp_size==1 fast path: shard_map's manual-sharding lowering is a
    functional no-op here but was NOT compile/dispatch-free on real TPU --
    a live 78-layer model showed gather (all layers in one XLA program,
    unlike scatter's per-layer calls) climbing 18s->37s and NOT plateauing
    across repeat calls, vs <1s before shard_map existed. So skip shard_map
    entirely when dp==1 and reuse the plain-jit body -- callers still always
    pass [dp_size, n]-shaped idx/page (this fn just unwraps dim 0).
    """
    kv_spec = d_sub.kv_sharding.spec
    dp_axis = kv_spec[0]  # mesh axis name the pool's token dim is sharded on
    tail_spec = kv_spec[1:]
    # gather returns a tuple of L per-layer arrays (NOT jnp.stack): stacking
    # forced scatter to index the device Array with un-jitted __getitem__
    # (~11ms Execute each x 48 layers).
    stack_spec = P(None, *tail_spec)

    if p_mesh.shape[dp_axis] == 1 and d_mesh.shape[dp_axis] == 1:
        page_spec = P(None, *tail_spec)
        gather_jit = jax.jit(
            lambda bufs, idx: tuple(b.at[idx[0]].get(out_sharding=page_spec) for b in bufs)
        )
        scatter_one_jit = jax.jit(
            lambda buf, idx, page: buf.at[idx[0]].set(page[0], out_sharding=kv_spec),
            donate_argnums=(0,),
        )
        return gather_jit, scatter_one_jit, NamedSharding(d_mesh, stack_spec)

    idx_spec = P(dp_axis, None)
    page_batch_spec = P(dp_axis, None, *tail_spec)

    # dp>1 gather: ALL layers in ONE jit (tuple bufs -> stacked output).
    # Per-layer (48 separate gather_one_jit Executes) head-of-line-blocked
    # D forward dispatch through Pathways' single ordered dispatch queue --
    # measured 2x ~650ms burst per done (gather 48 + scatter 48), costing
    # ~16% throughput. Precompile covers the first-compile cost of the
    # all-layers form (previously 18-37s trace/lower on 78L models).
    _gather_body = jax.shard_map(
        lambda buf, idx: buf.at[idx[0]].get()[None, ...],
        mesh=p_mesh,
        in_specs=(kv_spec, idx_spec),
        out_specs=P(dp_axis, None, *tail_spec),
        check_vma=False,
    )
    gather_jit = jax.jit(lambda bufs, idx: tuple(_gather_body(b, idx) for b in bufs))

    # scatter: donate reserves the full D-pool slice per layer simultaneously
    # (not sequentially) inside one jit -- all-39L=16.6G, 8L=14.7G both OOM'd
    # (~1.84G/layer, 9.9G free). Batch of 4 (~7.4G) fits; 48 per-layer
    # Executes -> ~13 (2 sub-pools x ceil(L/4)), cutting the dispatch-queue
    # burst from ~500ms to ~130ms.
    _scatter_body = jax.shard_map(
        lambda buf, idx, page: buf.at[idx[0]].set(page[0]),
        mesh=d_mesh,
        in_specs=(kv_spec, idx_spec, page_batch_spec),
        out_specs=kv_spec,
        check_vma=False,
    )
    # donate through shard_map only aliases in-place when the outer jit's
    # executable I/O has *explicitly identical* shardings on the donated input
    # and its output. Without in/out_shardings the SPMDShardToFullShape custom-
    # call at shard_map's exit is opaque to XLA buffer-assignment, so donation
    # degrades to free-then-copy of the whole D-pool layer (~3GB x _SB per
    # scatter, 30-150ms). Pinning both sides to d_sub.kv_sharding lets XLA set
    # input_output_alias -> the inner .at[].set() lowers to in-place scatter.
    d_kv_shard = d_sub.kv_sharding  # NamedSharding(d_mesh, kv_spec)
    d_idx_shard = NamedSharding(d_mesh, idx_spec)
    d_page_shard = NamedSharding(d_mesh, page_batch_spec)
    scatter_one_jit = jax.jit(
        _scatter_body,
        in_shardings=(d_kv_shard, d_idx_shard, d_page_shard),
        out_shardings=d_kv_shard,
        donate_argnums=(0,),
    )
    _SB = 4
    scatter_batch_jit = jax.jit(
        lambda bufs, idx, pages: tuple(_scatter_body(bufs[i], idx, pages[i]) for i in range(_SB)),
        in_shardings=((d_kv_shard,) * _SB, d_idx_shard, (d_page_shard,) * _SB),
        out_shardings=(d_kv_shard,) * _SB,
        donate_argnums=(0,),
    )

    def scatter_fn(bufs_list, idx, stacked):
        i, n = 0, len(bufs_list)
        while i + _SB <= n:
            new = scatter_batch_jit(
                tuple(bufs_list[i : i + _SB]), idx, tuple(stacked[i + j] for j in range(_SB))
            )
            for j in range(_SB):
                bufs_list[i + j] = new[j]
            i += _SB
        while i < n:
            bufs_list[i] = scatter_one_jit(bufs_list[i], idx, stacked[i])
            i += 1

    # dp>1 gather returns tuple of L [dp_size, n, *page_dims] arrays.
    return gather_jit, scatter_fn, NamedSharding(d_mesh, P(dp_axis, None, *tail_spec))


class PathwaysPDKVTransfer:
    """Paged KV pool P-mesh -> D-mesh via cross-slice device_put.

    gather(P pages) -> stack L layers -> device_put to d_mesh sharding
    -> scatter into D pool (donate). Single transfer per batch maximizes
    payload (measured 28.9 GB/s aggregate at 2048 MiB on 2x 8-device slices).

    SWAKVPool: full + swa sub-pools transferred independently. Input/output
    pages are in the *full* index space; swa pages are derived per side via
    each allocator's full_to_swa_index_mapping (page-aligned 1:1 because PD
    forces disable_radix + no chunked prefill, so every alloc is page-head).
    """

    def __init__(self, p_mesh: Mesh, d_mesh: Mesh, p_pool, d_pool, **kw) -> None:
        self.p_mesh = p_mesh
        self.d_mesh = d_mesh
        self.d_pool = d_pool
        # gather_jit/scatter_one_jit run per-dp-rank via shard_map (see
        # _make_pool_jits); the calling convention below still only feeds
        # them a single rank's worth of pages (dp_size==1 wrapped as
        # shape [1, n]) since the scheduler doesn't yet group migrated
        # pages by req.dp_rank -- that's the remaining piece for dp>1.
        self.p_dp_size = p_mesh.shape["data"]
        self.d_dp_size = d_mesh.shape["data"]
        self.is_swa = hasattr(p_pool, "swa_kv_pool")
        if self.is_swa:
            self.page_size = kw["page_size"]
            self.p_mapping = kw["p_alloc"].full_to_swa_index_mapping
            self.d_mapping = kw["d_alloc"].full_to_swa_index_mapping
            # swa layers only need the last sliding_window tokens on D. Gather
            # was moving all seq_len pages (16K/256=64) x 39 swa layers ~=
            # 2.6GB/done cross-slice; tail-only cuts that to ~40MB. D-side swa
            # slots outside the tail stay garbage (decode swa attention never
            # reads past the window) and get free_swa'd in _pd_drain_ready.
            self._swa_tail_pages = kw.get("swa_tail_pages", 0)
            sub_pools = [
                (p_pool.full_kv_pool, d_pool.full_kv_pool),
                (p_pool.swa_kv_pool, d_pool.swa_kv_pool),
            ]
        else:
            self._swa_tail_pages = 0
            sub_pools = [(p_pool, d_pool)]
        # Hetero-TP guard: gather/scatter move whole pages (global shape),
        # so P/D per-page shape must match. MLA pools are tensor-replicated
        # (P("data",None,None,None)) -> always match. MHA pools shard on
        # kv_heads: match iff kv_heads >= max(attention_tp_p, attention_tp_d)
        # (else get_total_num_kv_heads_with_replication pads to attention_tp
        # and the two sides' head axes differ). Fail here, not mid-transfer.
        for p_sub, d_sub in sub_pools:
            ps, ds = p_sub.kv_buffer[0].shape[1:], d_sub.kv_buffer[0].shape[1:]
            if ps != ds:
                raise RuntimeError(
                    f"[pathways_pd] hetero-tp KV per-page shape mismatch P={ps} D={ds}; "
                    f"model kv_heads must be >= max(attention_tp_p, attention_tp_d) "
                    f"for MHA/GQA, or use an MLA model"
                )
        # _jits[k] = (gather, scatter, d_stack_shard, p_sub_pool, d_sub_pool)
        self._jits = [
            (*_make_pool_jits(p_mesh, d_mesh, p_sub, d_sub), p_sub, d_sub)
            for p_sub, d_sub in sub_pools
        ]
        self._p_idx_shard = NamedSharding(p_mesh, P("data", None))
        self._d_idx_shard = NamedSharding(d_mesh, P("data", None))
        # Hetero-tp: device_put(tuple-of-L-layers, d_shard) dispatches L
        # separate ReshardOps through Pathways' dispatch queue (~4.5ms each on
        # 32->16); GLM L=78 -> ~350ms/gather vs same-shape homo path ~5ms.
        # Stack to one [L, ...] array on p_mesh (1 XLA prog), device_put once
        # (1 ReshardOp), unstack on d_mesh (1 XLA prog). Homo path unchanged.
        self._is_hetero = p_mesh.size != d_mesh.size
        if self._is_hetero:
            self._hetero_jits = []
            for k, (_, _, d_stack_shard, p_sub, _) in enumerate(self._jits):
                L = p_sub.layer_num
                big_spec = P(None, *d_stack_shard.spec)
                d_big_shard = NamedSharding(d_mesh, big_spec)
                stack_j = jax.jit(lambda t: jax.numpy.stack(t, axis=0))
                unstack_j = jax.jit(
                    lambda x, _L=L: tuple(x[i] for i in range(_L)),
                    out_shardings=d_stack_shard,
                )
                self._hetero_jits.append((stack_j, unstack_j, d_big_shard))
        logger.info(
            "[pathways_pd] kv_transfer is_swa=%s sub_pools=%d layers=%s",
            self.is_swa,
            len(sub_pools),
            [p.layer_num for p, _ in sub_pools],
        )

    def precompile(self) -> None:
        """Warm up gather/[stack/reshard/unstack]/scatter for all bucket shapes so
        the first real request doesn't pay the ~minutes-long trace/lower cost
        (dominant on hetero-tp where stack_j/unstack_j span L layers)."""
        t0 = time.perf_counter()
        # page 0 is the allocator sentinel (never handed out), safe to read/write.
        for n in _NPG_BUCKETS:
            _tb = time.perf_counter()
            p_pages = np.zeros((self.p_dp_size, n), np.int32)
            d_pages = np.zeros((self.d_dp_size, n), np.int32)
            d_stacked, bucket = self.gather_to_dmesh(p_pages)
            self.scatter_from_dmesh(d_pages, d_stacked, bucket)
            for sk in d_stacked:
                jax.block_until_ready(sk[-1] if isinstance(sk, tuple) else sk)
            logger.info(
                "[pathways_pd] precompile bucket=%d done in %.1fs", n, time.perf_counter() - _tb
            )
        logger.info(
            "[pathways_pd] KV transfer precompile done in %.1fs (hetero=%s)",
            time.perf_counter() - t0,
            self._is_hetero,
        )

    def _swa_pages(self, full_pages: np.ndarray, mapping, dp_rank: int = 0) -> np.ndarray:
        # `mapping` is a list of per-dp-rank arrays when dp_size>1 (each rank's
        # full_to_swa_index_mapping uses that rank's own local index space --
        # see allocator.py: "Each rank has independent [1, size_per_rank]
        # indices"), or a single array when dp_size==1. `full_pages` must all
        # belong to the SAME dp_rank (callers with a multi-rank batch must
        # split by rank before calling, since indices aren't comparable
        # across ranks) -- not yet done by any caller (dp=1 only today).
        m = mapping[dp_rank] if isinstance(mapping, list) else mapping
        return (m[full_pages.astype(np.int64) * self.page_size] // self.page_size).astype(np.int32)

    def _prep_idx(self, pages: np.ndarray, dp_size: int, side: str) -> tuple[np.ndarray, int, int]:
        """Normalize pages to [dp_size, bucket] with per-rank last-page padding.

        dp_size==1: pages is a flat [n] array (old callers) -> wrap to [1, n].
        dp_size>1: pages MUST already be [dp_size, n_per_rank] (caller used
        group_pages_by_dp_rank); every rank must have >=1 real page (all-zero
        rows corrupt that rank's page 0 -- see group_pages_by_dp_rank
        docstring's KNOWN LIMITATION). We assert on that here so it fails
        loudly instead of silently corrupting data.
        """
        pages = np.asarray(pages, np.int32)
        if dp_size == 1:
            pages = pages[None, :] if pages.ndim == 1 else pages
        assert pages.ndim == 2 and pages.shape[0] == dp_size, (
            f"[pathways_pd] {side} pages shape {pages.shape} vs dp_size={dp_size}: "
            f"dp>1 callers must pre-group via group_pages_by_dp_rank"
        )
        n = pages.shape[1]
        # Empty-rank rows (all-zero page indices) are harmless: allocator.py
        # never hands out index 0 (arange(1, size+1)), so page 0 is a reserved
        # sentinel on both sides -- gather reads it, scatter overwrites it,
        # nothing that any req's slots point at is touched.
        bucket = _bucket_npg(max(n, 1))
        if bucket > n:
            pad = np.repeat(pages[:, -1:], bucket - n, axis=1)
            pages = np.concatenate([pages, pad], axis=1)
        return pages, bucket, n

    def gather_to_dmesh(
        self, p_pages: np.ndarray, p_swa_pages=None
    ) -> tuple[jax.Array | tuple, int]:
        """gather P pool pages -> stack -> device_put to D mesh.

        Runs in the prefill thread; the returned d_stacked is a future on the
        D mesh — main-thread scatter consumes it via data-dep so the cross-slice
        RecvRefs interleaves with decode instead of stalling here for ~250ms.
        """
        pp2, bucket, n_real = self._prep_idx(p_pages, self.p_dp_size, "P")
        _no_tail = os.environ.get("SGLANG_PD_NO_SWA_TAIL")
        out = []
        for k, (gather_jit, _, d_stack_shard, p_sub, _) in enumerate(self._jits):
            if k == 0:
                pk = pp2
            elif _no_tail:
                pk = np.stack(
                    [self._swa_pages(pp2[i], self.p_mapping, i) for i in range(self.p_dp_size)]
                )
            elif p_swa_pages is not None:
                pt, _, _ = self._prep_idx(p_swa_pages, self.p_dp_size, "P")
                pk = np.stack(
                    [self._swa_pages(pt[i], self.p_mapping, i) for i in range(self.p_dp_size)]
                )
            else:
                tail = self._swa_tail_pages
                # slice tail from the REAL pages (before bucket padding), not
                # from padded pp2 -- otherwise when bucket-n_real >= tail the
                # tail is all-padding (last real page repeated) and swa gather
                # misses page[n_real-tail:n_real-1].
                _lo = max(0, n_real - tail) if tail else 0
                pt = pp2[:, _lo:n_real] if tail and tail < n_real else pp2[:, :n_real]
                pk = np.stack(
                    [self._swa_pages(pt[i], self.p_mapping, i) for i in range(self.p_dp_size)]
                )
                if os.environ.get("SGLANG_PD_DBG_KV"):
                    logger.info(
                        "[pd-swa-dbg gather] slice=%s n_real=%d bucket=%d tail=%d "
                        "pt=%s swa_pk=%s",
                        self.p_mesh.devices.flat[0].slice_index,
                        n_real,
                        bucket,
                        tail,
                        pt.tolist(),
                        pk.tolist(),
                    )
            p_idx = jax.device_put(pk, self._p_idx_shard)
            with jax.set_mesh(self.p_mesh):
                p_stacked = gather_jit(tuple(p_sub.kv_buffer), p_idx)
            if self._is_hetero:
                stack_j, unstack_j, d_big_shard = self._hetero_jits[k]
                with jax.set_mesh(self.p_mesh):
                    p_big = stack_j(p_stacked)
                d_big = jax.device_put(p_big, d_big_shard)
                with jax.set_mesh(self.d_mesh):
                    _d = unstack_j(d_big)
            else:
                _d = jax.device_put(p_stacked, d_stack_shard)
            if os.environ.get("SGLANG_PD_DBG_KV") and k == 0:
                _sp = np.asarray(
                    jax.device_get(p_stacked[0] if isinstance(p_stacked, tuple) else p_stacked)
                ).astype(np.float32)
                _sd = np.asarray(jax.device_get(_d[0] if isinstance(_d, tuple) else _d)).astype(
                    np.float32
                )
                logger.info(
                    "[pd-gather-dbg] slice=%s pk0=%s pk_last=%s p_sum=%.1f d_sum=%.1f "
                    "p_last=%.1f d_last=%.1f MATCH=%s",
                    self.p_mesh.devices.flat[0].slice_index,
                    list(pk[0, :3]),
                    list(pk[-1, :3]),
                    float(_sp.sum()),
                    float(_sd.sum()),
                    float(_sp[-1].sum()),
                    float(_sd[-1].sum()),
                    bool(np.allclose(_sp, _sd)),
                )
            out.append(_d)
        # Always tuple(out) even for single sub-pool: scatter_from_dmesh's
        # `stacked[k]` indexes by sub-pool k, so out[0] (a per-layer tuple)
        # unwrapped here would make stacked[0] = layer-0 not sub-pool-0.
        return (tuple(out), bucket)

    def scatter_from_dmesh(
        self, d_pages: np.ndarray, d_stacked, bucket: int, d_swa_pages=None
    ) -> None:
        """scatter d_stacked into D pool. Called on the main thread so the
        donated kv_buffer reassignment is ordered before the next decode
        forward dispatch (data-dependency guarantees device-side ordering)."""
        dp2, _, d_n_real = self._prep_idx(d_pages, self.d_dp_size, "D")
        # bucket comes from gather (P side) and MUST match dp2.shape[1] since
        # both sides use the same n_per_rank -> _bucket_npg. If d_pages n
        # differs from p_pages n (shouldn't: 1:1 page mapping) trust bucket.
        assert dp2.shape[1] == bucket, f"D pages n={dp2.shape[1]} != gather bucket={bucket}"
        stacked = d_stacked if isinstance(d_stacked, tuple) else (d_stacked,)
        # dp=1 fast-path gather output has NO leading dp dim ([L, n, ...]);
        # dp>1 shard_map gather output has it ([L, dp_size, n, ...]).
        wrap = self.d_dp_size == 1
        _t_acq = time.perf_counter()
        with self.d_pool._donate_lock, jax.set_mesh(self.d_mesh):
            _t_in = time.perf_counter()
            for k, (_, scatter_fn, _, _, d_sub) in enumerate(self._jits):
                if k == 0:
                    dk = dp2
                elif os.environ.get("SGLANG_PD_NO_SWA_TAIL"):
                    dk = np.stack(
                        [self._swa_pages(dp2[i], self.d_mapping, i) for i in range(self.d_dp_size)]
                    )
                elif d_swa_pages is not None:
                    dt, _, _ = self._prep_idx(d_swa_pages, self.d_dp_size, "D")
                    dk = np.stack(
                        [self._swa_pages(dt[i], self.d_mapping, i) for i in range(self.d_dp_size)]
                    )
                else:
                    tail = self._swa_tail_pages
                    _lo = max(0, d_n_real - tail) if tail else 0
                    dt = dp2[:, _lo:d_n_real] if tail and tail < d_n_real else dp2[:, :d_n_real]
                    dk = np.stack(
                        [self._swa_pages(dt[i], self.d_mapping, i) for i in range(self.d_dp_size)]
                    )
                    if os.environ.get("SGLANG_PD_DBG_KV"):
                        _sk = stacked[k]
                        _sk0 = _sk[0] if isinstance(_sk, tuple) else _sk
                        _sksum = float(np.asarray(jax.device_get(_sk0)).astype(np.float32).sum())
                        logger.info(
                            "[pd-swa-dbg scatter] d_n_real=%d dt=%s swa_dk=%s "
                            "stacked_k1_l0_sum=%.1f shape=%s",
                            d_n_real,
                            dt.tolist(),
                            dk.tolist(),
                            _sksum,
                            np.asarray(jax.device_get(_sk0)).shape,
                        )
                d_idx = jax.device_put(dk, self._d_idx_shard)
                if wrap:
                    for i in range(d_sub.layer_num):
                        d_sub.kv_buffer[i] = scatter_fn(
                            d_sub.kv_buffer[i], d_idx, stacked[k][i][None, ...]
                        )
                else:
                    scatter_fn(d_sub.kv_buffer, d_idx, stacked[k])
        if _PD_DBG:
            logger.info(
                "[pd-lock] scatter wait=%.2fms hold=%.2fms n_sub=%d",
                (_t_in - _t_acq) * 1e3,
                (time.perf_counter() - _t_in) * 1e3,
                len(self._jits),
            )

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
