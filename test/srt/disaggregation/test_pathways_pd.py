"""Pathways single-controller cross-slice PD unit tests.

Run: XLA_FLAGS=--xla_force_host_platform_device_count=8 JAX_PLATFORMS=cpu \
     pytest test/srt/disaggregation/test_pathways_pd.py -v
"""

from __future__ import annotations

import queue
import threading
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.disaggregation.pathways_pd import (
    PathwaysPDKVTransfer,
    group_by_slice,
    group_pages_by_dp_rank,
    make_slice_meshes,
    migrate_reqs_p_to_d,
    slots_to_ordered_pages,
)
from sgl_jax.srt.disaggregation.pathways_scheduler import PathwaysPDSchedulerMixin

NUM_PAGES = 16
PAGE_SIZE = 4
HEAD_DIM = 8
N_LAYERS = 3


@pytest.fixture(scope="module")
def meshes():
    devs = jax.devices()
    if len(devs) < 8:
        pytest.skip(f"need >=8 devices, got {len(devs)}")
    p_meshes, d_meshes = make_slice_meshes(dp_size=1, tp_per_side=len(devs) // 2)
    assert isinstance(p_meshes, list) and len(p_meshes) == 1
    assert isinstance(d_meshes, list) and len(d_meshes) == 1
    return p_meshes[0], d_meshes[0]


def _make_pool(mesh, fill: str):
    sh = NamedSharding(mesh, P("data", None, None, None))
    bufs = []
    for layer in range(N_LAYERS):
        if fill == "seq":
            host = (
                np.arange(NUM_PAGES * PAGE_SIZE * HEAD_DIM, dtype=np.float32).reshape(
                    NUM_PAGES, PAGE_SIZE, 1, HEAD_DIM
                )
                + layer * 1000
            ).astype(jnp.bfloat16)
        else:
            host = np.zeros((NUM_PAGES, PAGE_SIZE, 1, HEAD_DIM), jnp.bfloat16)
        bufs.append(jax.device_put(host, sh))
    [b.block_until_ready() for b in bufs]
    return SimpleNamespace(
        kv_buffer=bufs, layer_num=N_LAYERS, kv_sharding=sh, _donate_lock=threading.Lock()
    )


@pytest.mark.unit
def test_group_by_slice_fallback():
    devs = jax.devices()
    groups = group_by_slice(devs)
    assert len(groups) == 2
    assert sum(len(g) for g in groups.values()) == len(devs)
    all_ids = {d.id for g in groups.values() for d in g}
    assert len(all_ids) == len(devs)


@pytest.mark.unit
def test_make_slice_meshes(meshes):
    p_mesh, d_mesh = meshes
    assert p_mesh.devices.size == d_mesh.devices.size
    assert p_mesh.axis_names == ("data", "tensor")
    p_ids = {d.id for d in p_mesh.devices.flatten()}
    d_ids = {d.id for d in d_mesh.devices.flatten()}
    assert p_ids.isdisjoint(d_ids)


@pytest.mark.unit
def test_make_slice_meshes_hetero_tp():
    """tp_prefill != tp_per_side: P/D meshes have different tensor-axis size,
    slices assigned by device count. group_by_slice fallback splits 8 CPU
    devs into 2 groups of 4, so tp_p=tp_d=4 is the only combo we can test on
    CPU without slice_index; the by-size assignment is unit-tested via a
    monkeypatched group_by_slice returning unequal groups."""
    devs = jax.devices()
    if len(devs) < 6:
        pytest.skip("need >=6 devices")
    import sgl_jax.srt.disaggregation.pathways_pd as pd

    orig = pd.group_by_slice
    # fake: slice 0 = 4 dev (P tp4), slice 1 = 2 dev (D tp2)
    pd.group_by_slice = lambda ds: {0: list(ds)[:4], 1: list(ds)[4:6]}
    try:
        p_meshes, d_meshes = make_slice_meshes(dp_size=1, tp_per_side=2, tp_prefill=4)
        d_mesh = d_meshes[0]
        assert p_meshes[0].devices.size == 4
        assert d_mesh.devices.size == 2
        assert p_meshes[0].shape["tensor"] == 4
        assert d_mesh.shape["tensor"] == 2
        p_ids = {d.id for d in p_meshes[0].devices.flatten()}
        d_ids = {d.id for d in d_mesh.devices.flatten()}
        assert p_ids.isdisjoint(d_ids)
        # reversed roles: tp_p=2 (small P) tp_d=4 (big D) also assigns by size
        p2, d2 = make_slice_meshes(dp_size=1, tp_per_side=4, tp_prefill=2)
        assert p2[0].devices.size == 2 and d2[0].devices.size == 4
        # mismatch: request tp_p=8 but no size-8 slice -> loud fail
        with pytest.raises(RuntimeError, match="hetero-tp needs"):
            make_slice_meshes(dp_size=1, tp_per_side=2, tp_prefill=8)
    finally:
        pd.group_by_slice = orig


@pytest.mark.unit
def test_transfer_byte_equal_hetero_tp():
    """MLA-style pool (tensor-replicated P("data",None,None,None)) transfers
    byte-equal across meshes of DIFFERENT tensor-axis size. This is the
    GLM-5.2 hetero-TP path: device_put reshard is replicated->replicated
    (change device set only), no shape change, no explicit reshape."""
    devs = jax.devices()
    if len(devs) < 6:
        pytest.skip("need >=6 devices")
    import sgl_jax.srt.disaggregation.pathways_pd as pd

    orig = pd.group_by_slice
    pd.group_by_slice = lambda ds: {0: list(ds)[:4], 1: list(ds)[4:6]}
    try:
        p_meshes, d_meshes = make_slice_meshes(dp_size=1, tp_per_side=2, tp_prefill=4)
    finally:
        pd.group_by_slice = orig
    p_mesh, d_mesh = p_meshes[0], d_meshes[0]
    p_pool = _make_pool(p_mesh, fill="seq")
    d_pool = _make_pool(d_mesh, fill="zero")
    xfer = PathwaysPDKVTransfer(p_mesh, d_mesh, p_pool, d_pool)
    p_pages = np.array([1, 3, 5], np.int32)
    d_pages = np.array([0, 2, 4], np.int32)
    xfer.transfer(p_pages, d_pages)
    for layer in range(N_LAYERS):
        d_host = np.asarray(d_pool.kv_buffer[layer])
        p_host = np.asarray(p_pool.kv_buffer[layer])
        np.testing.assert_array_equal(d_host[d_pages], p_host[p_pages])


@pytest.mark.unit
def test_transfer_byte_equal(meshes):
    p_mesh, d_mesh = meshes
    p_pool = _make_pool(p_mesh, fill="seq")
    d_pool = _make_pool(d_mesh, fill="zero")
    xfer = PathwaysPDKVTransfer(p_mesh, d_mesh, p_pool, d_pool)

    p_pages = np.array([1, 3, 5, 7], np.int32)
    d_pages = np.array([0, 2, 4, 6], np.int32)
    xfer.transfer(p_pages, d_pages)

    for layer in range(N_LAYERS):
        d_host = np.asarray(d_pool.kv_buffer[layer])
        p_host = np.asarray(p_pool.kv_buffer[layer])
        np.testing.assert_array_equal(d_host[d_pages], p_host[p_pages])
        untouched = np.array([1, 3, 5, 7, 8, 9], np.int32)
        np.testing.assert_array_equal(d_host[untouched], np.zeros_like(d_host[untouched]))


@pytest.mark.unit
def test_gather_scatter_jits_dp2_no_cross_rank_leakage():
    """dp>1 kernel-level check (PathwaysPDKVTransfer's public API still
    raises NotImplementedError for dp>1 pending scheduler-side per-req
    dp_rank page grouping -- this test calls _make_pool_jits' gather_jit/
    scatter_one_jit directly to prove the underlying shard_map kernel
    itself is already dp>1-correct: each rank's local index only ever
    touches that SAME rank's physical shard, never another rank's."""
    from sgl_jax.srt.disaggregation.pathways_pd import _make_pool_jits

    devs = jax.devices()
    if len(devs) < 8:
        pytest.skip(f"need >=8 devices, got {len(devs)}")
    dp_size, tp_per_side = 2, 4
    p_meshes, d_meshes = make_slice_meshes(dp_size=dp_size, tp_per_side=tp_per_side)
    p_mesh, d_mesh = p_meshes[0], d_meshes[0]

    p_pool = _make_pool(p_mesh, fill="seq")  # NUM_PAGES=16 total, 8 pages/rank
    d_pool = _make_pool(d_mesh, fill="zero")
    gather_jit, scatter_one_jit, d_stack_shard = _make_pool_jits(p_mesh, d_mesh, p_pool, d_pool)

    pages_per_rank = NUM_PAGES // dp_size  # 8
    # rank0 gathers its local pages [0,2]; rank1 gathers its local pages [1,3]
    # (deliberately different per rank, so cross-rank leakage would be caught)
    local_idx = np.array([[0, 2], [1, 3]], dtype=np.int32)
    idx_dev = jax.device_put(local_idx, NamedSharding(p_mesh, P("data", None)))

    stacked = gather_jit(tuple(p_pool.kv_buffer), idx_dev)
    stacked_host = np.asarray(jax.device_put(stacked, d_stack_shard))

    # gather output shape now [L, dp_size, n, page_size, 1, head_dim] (per-layer
    # shard_map keeps the leading dp dim explicit instead of concatenating).
    assert stacked_host.shape[:3] == (N_LAYERS, dp_size, 2)
    p_host_full = [np.asarray(p_pool.kv_buffer[layer]) for layer in range(N_LAYERS)]
    for layer in range(N_LAYERS):
        # rank0's local page k lives at global page k; rank1's local page k
        # lives at global page (pages_per_rank + k) -- per-rank contiguous shard.
        expected_r0 = p_host_full[layer][[0, 2]]
        expected_r1 = p_host_full[layer][[pages_per_rank + 1, pages_per_rank + 3]]
        np.testing.assert_array_equal(stacked_host[layer, 0], expected_r0)
        np.testing.assert_array_equal(stacked_host[layer, 1], expected_r1)

    # scatter: write the gathered payload into DIFFERENT D-side local pages
    # per rank, verify each rank's write lands only in its own shard.
    scatter_idx = np.array([[5], [6]], dtype=np.int32)
    scatter_idx_dev = jax.device_put(scatter_idx, NamedSharding(d_mesh, P("data", None)))
    for layer in range(N_LAYERS):
        # build payload on host (numpy) to avoid ad-hoc eager JAX indexing on
        # an Explicit-sharded array, then device_put with the right sharding.
        payload_host = np.stack(
            [stacked_host[layer, 0, 0:1], stacked_host[layer, 1, 0:1]], axis=0
        )  # [dp_size=2, n=1, page_size, 1, head_dim]
        payload = jax.device_put(
            payload_host, NamedSharding(d_mesh, P("data", None, None, None, None))
        )
        d_pool.kv_buffer[layer] = scatter_one_jit(d_pool.kv_buffer[layer], scatter_idx_dev, payload)

    d_host_full = [np.asarray(d_pool.kv_buffer[layer]) for layer in range(N_LAYERS)]
    for layer in range(N_LAYERS):
        # rank0 wrote to its local page 5 (global page 5); rank1 wrote to its
        # local page 6 (global page pages_per_rank+6).
        np.testing.assert_array_equal(d_host_full[layer][5], p_host_full[layer][0])
        np.testing.assert_array_equal(
            d_host_full[layer][pages_per_rank + 6], p_host_full[layer][pages_per_rank + 1]
        )
        # everything else stays zero -- no cross-rank / off-target writes
        untouched_mask = np.ones(NUM_PAGES, dtype=bool)
        untouched_mask[[5, pages_per_rank + 6]] = False
        np.testing.assert_array_equal(
            d_host_full[layer][untouched_mask], np.zeros_like(d_host_full[layer][untouched_mask])
        )


@pytest.mark.unit
def test_transfer_bucket_padding(meshes):
    p_mesh, d_mesh = meshes
    p_pool = _make_pool(p_mesh, fill="seq")
    d_pool = _make_pool(d_mesh, fill="zero")
    xfer = PathwaysPDKVTransfer(p_mesh, d_mesh, p_pool, d_pool)

    p_pages = np.array([2, 4, 6], np.int32)
    d_pages = np.array([1, 3, 5], np.int32)
    xfer.transfer(p_pages, d_pages)

    for layer in range(N_LAYERS):
        d_host = np.asarray(d_pool.kv_buffer[layer])
        p_host = np.asarray(p_pool.kv_buffer[layer])
        np.testing.assert_array_equal(d_host[d_pages], p_host[p_pages])


@pytest.mark.unit
def test_transfer_on_d_mesh(meshes):
    p_mesh, d_mesh = meshes
    p_pool = _make_pool(p_mesh, fill="seq")
    d_pool = _make_pool(d_mesh, fill="zero")
    xfer = PathwaysPDKVTransfer(p_mesh, d_mesh, p_pool, d_pool)
    xfer.transfer(np.array([0], np.int32), np.array([0], np.int32))

    d_devs = set(d_mesh.devices.flatten().tolist())
    for buf in d_pool.kv_buffer:
        assert set(buf.sharding.device_set) == d_devs


@pytest.mark.unit
def test_slots_to_ordered_pages():
    slots = np.array([8, 9, 10, 11, 4, 5, 6, 7, 20, 21], np.int32)
    np.testing.assert_array_equal(slots_to_ordered_pages(slots, 4), [2, 1, 5])
    # mid-page resume (chunked req): [6,7] then [8,9,10] -> pages [1,2]
    np.testing.assert_array_equal(
        slots_to_ordered_pages(np.array([6, 7, 8, 9, 10], np.int32), 4), [1, 2]
    )


def test_group_pages_by_dp_rank_balanced():
    # rank0 req has 2 pages, rank1 req has 3 pages -> max_n=3, rank0 padded
    # by repeating its own last page (7), never rank1's data.
    pages_by_req = [(0, np.array([5, 7], np.int32)), (1, np.array([1, 2, 9], np.int32))]
    pages, valid = group_pages_by_dp_rank(pages_by_req, dp_size=2)
    np.testing.assert_array_equal(pages[0], [5, 7, 7])
    np.testing.assert_array_equal(valid[0], [True, True, False])
    np.testing.assert_array_equal(pages[1], [1, 2, 9])
    np.testing.assert_array_equal(valid[1], [True, True, True])


def test_group_pages_by_dp_rank_multi_req_same_rank():
    # two reqs on the same rank get concatenated in order, not interleaved.
    pages_by_req = [
        (0, np.array([3], np.int32)),
        (1, np.array([4, 5], np.int32)),
        (0, np.array([6, 8], np.int32)),
    ]
    pages, valid = group_pages_by_dp_rank(pages_by_req, dp_size=2)
    np.testing.assert_array_equal(pages[0], [3, 6, 8])
    np.testing.assert_array_equal(valid[0], [True, True, True])
    np.testing.assert_array_equal(pages[1], [4, 5, 5])
    np.testing.assert_array_equal(valid[1], [True, True, False])


def test_group_pages_by_dp_rank_empty_rank_marked_invalid():
    # KNOWN LIMITATION case: rank1 has zero reqs this round. Its row must
    # come back all-invalid (not silently defaulted to a "safe" real page --
    # there isn't one), so a caller that forgets to check `valid` can't
    # mistake page-index 0 for a real transfer.
    pages_by_req = [(0, np.array([2, 4], np.int32))]
    pages, valid = group_pages_by_dp_rank(pages_by_req, dp_size=2)
    np.testing.assert_array_equal(pages[0], [2, 4])
    np.testing.assert_array_equal(valid[0], [True, True])
    np.testing.assert_array_equal(valid[1], [False, False])


def test_group_pages_by_dp_rank_no_reqs():
    pages, valid = group_pages_by_dp_rank([], dp_size=2)
    assert pages.shape == (2, 0)
    assert valid.shape == (2, 0)


def _make_swa_pool(mesh, fill: str, n_swa_layers: int = 2):
    full = _make_pool(mesh, fill)
    swa = SimpleNamespace(
        kv_buffer=[_make_pool(mesh, fill).kv_buffer[i] for i in range(n_swa_layers)],
        layer_num=n_swa_layers,
        kv_sharding=full.kv_sharding,
    )
    return SimpleNamespace(full_kv_pool=full, swa_kv_pool=swa, _donate_lock=threading.Lock())


def _identity_swa_alloc():
    # full slot i <-> swa slot i (page-aligned 1:1, the PD invariant)
    return SimpleNamespace(
        full_to_swa_index_mapping=np.arange(NUM_PAGES * PAGE_SIZE, dtype=np.int64)
    )


@pytest.mark.unit
def test_swa_dual_pool_transfer(meshes):
    """SWAKVPool path: both full and swa sub-pools are gathered/scattered,
    and swa pages are derived via full_to_swa_index_mapping on each side."""
    p_mesh, d_mesh = meshes
    p_pool = _make_swa_pool(p_mesh, "seq")
    d_pool = _make_swa_pool(d_mesh, "zero")
    xfer = PathwaysPDKVTransfer(
        p_mesh,
        d_mesh,
        p_pool,
        d_pool,
        page_size=PAGE_SIZE,
        p_alloc=_identity_swa_alloc(),
        d_alloc=_identity_swa_alloc(),
    )
    assert xfer.is_swa and len(xfer._jits) == 2

    p_pages = np.array([1, 3, 5], np.int32)
    d_pages = np.array([0, 2, 4], np.int32)
    xfer.transfer(p_pages, d_pages)

    for layer in range(N_LAYERS):
        d_host = np.asarray(d_pool.full_kv_pool.kv_buffer[layer])
        p_host = np.asarray(p_pool.full_kv_pool.kv_buffer[layer])
        np.testing.assert_array_equal(d_host[d_pages], p_host[p_pages])
    for layer in range(2):
        d_host = np.asarray(d_pool.swa_kv_pool.kv_buffer[layer])
        p_host = np.asarray(p_pool.swa_kv_pool.kv_buffer[layer])
        np.testing.assert_array_equal(d_host[d_pages], p_host[p_pages])
        # swa pages outside d_pages stay zero (no over-write from bucket pad)
        np.testing.assert_array_equal(d_host[[1, 3, 6, 7]], 0)


@pytest.mark.unit
def test_swa_page_mapping(meshes):
    """_swa_pages must remap full-space pages through each side's mapping."""
    p_mesh, d_mesh = meshes
    p_pool = _make_swa_pool(p_mesh, "seq")
    d_pool = _make_swa_pool(d_mesh, "zero")
    # P side: full page k -> swa page (k+2) mod NUM_PAGES (page-aligned shift)
    p_map = np.arange(NUM_PAGES * PAGE_SIZE, dtype=np.int64)
    p_map = ((p_map // PAGE_SIZE + 2) % NUM_PAGES) * PAGE_SIZE + p_map % PAGE_SIZE
    p_alloc = SimpleNamespace(full_to_swa_index_mapping=p_map)
    xfer = PathwaysPDKVTransfer(
        p_mesh,
        d_mesh,
        p_pool,
        d_pool,
        page_size=PAGE_SIZE,
        p_alloc=p_alloc,
        d_alloc=_identity_swa_alloc(),
    )
    full_pages = np.array([0, 1, 5], np.int32)
    np.testing.assert_array_equal(xfer._swa_pages(full_pages, xfer.p_mapping), [2, 3, 7])
    np.testing.assert_array_equal(xfer._swa_pages(full_pages, xfer.d_mapping), full_pages)


def test_swa_pages_dp_rank_indexing():
    """dp>1 building block: _swa_pages must select the CALLER-specified
    dp_rank's mapping array, not always rank 0 -- each rank's mapping uses
    that rank's own local index space (allocator.py: "Each rank has
    independent [1, size_per_rank] indices"), so mixing ranks is wrong.
    Default dp_rank=0 preserves today's dp=1-only behavior exactly."""
    fake = SimpleNamespace(page_size=PAGE_SIZE, _swa_pages=PathwaysPDKVTransfer._swa_pages)
    full_pages = np.array([0, 1, 2], np.int32)
    # rank 0: full page k -> swa page k (identity)
    map_r0 = np.arange(NUM_PAGES * PAGE_SIZE, dtype=np.int64)
    # rank 1: full page k -> swa page (k+1) mod NUM_PAGES (shifted by one page)
    map_r1 = ((map_r0 // PAGE_SIZE + 1) % NUM_PAGES) * PAGE_SIZE + map_r0 % PAGE_SIZE
    mapping = [map_r0, map_r1]

    # default dp_rank=0 (unchanged call signature still works)
    np.testing.assert_array_equal(fake._swa_pages(fake, full_pages, mapping), full_pages)
    np.testing.assert_array_equal(fake._swa_pages(fake, full_pages, mapping, dp_rank=0), full_pages)
    # dp_rank=1 must use map_r1, NOT silently fall back to map_r0
    np.testing.assert_array_equal(
        fake._swa_pages(fake, full_pages, mapping, dp_rank=1), full_pages + 1
    )
    # non-list mapping (dp_size==1 real-world case): dp_rank is ignored, same array used
    np.testing.assert_array_equal(fake._swa_pages(fake, full_pages, map_r0, dp_rank=1), full_pages)


# ---- e2e: full migrate_reqs_p_to_d path on CPU mock pools ----


class _MockR2T:
    def __init__(self, max_reqs: int, max_len: int):
        self.req_to_token = np.full((max_reqs, max_len), -1, np.int32)
        self.free_slots = list(range(max_reqs))

    def alloc(self, reqs):
        for r in reqs:
            r.req_pool_idx = self.free_slots.pop(0)


class _MockAlloc:
    def __init__(self, n_tokens: int):
        self._free = list(range(n_tokens))

    def alloc(self, n, dp_rank=0):
        if len(self._free) < n:
            return None
        out, self._free = np.asarray(self._free[:n], np.int32), self._free[n:]
        return out

    def free(self, slots, dp_rank=0):
        self._free.extend(int(s) for s in slots)

    def available_size(self):
        return len(self._free)


def _mk_req(rid, fill_ids, p_r2t, p_alloc):
    r = SimpleNamespace(
        rid=rid,
        fill_ids=list(fill_ids),
        dp_rank=0,
        req_pool_idx=None,
        prefix_indices=None,
        last_node=object(),
        cache_protected_len=99,
        kv_committed_len=0,
        kv_allocated_len=0,
    )
    p_r2t.alloc([r])
    slots = p_alloc.alloc(len(fill_ids))
    p_r2t.req_to_token[r.req_pool_idx, : len(fill_ids)] = slots
    return r, slots


@pytest.mark.unit
def test_migrate_reqs_e2e(meshes):
    """End-to-end migrate on CPU mock: P pool -> transfer -> D pool, plus
    req bookkeeping (req_pool_idx swap, prefix_indices remap, P slots freed)."""
    p_mesh, d_mesh = meshes
    p_pool = _make_pool(p_mesh, "seq")
    d_pool = _make_pool(d_mesh, "zero")
    xfer = PathwaysPDKVTransfer(p_mesh, d_mesh, p_pool, d_pool)

    p_r2t, d_r2t = _MockR2T(8, 32), _MockR2T(8, 32)
    p_alloc = _MockAlloc(NUM_PAGES * PAGE_SIZE)
    d_alloc = _MockAlloc(NUM_PAGES * PAGE_SIZE)

    r0, p_slots0 = _mk_req("r0", range(6), p_r2t, p_alloc)  # 6 tok -> 2 pages
    r1, p_slots1 = _mk_req("r1", range(9), p_r2t, p_alloc)  # 9 tok -> 3 pages
    p_avail_before = p_alloc.available_size()
    assert (r0.req_pool_idx, r1.req_pool_idx) == (0, 1)

    migrate_reqs_p_to_d([r0, r1], PAGE_SIZE, p_r2t, p_alloc, d_r2t, d_alloc, xfer)

    # --- req bookkeeping ---
    assert r0.req_pool_idx == 0 and r1.req_pool_idx == 1  # reassigned on D side
    assert sorted(p_r2t.free_slots) == list(range(8))  # P r2t fully freed
    assert p_alloc.available_size() == p_avail_before + 6 + 9  # P slots freed
    assert d_alloc.available_size() == NUM_PAGES * PAGE_SIZE - (2 + 3) * PAGE_SIZE
    for r, n in ((r0, 6), (r1, 9)):
        assert r.kv_committed_len == n and r.kv_allocated_len == n
        assert r.last_node is None and r.cache_protected_len == 0
        np.testing.assert_array_equal(d_r2t.req_to_token[r.req_pool_idx, :n], r.prefix_indices)

    # --- KV bytes: D[d_slot] == P[p_slot] for every migrated token, all layers ---
    for r, p_slots in ((r0, p_slots0), (r1, p_slots1)):
        d_slots = r.prefix_indices
        for layer in range(N_LAYERS):
            p_host = np.asarray(p_pool.kv_buffer[layer]).reshape(-1, HEAD_DIM)
            d_host = np.asarray(d_pool.kv_buffer[layer]).reshape(-1, HEAD_DIM)
            np.testing.assert_array_equal(d_host[d_slots], p_host[p_slots])


class _FakeDrainSelf:
    """Minimal stand-in for `Scheduler` exposing only what
    `_pd_drain_ready_multi` touches, so the multi-item drain loop can be
    tested without booting a real scheduler/mesh/allocator stack."""

    def __init__(self, ready_items=(), defer_items=(), drain_fn=None, ready_maxsize=8):
        self._pd_ready_q = queue.Queue(maxsize=ready_maxsize)
        for it in ready_items:
            self._pd_ready_q.put_nowait(it)
        self._pd_defer = list(defer_items)
        self._drain_fn = drain_fn
        self.drain_calls = 0

    def _pd_drain_ready(self):
        self.drain_calls += 1
        self._drain_fn(self)


def _drain_one_ready(fake: _FakeDrainSelf) -> None:
    """Simulates a successful (non-deferred) drain: pop one ready_q item."""
    try:
        fake._pd_ready_q.get_nowait()
    except queue.Empty:
        pass


def _drain_stuck_defer(fake: _FakeDrainSelf) -> None:
    """Simulates a D-pool-full item that never fits: pop + re-append,
    leaving both ready_q and defer-list lengths unchanged."""
    item = fake._pd_defer.pop(0)
    fake._pd_defer.append(item)


def test_drain_multi_clears_backlog():
    """A full ready_q backlog drains in a single tick, not one-per-tick."""
    fake = _FakeDrainSelf(ready_items=list(range(5)), drain_fn=_drain_one_ready)
    n_calls = PathwaysPDSchedulerMixin._pd_drain_ready_multi(fake)
    assert fake._pd_ready_q.empty()
    assert n_calls == 5
    assert fake.drain_calls == 5


def test_drain_multi_empty_makes_no_calls():
    """Nothing to drain -> the loop must not call _pd_drain_ready at all."""
    fake = _FakeDrainSelf(drain_fn=_drain_one_ready)
    n_calls = PathwaysPDSchedulerMixin._pd_drain_ready_multi(fake)
    assert n_calls == 0
    assert fake.drain_calls == 0


def test_drain_multi_stops_on_no_progress():
    """A permanently-stuck deferred item (D pool never frees up) must not
    burn the full maxsize+1 retry budget -- one no-op call, then exit."""
    fake = _FakeDrainSelf(defer_items=["stuck-req"], drain_fn=_drain_stuck_defer)
    n_calls = PathwaysPDSchedulerMixin._pd_drain_ready_multi(fake)
    assert n_calls == 1
    assert fake._pd_defer == ["stuck-req"]  # still stuck, untouched otherwise


def test_drain_multi_mixed_backlog_then_stuck_defer():
    """Ready_q items drain first (each shrinks the queue -> progress), then
    a stuck defer item causes exit on the next no-progress call."""
    fake = _FakeDrainSelf(ready_items=[1, 2, 3], defer_items=["stuck-req"])

    def drain(f: _FakeDrainSelf) -> None:
        if not f._pd_ready_q.empty():
            _drain_one_ready(f)
        else:
            _drain_stuck_defer(f)

    fake._drain_fn = drain
    n_calls = PathwaysPDSchedulerMixin._pd_drain_ready_multi(fake)
    assert fake._pd_ready_q.empty()
    assert n_calls == 4  # 3 ready items + 1 no-progress defer call before exit
    assert fake._pd_defer == ["stuck-req"]


# ---------------------------------------------------------------------------
# D-side token-level admission reservation (#1427 follow-up: KV backpressure)
# ---------------------------------------------------------------------------


class _FakeReq:
    def __init__(self, rid: str, n_tokens: int):
        self.rid = rid
        self.origin_input_ids = [0] * n_tokens


class _FakeFullAlloc:
    """SWA-style allocator: full_available_size differs from (min-based)
    available_size, so the gate must read the full pool, not the min."""

    def __init__(self, full_by_rank, min_by_rank=None):
        self._full = full_by_rank
        self._min = min_by_rank or full_by_rank

    def full_available_size(self, dp_rank: int = 0) -> int:
        return self._full[dp_rank]

    def available_size(self, dp_rank: int = 0) -> int:
        return self._min[dp_rank]


class _FakeMinOnlyAlloc:
    """Non-SWA allocator: only available_size exists (getattr fallback)."""

    def __init__(self, by_rank):
        self._by_rank = by_rank

    def available_size(self, dp_rank: int = 0) -> int:
        return self._by_rank[dp_rank]


class _FakeGateSelf(PathwaysPDSchedulerMixin):
    """Subclasses the mixin so the gate/ledger helpers under test can call
    each other through self, with only the state they touch defined here."""

    page_size = PAGE_SIZE

    def __init__(self, d_allocs, dp_size=1, reserved_per=0, d_running=0):
        self.d_allocs = d_allocs
        self.dp_size = dp_size
        self._pd_reserved_per = reserved_per
        self._d_running = d_running
        self._pd_inflight_rids = set()
        self._pd_inflight_tokens = 0
        self._pd_gate_ticks = 0

    def _pd_total_d_running(self) -> int:
        return self._d_running


def test_pd_req_kv_tokens_page_padded():
    fake = _FakeGateSelf(d_allocs=[])
    f = PathwaysPDSchedulerMixin._pd_req_kv_tokens
    assert f(fake, _FakeReq("a", 1)) == PAGE_SIZE
    assert f(fake, _FakeReq("b", PAGE_SIZE)) == PAGE_SIZE
    assert f(fake, _FakeReq("c", PAGE_SIZE + 1)) == 2 * PAGE_SIZE


def test_track_untrack_inflight_idempotent():
    fake = _FakeGateSelf(d_allocs=[])
    r = _FakeReq("r1", 5)  # pads to 8
    PathwaysPDSchedulerMixin._pd_track_inflight(fake, r)
    assert fake._pd_inflight_tokens == 8
    # chunked req: chunk-2..N re-track the same rid -> counted once
    PathwaysPDSchedulerMixin._pd_track_inflight(fake, r)
    assert fake._pd_inflight_tokens == 8
    PathwaysPDSchedulerMixin._pd_untrack_inflight(fake, r)
    assert fake._pd_inflight_tokens == 0
    assert not fake._pd_inflight_rids
    # double-untrack (failure path after success path) must not go negative
    PathwaysPDSchedulerMixin._pd_untrack_inflight(fake, r)
    assert fake._pd_inflight_tokens == 0


def test_token_gate_open_under_capacity():
    fake = _FakeGateSelf(d_allocs=[_FakeMinOnlyAlloc({0: 100})])
    PathwaysPDSchedulerMixin._pd_track_inflight(fake, _FakeReq("r1", 40))
    assert not PathwaysPDSchedulerMixin._pd_token_gate_closed(fake)


def test_token_gate_closes_on_projected_exhaustion():
    # inflight 40 + reserved 2*(running 20 + inflight 1) = 82 >= avail 80
    fake = _FakeGateSelf(d_allocs=[_FakeMinOnlyAlloc({0: 80})], reserved_per=2, d_running=20)
    PathwaysPDSchedulerMixin._pd_track_inflight(fake, _FakeReq("r1", 40))
    assert PathwaysPDSchedulerMixin._pd_token_gate_closed(fake)


def test_token_gate_sums_across_d_and_ranks():
    # 2 D targets x 2 ranks x 30 = 120 total; inflight 100 < 120 -> open
    allocs = [_FakeMinOnlyAlloc({0: 30, 1: 30}), _FakeMinOnlyAlloc({0: 30, 1: 30})]
    fake = _FakeGateSelf(d_allocs=allocs, dp_size=2)
    PathwaysPDSchedulerMixin._pd_track_inflight(fake, _FakeReq("r1", 100))
    assert not PathwaysPDSchedulerMixin._pd_token_gate_closed(fake)
    PathwaysPDSchedulerMixin._pd_track_inflight(fake, _FakeReq("r2", 100))
    assert PathwaysPDSchedulerMixin._pd_token_gate_closed(fake)


def test_token_gate_reads_full_pool_not_min_for_swa():
    # swa side transiently tiny (min=4) but full pool has room (100): the
    # IL projection must gate on the full pool -- the swa transient is
    # handled exactly by the drain-time gate + defer. min() here would
    # wedge admission for SWA models whose swa pool is window-sized.
    fake = _FakeGateSelf(d_allocs=[_FakeFullAlloc({0: 100}, min_by_rank={0: 4})])
    PathwaysPDSchedulerMixin._pd_track_inflight(fake, _FakeReq("r1", 40))
    assert not PathwaysPDSchedulerMixin._pd_token_gate_closed(fake)


def test_token_gate_env_escape(monkeypatch):
    monkeypatch.setenv("SGLANG_PD_NO_TOKEN_GATE", "1")
    fake = _FakeGateSelf(d_allocs=[_FakeMinOnlyAlloc({0: 0})])
    PathwaysPDSchedulerMixin._pd_track_inflight(fake, _FakeReq("r1", 400))
    assert not PathwaysPDSchedulerMixin._pd_token_gate_closed(fake)


def test_fuse_for_batch_is_batch_level():
    """Fused-sample eligibility must consider the batch's logprob flags, not
    just the worker flag -- the overlap client picks its unpack arity from
    this predicate (#1469 review)."""
    from types import SimpleNamespace as _NS

    from sgl_jax.srt.managers.tp_worker import ModelWorker

    w = ModelWorker.__new__(ModelWorker)
    w._pd_fuse_sample = True
    dec = _NS(is_decode=lambda: True)
    ext = _NS(is_decode=lambda: False)
    plain = _NS(return_logprob=False, return_output_logprob_only=False, forward_mode=dec)
    lp = _NS(return_logprob=True, return_output_logprob_only=False, forward_mode=dec)
    olp = _NS(return_logprob=False, return_output_logprob_only=True, forward_mode=dec)
    prefill = _NS(return_logprob=False, return_output_logprob_only=False, forward_mode=ext)
    assert w._pd_fuse_for_batch(plain)
    assert not w._pd_fuse_for_batch(lp)
    assert not w._pd_fuse_for_batch(olp)
    # the PD prefill thread's EXTEND batches must never take the fused path
    assert not w._pd_fuse_for_batch(prefill)
    w._pd_fuse_sample = False
    assert not w._pd_fuse_for_batch(plain)
