"""Manual TPU validation harness for HiCache L2 (RFC-1.0 + RFC-1.2).

NOT part of the CPU unit suite — run this on a real multi-chip TPU to exercise
the behaviors CPU cannot cover:

  1. ``pinned_host`` memory_kind is *actually* applied to staged slots (the CPU
     path silently falls back to a device sharding — see ``_make_host_sharding``).
  2. Multi-chip sharded D2H/H2D round-trip is bit-exact under the real
     ``_slot_spec`` derivation (replicated layer axis, dropped page/DP axis, TP
     axis kept on heads), at ``page_size`` both ==1 and >1 (a slot holds a whole
     device page).
  3. The async-write / sync-load split over :class:`HiCacheController` round-trips.
  4. The measured D2H/H2D bandwidth asymmetry the design is premised on
     (spike: ~2.8 GB/s D2H vs ~45.7 GB/s H2D).
  5. DP>1 + page_size>1: the controller speaks GLOBAL device page ids; the tree
     globalizes per-rank LOCAL pages (``local_page + dp_rank * pages_per_shard``)
     before crossing the boundary. Two ranks backing up the *same* local page id
     with different KV must each load back their own physical page — never the
     other rank's. This exercises the real ``data`` + ``tensor`` sharding that
     CPU single-device cannot.

Usage (on a TPU pod, inside the venv, from the repo root)::

    python -m sgl_jax.test.mem_cache.tpu_validate_hicache

Exit code 0 = all checks passed. Results are printed in a form ready to paste
into ``docs/design/rfc_1_3_e2e_validation.md``.
"""

from __future__ import annotations

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec

from sgl_jax.srt.mem_cache.hicache_controller import HiCacheController
from sgl_jax.srt.mem_cache.host_kv_pool import LRUHostKVPool, _make_host_sharding
from sgl_jax.srt.mem_cache.memory_pool import MHATokenToKVPool
from sgl_jax.srt.utils.mesh_utils import create_device_mesh

# Sized so a single slot is non-trivial for a meaningful bandwidth read while
# the pool/device stay small. head_num is sharded over the TP axis. The device
# size scales with page_size so there are always plenty of device pages to
# address regardless of page size.
_HEAD_NUM = 16
_HEAD_DIM = 128
_LAYER_NUM = 16
_DTYPE = jnp.bfloat16
_DEVICE_PAGES = 64  # device pages available for the round-trip checks
_HOST_PAGES = 16


def _section(title: str) -> None:
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")


def _build(page_size: int, dp_size: int, ici_parallelism):
    mesh = create_device_mesh(ici_parallelism=ici_parallelism, dcn_parallelism=[1, 1])
    device_pool = MHATokenToKVPool(
        size=_DEVICE_PAGES * page_size,
        page_size=page_size,
        dtype=_DTYPE,
        head_num=_HEAD_NUM,
        head_dim=_HEAD_DIM,
        layer_num=_LAYER_NUM,
        mesh=mesh,
        dp_size=dp_size,
    )
    per_layer_shape = tuple(int(d) for d in device_pool.kv_buffer[0].shape[1:])
    host_pool = LRUHostKVPool(
        device_pool=device_pool,
        pool_size=_HOST_PAGES,
        page_size=page_size,
        layer_num=device_pool.layer_num,
        per_layer_shape=per_layer_shape,
        dtype=device_pool.dtype,
        mesh=mesh,
        partition_spec=device_pool.kv_sharding.spec,
    )
    return mesh, device_pool, host_pool


def _fill_page(device_pool, device_page: int, seed: int):
    """Fill a whole device PAGE (leading-axis index) with random KV; return the
    per-layer numpy copies."""
    orig = []
    for layer in range(device_pool.layer_num):
        buf = device_pool.kv_buffer[layer]
        vals = jax.random.normal(
            jax.random.PRNGKey(seed * 100 + layer), buf.shape[1:], buf.dtype
        )
        device_pool.kv_buffer[layer] = buf.at[device_page].set(
            vals, out_sharding=buf.sharding
        )
        orig.append(np.asarray(device_pool.kv_buffer[layer])[device_page])
    return orig


def check_pinned_host(host_pool, device_pool, page_size: int) -> bool:
    _section(f"1. pinned_host memory_kind actually applied (page_size={page_size})")
    pages = host_pool.alloc(1)
    b = int(pages[0])
    host_pool.stage_backup([3], [b])
    host_pool.flush_backup([b])
    slot = host_pool._slots[b]
    kind = slot.sharding.memory_kind
    print(f"staged slot shape={slot.shape} dtype={slot.dtype} memory_kind={kind!r}")
    host_pool.free([b])
    ok = kind == "pinned_host"
    if not ok:
        print(
            "  FAIL: slot is not on pinned_host — either the platform lacks host "
            "memory kinds (CPU) or IPC_LOCK is missing on the pod spec."
        )
    return ok


def check_roundtrip(host_pool, device_pool, page_size: int) -> bool:
    _section(
        f"2. multi-chip sharded D2H/H2D page round-trip bit-exact (page_size={page_size})"
    )
    pairs = [(2, 10), (5, 11), (9, 12)]  # (src_page, dst_page)
    origs = {src: _fill_page(device_pool, src, seed=src) for src, _ in pairs}
    pages = [int(p) for p in host_pool.alloc(len(pairs))]
    srcs = [s for s, _ in pairs]
    dsts = [d for _, d in pairs]
    host_pool.stage_backup(srcs, pages)
    host_pool.flush_backup(pages)
    host_pool.copy_to_device(pages, dsts)
    ok = True
    for (src, dst) in pairs:
        for layer in range(device_pool.layer_num):
            got = np.asarray(device_pool.kv_buffer[layer])[dst]
            if not np.array_equal(got, origs[src][layer]):
                print(f"  FAIL: src={src} dst={dst} layer={layer} mismatch")
                ok = False
    host_pool.free(pages)
    if ok:
        print(
            f"  OK: {len(pairs)} pages x {device_pool.layer_num} layers bit-exact "
            f"(each page = {page_size} token(s))"
        )
    return ok


def check_controller(host_pool, device_pool, page_size: int) -> bool:
    _section(
        f"3. HiCacheController async-write / sync-load round-trip (page_size={page_size})"
    )
    ctrl = HiCacheController(host_pool, device_pool)
    ok = True
    try:
        src, dst = 7, 13  # device page ids
        orig = _fill_page(device_pool, src, seed=7)
        b = int(host_pool.alloc(1)[0])
        ctrl.write([src], [b])
        ctrl.drain_pending()
        ctrl.load([b], [dst])
        for layer in range(device_pool.layer_num):
            got = np.asarray(device_pool.kv_buffer[layer])[dst]
            if not np.array_equal(got, orig[layer]):
                print(f"  FAIL: layer={layer} mismatch after write->drain->load")
                ok = False
        before = host_pool.available_size()
        ctrl.evict_callback([b])
        if host_pool.available_size() != before + 1:
            print("  FAIL: evict_callback did not free the slot")
            ok = False
        if ok:
            print("  OK: write->drain->load bit-exact; evict frees slot")
    finally:
        ctrl.shutdown()
    return ok


def check_dp_page_isolation(page_size: int = 2, dp_size: int = 2) -> bool:
    """DP>1 + page>1: two ranks back up the SAME local page id with different KV;
    each must load back its own physical page (global-page math under real
    data+tensor sharding). Mirrors ``UnifiedRadixCache._to_global_device_pages``.
    """
    _section(
        f"5. DP={dp_size} + page_size={page_size} rank isolation "
        "(global device page math)"
    )
    if len(jax.devices()) < dp_size * 2:
        print(f"  SKIP: need >= {dp_size * 2} chips for a data={dp_size},tensor>=2 mesh")
        return True
    tensor = len(jax.devices()) // dp_size
    mesh, device_pool, host_pool = _build(
        page_size=page_size, dp_size=dp_size, ici_parallelism=[dp_size, tensor]
    )
    ctrl = HiCacheController(host_pool, device_pool)
    ok = True
    try:
        pages_per_shard = device_pool.kv_buffer[0].shape[0] // dp_size

        def global_page(local_page: int, rank: int) -> int:
            return local_page + rank * pages_per_shard

        local_page = 3  # SAME local page id on BOTH ranks
        gp0 = global_page(local_page, 0)
        gp1 = global_page(local_page, 1)
        if gp0 == gp1:
            print("  FAIL: global pages collide across ranks")
            return False
        orig0 = _fill_page(device_pool, gp0, seed=1)
        orig1 = _fill_page(device_pool, gp1, seed=2)  # different KV

        # Back up both ranks' pages, then restore each to a FRESH global page in
        # its own shard and verify isolation.
        h0 = int(host_pool.alloc(1)[0])
        h1 = int(host_pool.alloc(1)[0])
        ctrl.write([gp0], [h0])
        ctrl.write([gp1], [h1])
        ctrl.drain_pending()

        dst0 = global_page(local_page + 1, 0)
        dst1 = global_page(local_page + 1, 1)
        ctrl.load([h0], [dst0])
        ctrl.load([h1], [dst1])

        for layer in range(device_pool.layer_num):
            got0 = np.asarray(device_pool.kv_buffer[layer])[dst0]
            got1 = np.asarray(device_pool.kv_buffer[layer])[dst1]
            if not np.array_equal(got0, orig0[layer]):
                print(f"  FAIL: rank0 layer={layer} mismatch (got rank1's data?)")
                ok = False
            if not np.array_equal(got1, orig1[layer]):
                print(f"  FAIL: rank1 layer={layer} mismatch")
                ok = False
            if np.array_equal(got0, orig1[layer]):
                print(f"  FAIL: rank0 layer={layer} loaded rank1's physical page")
                ok = False
        if ok:
            print(
                f"  OK: dp_size={dp_size} pages_per_shard={pages_per_shard} "
                f"gp0={gp0} gp1={gp1}; both ranks bit-exact and isolated"
            )
    finally:
        ctrl.shutdown()
    return ok


def check_dp_full_chain(page_size: int = 128, dp_size: int = 2) -> bool:
    """Real ``UnifiedRadixCache`` DP control-plane round-trip on TPU.

    Unlike check 5 (which drives the low-level pool/controller directly and
    *reimplements* the global-page math), this runs the **production** path end
    to end: real ``PagedTokenToKVPoolAllocator`` (dp_rank-aware), radix tree
    insert -> write-through backup (D2H offload) -> device eviction (demote to
    host) -> match_prefix host hit -> ``init_load_back`` (H2D), which itself
    calls ``write_backup`` / ``_to_global_device_pages`` / the token<->page
    boundary conversions. Two ranks store the SAME token ids with different KV;
    each must reload its own data (rank isolation) bit-exact. Eviction, offload,
    and loadback are made visible via counters.
    """
    import time as _time

    from sgl_jax.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
    from sgl_jax.srt.mem_cache.base_prefix_cache import (
        EvictParams,
        InsertParams,
        MatchPrefixParams,
    )
    from sgl_jax.srt.mem_cache.memory_pool import ReqToTokenPool
    from sgl_jax.srt.mem_cache.radix_cache import RadixKey
    from sgl_jax.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    _section(
        f"6. REAL UnifiedRadixCache DP={dp_size} full chain "
        f"(page_size={page_size}: insert->offload->evict->loadback)"
    )
    if len(jax.devices()) < dp_size * 2:
        print(f"  SKIP: need >= {dp_size * 2} chips for a data={dp_size},tensor>=2 mesh")
        return True

    head_num, head_dim, layer_num = 4, 8, 2
    tensor = len(jax.devices()) // dp_size
    tokens_per_rank = page_size * 2  # two pages per rank
    device_size = page_size * 16  # room to insert both ranks + reload after evict

    mesh = create_device_mesh(
        ici_parallelism=[dp_size, tensor], dcn_parallelism=[1, 1]
    )
    jax.sharding.set_mesh(mesh)
    kv_cache = MHATokenToKVPool(
        size=device_size,
        page_size=page_size,
        dtype=jnp.float32,
        head_num=head_num,
        head_dim=head_dim,
        layer_num=layer_num,
        mesh=mesh,
        dp_size=dp_size,
    )
    allocator = PagedTokenToKVPoolAllocator(
        size=device_size, page_size=page_size, kvcache=kv_cache, dp_size=dp_size
    )
    req_pool = ReqToTokenPool(size=64, max_context_len=2048, dtype=np.int32)
    cache = UnifiedRadixCache(
        req_to_token_pool=req_pool,
        token_to_kv_pool_allocator=allocator,
        page_size=page_size,
        kv_head_num=head_num,
        head_dim=head_dim,
        layer_num=layer_num,
        max_seq_len=2048,
        dtype=jnp.float32,
    )
    host_pool = LRUHostKVPool(
        device_pool=kv_cache,
        pool_size=32,
        page_size=kv_cache.page_size,
        layer_num=kv_cache.layer_num,
        per_layer_shape=tuple(int(d) for d in kv_cache.kv_buffer[0].shape[1:]),
        dtype=kv_cache.dtype,
        mesh=mesh,
        partition_spec=kv_cache.kv_sharding.spec,
    )
    controller = HiCacheController(host_pool, kv_cache)
    cache.host_pool = host_pool
    cache.hicache_controller = controller
    cache.hicache_enabled = True
    cache.write_through_threshold = 1
    for component in cache._components_tuple:
        component._full_kv_pool_host = host_pool

    pages_per_shard = kv_cache.kv_buffer[0].shape[0] // dp_size
    gather_spec = PartitionSpec(*tuple(kv_cache.kv_sharding.spec)[1:])
    gather_sharding = NamedSharding(mesh, gather_spec)

    def _global_page(local_token, dp_rank):
        return int(local_token) // page_size + dp_rank * pages_per_shard

    def _read_token(layer, local_token, dp_rank):
        gp = _global_page(local_token, dp_rank)
        off = int(local_token) % page_size
        page = kv_cache.kv_buffer[layer].at[gp].get(out_sharding=gather_sharding)
        return np.asarray(page)[off]

    def _key(token_ids, dp_rank):
        return RadixKey(token_ids=token_ids, extra_key=None, dp_rank=dp_rank)

    def _fill(dp_rank, n, seed):
        local = allocator.alloc(n, dp_rank=dp_rank)
        token_shape = tuple(kv_cache.kv_buffer[0].shape[2:])
        orig = []
        for i, lidx in enumerate(local):
            gp = _global_page(lidx, dp_rank)
            off = int(lidx) % page_size
            per_layer = []
            for layer in range(layer_num):
                buf = kv_cache.kv_buffer[layer]
                vals = jax.random.normal(
                    jax.random.PRNGKey(seed * 100000 + dp_rank * 1000 + i * 10 + layer),
                    token_shape,
                    buf.dtype,
                )
                kv_cache.kv_buffer[layer] = buf.at[gp, off].set(
                    vals, out_sharding=buf.sharding
                )
                per_layer.append(_read_token(layer, lidx, dp_rank))
            orig.append(per_layer)
        return local, orig

    def _settle(timeout=10.0):
        controller.drain_pending()
        deadline = _time.time() + timeout
        peak = 0
        while cache.ongoing_write and _time.time() < deadline:
            peak = max(peak, len(cache.ongoing_write))
            cache.check_hicache_events()
            _time.sleep(0.005)
        return peak

    ok = True
    try:
        tokens = list(range(tokens_per_rank))  # identical ids on BOTH ranks
        local0, orig0 = _fill(0, tokens_per_rank, seed=1)
        local1, orig1 = _fill(1, tokens_per_rank, seed=2)  # different KV

        host_avail_before = host_pool.available_size()
        for dp_rank, local in ((0, local0), (1, local1)):
            key = _key(tokens, dp_rank)
            cache.insert(InsertParams(key=key, value=local))
            cache.insert(InsertParams(key=key, value=local))  # hit -> write-through
        peak_inflight = _settle()
        host_avail_after = host_pool.available_size()
        pages_offloaded = host_avail_before - host_avail_after
        expected_pages = (tokens_per_rank // page_size) * dp_size
        print(
            f"  OFFLOAD(D2H): host pages used={pages_offloaded} "
            f"(expected {expected_pages}), peak D2H in-flight={peak_inflight}"
        )
        if pages_offloaded != expected_pages:
            print("  FAIL: backup did not consume the expected host pages")
            ok = False

        # Demote both ranks' device KV to host.
        for dp_rank in (0, 1):
            cache.evict(EvictParams(num_tokens=tokens_per_rank, dp_rank=dp_rank))

        for dp_rank in (0, 1):
            mr = cache.match_prefix(MatchPrefixParams(key=_key(tokens, dp_rank)))
            if len(mr.device_indices) != 0:
                print(f"  FAIL: rank{dp_rank} still has device indices after evict")
                ok = False
            if mr.host_hit_length != tokens_per_rank:
                print(
                    f"  FAIL: rank{dp_rank} host_hit_length={mr.host_hit_length} "
                    f"!= {tokens_per_rank}"
                )
                ok = False
        print(
            f"  EVICT: both ranks demoted device->host "
            f"(device_indices=0, host_hit_length={tokens_per_rank})"
        )

        # Reload each rank and verify bit-exact against ITS OWN fixture.
        for dp_rank, orig in ((0, orig0), (1, orig1)):
            mr = cache.match_prefix(MatchPrefixParams(key=_key(tokens, dp_rank)))
            new_local, _ = cache.init_load_back(
                mr.last_host_node, mr.host_hit_length, mem_quota=device_size
            )
            n_loaded = len(new_local)
            if n_loaded != tokens_per_rank:
                print(f"  FAIL: rank{dp_rank} loaded {n_loaded} != {tokens_per_rank}")
                ok = False
                continue
            for i, lidx in enumerate(new_local):
                for layer in range(layer_num):
                    got = _read_token(layer, lidx, dp_rank)
                    if not np.allclose(got, orig[i][layer]):
                        print(f"  FAIL: rank{dp_rank} token{i} layer{layer} mismatch")
                        ok = False
                        break
            # cross-rank isolation: rank's reloaded data must NOT equal the other
            other = orig1 if dp_rank == 0 else orig0
            if np.allclose(_read_token(0, new_local[0], dp_rank), other[0][0]):
                print(f"  FAIL: rank{dp_rank} reloaded the OTHER rank's KV")
                ok = False
            print(
                f"  LOADBACK(H2D): rank{dp_rank} reloaded "
                f"{n_loaded // page_size} page(s) bit-exact, isolated"
            )
        if ok:
            print(
                f"  OK: real DP={dp_size} chain at page_size={page_size} "
                "(offload + evict + loadback) bit-exact and rank-isolated"
            )
    finally:
        controller.shutdown()
    return ok


def measure_bandwidth(host_pool, device_pool) -> bool:
    _section("4. D2H / H2D bandwidth (pure transfer, large array)")
    # The per-page slot is far too small to measure bandwidth: a single
    # device_put dispatch dwarfs the transfer, so the per-page number reflects
    # call overhead, not bandwidth. Measure instead a ~256 MB device array
    # round-tripped device<->pinned_host, head sharded over the TP axis
    # (mirroring the real KV layout). This is the spike's method and is what
    # actually exposes the D2H/H2D asymmetry.
    mesh = device_pool.kv_buffer[0].sharding.mesh
    spec = PartitionSpec(None, "tensor", None, None)
    dev_sharding = NamedSharding(mesh, spec)
    host_sharding = _make_host_sharding(mesh, spec)

    n_tok = 32768  # 32768 * 16 * 2 * 128 * 2B = 256 MB
    arr = jax.device_put(
        jax.random.normal(
            jax.random.PRNGKey(0), (n_tok, _HEAD_NUM, 2, _HEAD_DIM), _DTYPE
        ),
        dev_sharding,
    )
    jax.block_until_ready(arr)
    nbytes = arr.size * jnp.dtype(_DTYPE).itemsize
    reps = 10

    h = jax.device_put(arr, host_sharding)
    jax.block_until_ready(h)  # warm up
    t0 = time.perf_counter()
    for _ in range(reps):
        h = jax.device_put(arr, host_sharding)
        jax.block_until_ready(h)
    d2h_s = (time.perf_counter() - t0) / reps

    d = jax.device_put(h, dev_sharding)
    jax.block_until_ready(d)  # warm up
    t0 = time.perf_counter()
    for _ in range(reps):
        d = jax.device_put(h, dev_sharding)
        jax.block_until_ready(d)
    h2d_s = (time.perf_counter() - t0) / reps

    d2h_gbs = nbytes / d2h_s / 1e9
    h2d_gbs = nbytes / h2d_s / 1e9
    kind = h.sharding.memory_kind
    print(f"array_bytes={nbytes / 1e6:.1f} MB  reps={reps}  host_kind={kind!r}")
    print(f"  D2H: {d2h_s * 1e3:.3f} ms  {d2h_gbs:.2f} GB/s")
    print(f"  H2D: {h2d_s * 1e3:.3f} ms  {h2d_gbs:.2f} GB/s")
    print(f"  H2D/D2H ratio: {h2d_gbs / d2h_gbs:.1f}x")
    # informational, not a hard gate
    return True


def run_suite(page_size: int) -> dict[str, bool]:
    mesh, device_pool, host_pool = _build(
        page_size=page_size, dp_size=1, ici_parallelism=[1, -1]
    )
    print(f"\n##### page_size={page_size} #####")
    print(f"mesh: {mesh}")
    print(f"device kv_buffer[0].shape: {device_pool.kv_buffer[0].shape}")
    print(f"device kv_buffer[0].sharding: {device_pool.kv_buffer[0].sharding}")
    results = {
        f"pinned_host[ps={page_size}]": check_pinned_host(
            host_pool, device_pool, page_size
        ),
        f"roundtrip[ps={page_size}]": check_roundtrip(
            host_pool, device_pool, page_size
        ),
        f"controller[ps={page_size}]": check_controller(
            host_pool, device_pool, page_size
        ),
    }
    if page_size == 1:
        results["bandwidth"] = measure_bandwidth(host_pool, device_pool)
    return results


def main() -> int:
    print(f"jax devices: {jax.devices()}")
    print(f"process_count={jax.process_count()} process_index={jax.process_index()}")

    results: dict[str, bool] = {}
    for page_size in (1, 64):
        results.update(run_suite(page_size))
    results["dp2_page2_isolation"] = check_dp_page_isolation(page_size=2, dp_size=2)
    results["dp2_page128_full_chain"] = check_dp_full_chain(page_size=128, dp_size=2)

    _section("SUMMARY")
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
