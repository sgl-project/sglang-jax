"""Benchmark for FlashAttention.get_forward_metadata optimization.

Constructs mock data matching MiMo-V2-Flash server config:
  dp_size=2, per_dp_bs_size=64, page_size=256, context_length=262144

Usage:
    python -m sgl_jax.test.bench_forward_metadata
"""

import time
from dataclasses import dataclass
from enum import IntEnum, auto

import numpy as np


# Minimal mocks to avoid importing the full codebase
class ForwardMode(IntEnum):
    EXTEND = auto()
    DECODE = auto()


@dataclass
class MockBatch:
    forward_mode: ForwardMode
    cache_loc: np.ndarray
    seq_lens: np.ndarray
    extend_seq_lens: np.ndarray | None
    dp_size: int
    per_dp_bs_size: int


def build_decode_batch(
    dp_size: int = 2,
    per_dp_bs_size: int = 64,
    page_size: int = 256,
    min_seq_len: int = 1024,
    max_seq_len: int = 32768,
    seed: int = 42,
) -> MockBatch:
    """Construct a decode-mode batch with realistic data sizes."""
    rng = np.random.RandomState(seed)
    total_bs = dp_size * per_dp_bs_size

    # Random seq_lens in [min_seq_len, max_seq_len], aligned to page_size
    raw_lens = rng.randint(min_seq_len, max_seq_len + 1, size=total_bs)
    seq_lens = ((raw_lens + page_size - 1) // page_size * page_size).astype(np.int32)

    # Build cache_loc: for each request, token slots are contiguous pages
    # Total tokens per DP rank must be equal for reshape to work
    cache_loc_parts = []
    slot_offset = 0
    for dp in range(dp_size):
        dp_start = dp * per_dp_bs_size
        dp_end = dp_start + per_dp_bs_size
        dp_seq_lens = seq_lens[dp_start:dp_end]
        rank_total = int(np.sum(dp_seq_lens))

        # Assign contiguous slot ranges per request
        rank_slots = np.arange(slot_offset, slot_offset + rank_total, dtype=np.int32)
        cache_loc_parts.append(rank_slots)
        slot_offset += rank_total

    # Ensure equal per-dp loc lengths by padding shorter ranks
    max_rank_len = max(len(p) for p in cache_loc_parts)
    for i in range(len(cache_loc_parts)):
        if len(cache_loc_parts[i]) < max_rank_len:
            pad = np.zeros(max_rank_len - len(cache_loc_parts[i]), dtype=np.int32)
            cache_loc_parts[i] = np.concatenate([cache_loc_parts[i], pad])

    cache_loc = np.concatenate(cache_loc_parts)

    return MockBatch(
        forward_mode=ForwardMode.DECODE,
        cache_loc=cache_loc,
        seq_lens=seq_lens,
        extend_seq_lens=None,
        dp_size=dp_size,
        per_dp_bs_size=per_dp_bs_size,
    )


def build_swa_mapping(dp_size: int, pool_size: int = 2_000_000, seed: int = 123):
    """Build a list of swa_index_mapping arrays (one per DP rank)."""
    rng = np.random.RandomState(seed)
    # Each mapping is a permutation-like array: full_slot -> swa_slot
    mappings = []
    for _ in range(dp_size):
        mapping = rng.randint(0, pool_size, size=pool_size, dtype=np.int32)
        mappings.append(mapping)
    return mappings


# ---- Original implementation (copied from flashattention_backend.py) ----
def get_forward_metadata_original(page_size, batch, swa_mapping=None):
    """Original implementation with Python for-loops."""
    total_loc_len = len(batch.cache_loc)
    per_dp_loc_len = total_loc_len // batch.dp_size

    # Phase 1: page_indices
    page_indices_list = []
    for i in range(batch.dp_size):
        start = i * per_dp_loc_len
        end = (i + 1) * per_dp_loc_len
        rank_cache_loc = batch.cache_loc[start:end]
        remainder = len(rank_cache_loc) % page_size
        if remainder > 0:
            pad_len = page_size - remainder
            rank_cache_loc = np.concatenate([rank_cache_loc, np.zeros(pad_len, dtype=np.int32)])
        rank_selected_locs = rank_cache_loc[::page_size]
        rank_page_indices = rank_selected_locs // page_size
        page_indices_list.append(rank_page_indices)
    page_indices = np.concatenate(page_indices_list)

    # Phase 2: swa_page_indices
    swa_page_indices = None
    if swa_mapping is not None:
        swa_page_indices_list = []
        for i in range(batch.dp_size):
            start = i * per_dp_loc_len
            end = (i + 1) * per_dp_loc_len
            rank_cache_loc = batch.cache_loc[start:end]
            mapping = swa_mapping[i] if isinstance(swa_mapping, list) else swa_mapping
            rank_swa_cache_loc = mapping[rank_cache_loc]
            remainder = len(rank_swa_cache_loc) % page_size
            if remainder > 0:
                pad_len = page_size - remainder
                rank_swa_cache_loc = np.concatenate(
                    [rank_swa_cache_loc, np.zeros(pad_len, dtype=np.int32)]
                )
            rank_swa_selected = rank_swa_cache_loc[::page_size]
            rank_swa_page_indices = rank_swa_selected // page_size
            swa_page_indices_list.append(rank_swa_page_indices)
        swa_page_indices = np.concatenate(swa_page_indices_list)

    # Phase 3: cu_q_lens (DECODE)
    if batch.forward_mode == ForwardMode.DECODE:
        cu_q_lens_sections = []
        for i in range(0, len(batch.seq_lens), batch.per_dp_bs_size):
            section_cu = np.concatenate(
                [
                    np.array([0], dtype=np.int32),
                    np.cumsum(np.ones(batch.per_dp_bs_size, dtype=np.int32)),
                ]
            )
            cu_q_lens_sections.append(section_cu)
        cu_q_lens = np.concatenate(cu_q_lens_sections)
    elif batch.forward_mode == ForwardMode.EXTEND:
        cu_q_lens_sections = []
        for i in range(0, len(batch.extend_seq_lens), batch.per_dp_bs_size):
            section_lens = batch.extend_seq_lens[i : i + batch.per_dp_bs_size]
            section_cu = np.concatenate(
                [
                    np.array([0], dtype=np.int32),
                    np.cumsum(section_lens, dtype=np.int32),
                ]
            )
            cu_q_lens_sections.append(section_cu)
        cu_q_lens = np.concatenate(cu_q_lens_sections)

    # Phase 4: seq_lens
    seq_lens = np.copy(batch.seq_lens)
    aligned_seq_lens = ((batch.seq_lens + page_size - 1) // page_size) * page_size

    # Phase 5: cu_kv_lens
    cu_kv_lens_sections = []
    for i in range(0, len(aligned_seq_lens), batch.per_dp_bs_size):
        section_lens = aligned_seq_lens[i : i + batch.per_dp_bs_size]
        section_cu = np.concatenate(
            [
                np.array([0], dtype=np.int32),
                np.cumsum(section_lens, dtype=np.int32),
            ]
        )
        cu_kv_lens_sections.append(section_cu)
    cu_kv_lens = np.concatenate(cu_kv_lens_sections)

    # Phase 6: distribution
    distribution_list = []
    for i in range(0, len(batch.seq_lens), batch.per_dp_bs_size):
        section_seq_lens = batch.seq_lens[i : i + batch.per_dp_bs_size]
        local_num_seqs = np.sum(section_seq_lens > 0, dtype=np.int32)
        if batch.forward_mode == ForwardMode.DECODE:
            dist = np.array([local_num_seqs, local_num_seqs, local_num_seqs], dtype=np.int32)
        else:
            dist = np.array([0, local_num_seqs, local_num_seqs], dtype=np.int32)
        distribution_list.append(dist)
    distribution = np.concatenate(distribution_list)

    return page_indices, swa_page_indices, cu_q_lens, cu_kv_lens, seq_lens, distribution


# ---- Optimized implementation ----
def get_forward_metadata_optimized(page_size, batch, swa_mapping=None):
    """Optimized: stride-first for SWA, vectorized numpy ops, no Python for-loops."""
    total_loc_len = len(batch.cache_loc)
    per_dp_loc_len = total_loc_len // batch.dp_size

    # Phase 1: page_indices — reshape to 2D (view), stride (view), divide
    cache_loc_2d = batch.cache_loc.reshape(batch.dp_size, per_dp_loc_len)
    # Stride first: pick one slot per page. np stride handles non-aligned lengths.
    strided_2d = cache_loc_2d[:, ::page_size]  # shape (dp_size, n_pages_per_rank)
    page_indices = (strided_2d // page_size).ravel()

    # Phase 2: swa_page_indices — KEY OPTIMIZATION: stride first, then mapping
    # mapping[a][::s] == mapping[a[::s]] algebraically, so apply mapping on
    # ~4K strided entries instead of ~1.1M full entries.
    swa_page_indices = None
    if swa_mapping is not None:
        n_pages = strided_2d.shape[1]
        swa_strided = np.empty((batch.dp_size, n_pages), dtype=np.int32)
        for i in range(batch.dp_size):
            m = swa_mapping[i] if isinstance(swa_mapping, list) else swa_mapping
            swa_strided[i] = m[strided_2d[i]]
        swa_page_indices = (swa_strided // page_size).ravel()

    # Phase 3: cu_q_lens — precomputed, no loops
    if batch.forward_mode == ForwardMode.DECODE:
        single_cu = np.arange(batch.per_dp_bs_size + 1, dtype=np.int32)
        cu_q_lens = np.tile(single_cu, batch.dp_size)
    elif batch.forward_mode == ForwardMode.EXTEND:
        ext_2d = batch.extend_seq_lens.reshape(batch.dp_size, batch.per_dp_bs_size)
        cu_q_2d = np.zeros((batch.dp_size, batch.per_dp_bs_size + 1), dtype=np.int32)
        cu_q_2d[:, 1:] = np.cumsum(ext_2d, axis=1)
        cu_q_lens = cu_q_2d.ravel()

    # Phase 4: seq_lens — no copy needed, device_put makes its own copy
    seq_lens = batch.seq_lens
    aligned_seq_lens = ((batch.seq_lens + page_size - 1) // page_size) * page_size

    # Phase 5: cu_kv_lens — vectorized 2D cumsum
    aligned_2d = aligned_seq_lens.reshape(batch.dp_size, batch.per_dp_bs_size)
    cu_kv_2d = np.zeros((batch.dp_size, batch.per_dp_bs_size + 1), dtype=np.int32)
    cu_kv_2d[:, 1:] = np.cumsum(aligned_2d, axis=1)
    cu_kv_lens = cu_kv_2d.ravel()

    # Phase 6: distribution — vectorized
    seq_lens_2d = batch.seq_lens.reshape(batch.dp_size, batch.per_dp_bs_size)
    local_num_seqs = np.sum(seq_lens_2d > 0, axis=1, dtype=np.int32)
    if batch.forward_mode == ForwardMode.DECODE:
        distribution = np.repeat(local_num_seqs, 3)
    else:
        distribution = np.column_stack(
            [np.zeros_like(local_num_seqs), local_num_seqs, local_num_seqs]
        ).ravel()

    return page_indices, swa_page_indices, cu_q_lens, cu_kv_lens, seq_lens, distribution


def verify_correctness(page_size, batch, swa_mapping):
    """Verify optimized output matches original exactly."""
    orig = get_forward_metadata_original(page_size, batch, swa_mapping)
    opt = get_forward_metadata_optimized(page_size, batch, swa_mapping)

    names = [
        "page_indices",
        "swa_page_indices",
        "cu_q_lens",
        "cu_kv_lens",
        "seq_lens",
        "distribution",
    ]
    all_ok = True
    for name, o, p in zip(names, orig, opt):
        if o is None and p is None:
            continue
        if o is None or p is None:
            print(f"  FAIL {name}: one is None")
            all_ok = False
            continue
        if not np.array_equal(o, p):
            diff_count = np.sum(o != p)
            print(f"  FAIL {name}: {diff_count}/{len(o)} elements differ")
            print(f"    orig[:10] = {o[:10]}")
            print(f"    opt[:10]  = {p[:10]}")
            all_ok = False
        else:
            print(f"  OK   {name}: shape={o.shape}")
    return all_ok


def benchmark(fn, page_size, batch, swa_mapping, n_iters=200, warmup=10):
    """Time a function over n_iters, return median time in ms."""
    for _ in range(warmup):
        fn(page_size, batch, swa_mapping)

    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn(page_size, batch, swa_mapping)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times.sort()
    return {
        "median_ms": times[len(times) // 2],
        "p95_ms": times[int(len(times) * 0.95)],
        "mean_ms": sum(times) / len(times),
        "min_ms": times[0],
    }


def main():
    page_size = 256
    dp_size = 2
    per_dp_bs_size = 64

    print(f"Config: dp_size={dp_size}, per_dp_bs={per_dp_bs_size}, page_size={page_size}")

    batch = build_decode_batch(
        dp_size=dp_size,
        per_dp_bs_size=per_dp_bs_size,
        page_size=page_size,
        min_seq_len=1024,
        max_seq_len=32768,
    )
    print(f"Batch: cache_loc.shape={batch.cache_loc.shape}, seq_lens.shape={batch.seq_lens.shape}")
    print(f"Total cache_loc entries: {len(batch.cache_loc):,}")
    print(f"Mean seq_len: {np.mean(batch.seq_lens):.0f}")

    # Build SWA mapping
    max_slot = int(np.max(batch.cache_loc)) + 1
    pool_size = max(max_slot, 2_000_000)
    swa_mapping = build_swa_mapping(dp_size, pool_size)
    print(f"SWA mapping pool_size: {pool_size:,}")

    # Verify correctness
    print("\nVerifying correctness...")
    ok = verify_correctness(page_size, batch, swa_mapping)
    if ok:
        print("All outputs match!\n")
    else:
        print("MISMATCH detected!\n")
        return

    # Benchmark
    n_iters = 200
    print(f"Benchmarking ({n_iters} iterations)...")

    orig_stats = benchmark(get_forward_metadata_original, page_size, batch, swa_mapping, n_iters)
    print(
        f"  Original:  median={orig_stats['median_ms']:.3f}ms  "
        f"p95={orig_stats['p95_ms']:.3f}ms  mean={orig_stats['mean_ms']:.3f}ms"
    )

    opt_stats = benchmark(get_forward_metadata_optimized, page_size, batch, swa_mapping, n_iters)
    print(
        f"  Optimized: median={opt_stats['median_ms']:.3f}ms  "
        f"p95={opt_stats['p95_ms']:.3f}ms  mean={opt_stats['mean_ms']:.3f}ms"
    )

    speedup = orig_stats["median_ms"] / opt_stats["median_ms"]
    print(f"\n  Speedup: {speedup:.2f}x (median)")


def build_extend_batch(
    dp_size: int = 2,
    per_dp_bs_size: int = 8,
    page_size: int = 256,
    min_seq_len: int = 256,
    max_seq_len: int = 8192,
    seed: int = 99,
) -> MockBatch:
    """Construct an extend-mode batch."""
    rng = np.random.RandomState(seed)
    total_bs = dp_size * per_dp_bs_size

    raw_lens = rng.randint(min_seq_len, max_seq_len + 1, size=total_bs)
    seq_lens = ((raw_lens + page_size - 1) // page_size * page_size).astype(np.int32)

    # extend_seq_lens: portion being extended (subset of seq_lens)
    extend_seq_lens = np.minimum(rng.randint(1, 2048, size=total_bs), seq_lens).astype(np.int32)

    cache_loc_parts = []
    slot_offset = 0
    for dp in range(dp_size):
        dp_start = dp * per_dp_bs_size
        dp_end = dp_start + per_dp_bs_size
        dp_seq_lens = seq_lens[dp_start:dp_end]
        rank_total = int(np.sum(dp_seq_lens))
        rank_slots = np.arange(slot_offset, slot_offset + rank_total, dtype=np.int32)
        cache_loc_parts.append(rank_slots)
        slot_offset += rank_total

    max_rank_len = max(len(p) for p in cache_loc_parts)
    for i in range(len(cache_loc_parts)):
        if len(cache_loc_parts[i]) < max_rank_len:
            pad = np.zeros(max_rank_len - len(cache_loc_parts[i]), dtype=np.int32)
            cache_loc_parts[i] = np.concatenate([cache_loc_parts[i], pad])

    cache_loc = np.concatenate(cache_loc_parts)

    return MockBatch(
        forward_mode=ForwardMode.EXTEND,
        cache_loc=cache_loc,
        seq_lens=seq_lens,
        extend_seq_lens=extend_seq_lens,
        dp_size=dp_size,
        per_dp_bs_size=per_dp_bs_size,
    )


def test_edge_cases():
    """Test correctness across various edge cases."""
    page_size = 256
    cases = [
        # (description, batch_builder_kwargs, use_swa)
        (
            "decode dp=1 bs=1",
            dict(dp_size=1, per_dp_bs_size=1, min_seq_len=256, max_seq_len=256),
            False,
        ),
        (
            "decode dp=1 bs=1 + SWA",
            dict(dp_size=1, per_dp_bs_size=1, min_seq_len=256, max_seq_len=256),
            True,
        ),
        (
            "decode dp=4 bs=32",
            dict(dp_size=4, per_dp_bs_size=32, min_seq_len=256, max_seq_len=65536),
            True,
        ),
        (
            "decode dp=2 bs=64 long seq",
            dict(dp_size=2, per_dp_bs_size=64, min_seq_len=16384, max_seq_len=262144),
            True,
        ),
        (
            "decode dp=1 bs=128 no SWA",
            dict(dp_size=1, per_dp_bs_size=128, min_seq_len=512, max_seq_len=4096),
            False,
        ),
        (
            "decode dp=2 bs=1 minimal",
            dict(dp_size=2, per_dp_bs_size=1, min_seq_len=256, max_seq_len=256),
            True,
        ),
    ]

    all_passed = True
    for desc, kwargs, use_swa in cases:
        batch = build_decode_batch(page_size=page_size, **kwargs)
        swa = None
        if use_swa:
            max_slot = int(np.max(batch.cache_loc)) + 1
            swa = build_swa_mapping(kwargs["dp_size"], max(max_slot, 1000))

        orig = get_forward_metadata_original(page_size, batch, swa)
        opt = get_forward_metadata_optimized(page_size, batch, swa)

        names = [
            "page_indices",
            "swa_page_indices",
            "cu_q_lens",
            "cu_kv_lens",
            "seq_lens",
            "distribution",
        ]
        ok = True
        for name, o, p in zip(names, orig, opt):
            if o is None and p is None:
                continue
            if not np.array_equal(o, p):
                ok = False
                break

        status = "PASS" if ok else "FAIL"
        if not ok:
            all_passed = False
        print(
            f"  {status}  {desc}  (bs={kwargs['dp_size']*kwargs['per_dp_bs_size']}, swa={use_swa})"
        )

    # EXTEND mode tests
    extend_cases = [
        ("extend dp=2 bs=8 + SWA", dict(dp_size=2, per_dp_bs_size=8), True),
        ("extend dp=1 bs=4 no SWA", dict(dp_size=1, per_dp_bs_size=4), False),
    ]
    for desc, kwargs, use_swa in extend_cases:
        batch = build_extend_batch(page_size=page_size, **kwargs)
        swa = None
        if use_swa:
            max_slot = int(np.max(batch.cache_loc)) + 1
            swa = build_swa_mapping(kwargs["dp_size"], max(max_slot, 1000))

        orig = get_forward_metadata_original(page_size, batch, swa)
        opt = get_forward_metadata_optimized(page_size, batch, swa)

        names = [
            "page_indices",
            "swa_page_indices",
            "cu_q_lens",
            "cu_kv_lens",
            "seq_lens",
            "distribution",
        ]
        ok = True
        for name, o, p in zip(names, orig, opt):
            if o is None and p is None:
                continue
            if not np.array_equal(o, p):
                ok = False
                break

        status = "PASS" if ok else "FAIL"
        if not ok:
            all_passed = False
        print(
            f"  {status}  {desc}  (bs={kwargs['dp_size']*kwargs['per_dp_bs_size']}, swa={use_swa})"
        )

    return all_passed


if __name__ == "__main__":
    print("=" * 60)
    print("Edge case correctness tests")
    print("=" * 60)
    ok = test_edge_cases()
    if ok:
        print("All edge cases passed!\n")
    else:
        print("SOME EDGE CASES FAILED!\n")
        exit(1)

    print("=" * 60)
    print("Performance benchmark")
    print("=" * 60)
    main()
