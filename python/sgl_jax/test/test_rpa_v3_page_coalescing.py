"""RPA v3 coalesce_page_dma on page tables that are not increasing.

With coalesce_page_dma, the KV-cache fetch and the new-KV writeback issue
one DMA per run of physically contiguous pages, so a fully contiguous range
is a single DMA. A request's page list is not
increasing in general: a radix-cached prefix followed by newly allocated
pages, or a prefill chunk that continues a partly filled page, gives lists
such as [5, 6, 2, 8] or [9, 3, 11]. They are not contiguous although the
first and last pages are n - 1 apart.

Every page of the cache holds distinct data, including the pages of other
requests and free pages. Each case checks the attention output against the
reference and the whole updated KV cache exactly, so reading a wrong page,
writing into another request's page and skipping a write all fail.
"""

from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels.ragged_paged_attention import ragged_paged_attention_v3
from sgl_jax.srt.kernels.ragged_paged_attention.ragged_paged_attention_v3 import (
    RpaCase,
    get_default_block_sizes,
    ragged_paged_attention,
    ref_ragged_paged_attention,
)
from sgl_jax.srt.mem_cache.allocator import (
    PagedTokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)

pytestmark = pytest.mark.skipif(jax.default_backend() != "tpu", reason="Requires TPU DMA support")

NUM_HEADS, HEAD_DIM, PAGE_SIZE, NUM_PAGES = 4, 128, 16, 32
# bkv_sz = 64 tokens = 4 pages per KV block, so one block spans several pages.
D_BLOCK_SIZES = (1, 64, 1, 32)
M_BLOCK_SIZES = (16, 64, 16, 32)


def _make_batch(sequences, *, num_pages=NUM_PAGES, page_size=PAGE_SIZE, seed=0):
    """Each sequence is (new token count, cached prefix length, physical pages)."""
    rng = np.random.default_rng(seed)
    q_lens = np.array([s[0] for s in sequences])
    prefixes = np.array([s[1] for s in sequences])
    page_lists = [np.asarray(s[2], np.int32) for s in sequences]
    kv_lens = q_lens + prefixes
    page_counts = np.array([len(pages) for pages in page_lists])
    assert np.all(page_counts == (kv_lens + page_size - 1) // page_size), page_counts
    all_pages = np.concatenate(page_lists)
    assert len(np.unique(all_pages)) == len(all_pages) and all_pages.max() < num_pages
    cu_q_lens = np.r_[0, np.cumsum(q_lens)]
    page_offsets = np.r_[0, np.cumsum(page_counts)]
    num_tokens = int(cu_q_lens[-1])
    padded_tokens = (num_tokens + 31) // 32 * 32
    # Distinct data on every page, so reading or writing a wrong page shows up.
    cache = rng.integers(-4, 5, size=(num_pages, page_size, 1, 2, HEAD_DIM)).astype(np.float32)
    expected_cache = cache.copy()
    new_kv = np.zeros((padded_tokens, 2, HEAD_DIM), np.float32)
    ref_pages = np.zeros((len(sequences), max(page_counts)), np.int32)
    for i, (prefix, pages) in enumerate(zip(prefixes, page_lists)):
        kv = rng.integers(-4, 5, size=(kv_lens[i], 2, HEAD_DIM)).astype(np.float32)
        ref_pages[i, : len(pages)] = pages
        positions = np.arange(kv_lens[i])
        physical_pages, slots = pages[positions // page_size], positions % page_size
        expected_cache[physical_pages, slots, 0] = kv
        cache[physical_pages[:prefix], slots[:prefix], 0] = kv[:prefix]
        new_kv[cu_q_lens[i] : cu_q_lens[i + 1]] = kv[prefix:]
    queries = rng.uniform(-0.25, 0.25, (padded_tokens, NUM_HEADS, HEAD_DIM)).astype(np.float32)
    return dict(
        queries=queries,
        new_kv=new_kv,
        cache=cache,
        expected_cache=expected_cache,
        kv_lens=kv_lens.astype(np.int32),
        page_indices=all_pages,
        cu_q_lens=cu_q_lens.astype(np.int32),
        cu_kv_lens=(page_offsets * page_size).astype(np.int32),
        ref_pages=ref_pages,
        num_seqs=len(sequences),
        num_tokens=num_tokens,
        all_decode=bool(np.all(q_lens == 1)),
    )


def _reference(batch):
    out = ref_ragged_paged_attention(
        jnp.asarray(batch["queries"], jnp.bfloat16),
        jnp.asarray(batch["expected_cache"][:, :, :, 0, :], jnp.bfloat16),
        jnp.asarray(batch["expected_cache"][:, :, :, 1, :], jnp.bfloat16),
        jnp.asarray(batch["kv_lens"]),
        jnp.asarray(batch["ref_pages"]),
        jnp.asarray(batch["cu_q_lens"]),
        jnp.array([batch["num_seqs"]], jnp.int32),
        sm_scale=HEAD_DIM**-0.5,
    )
    return np.asarray(out, np.float32)


def _run(batch, *, chunk_prefill_size=None, **kwargs):
    n = batch["num_seqs"]
    if batch["all_decode"]:
        distribution = [n, n, n]
        kwargs.setdefault("d_block_sizes", D_BLOCK_SIZES)
    elif chunk_prefill_size is not None:
        distribution = [0, n, n]
        kwargs.setdefault("p_block_sizes", M_BLOCK_SIZES)
    else:
        distribution = [0, 0, n]
    kwargs.setdefault("m_block_sizes", M_BLOCK_SIZES)
    # Fresh device arrays per call: queries/keys/values/cache are donated.
    output, cache = ragged_paged_attention(
        jnp.asarray(batch["queries"], jnp.bfloat16),
        jnp.asarray(batch["new_kv"][:, 0:1], jnp.bfloat16),
        jnp.asarray(batch["new_kv"][:, 1:2], jnp.bfloat16),
        jnp.asarray(batch["cache"], jnp.bfloat16),
        jnp.asarray(batch["kv_lens"]),
        jnp.asarray(batch["page_indices"]),
        jnp.asarray(batch["cu_q_lens"]),
        jnp.asarray(batch["cu_kv_lens"]),
        jnp.asarray(distribution, jnp.int32),
        None,
        sm_scale=HEAD_DIM**-0.5,
        chunk_prefill_size=chunk_prefill_size,
        **kwargs,
    )
    output, cache = jax.device_get((output, cache))
    return output[: batch["num_tokens"]].astype(np.float32), cache.astype(np.float32)


def _check(batch, output, cache):
    wrong_pages = np.flatnonzero(np.any(cache != batch["expected_cache"], axis=(1, 2, 3, 4)))
    assert wrong_pages.size == 0, f"KV cache pages {wrong_pages.tolist()} written wrong"
    assert np.isfinite(output).all(), f"{np.isnan(output).sum()} NaN in the attention output"
    np.testing.assert_allclose(output, _reference(batch), atol=0.03, rtol=0.03)


CASES = {
    # Decode: the cache fetch of a KV block with 3-4 non-contiguous pages.
    "decode-radix-prefix": dict(
        # Prefix pages 5, 6 from the radix cache, then 2 and 8 from the free list;
        # page 7 belongs to another request.
        sequences=[(1, 62, [5, 6, 2, 8]), (1, 10, [7]), (1, 30, [3, 4])],
    ),
    "decode-endpoints": dict(sequences=[(1, 63, [10, 3, 7, 13])]),
    "decode-three-pages": dict(sequences=[(1, 40, [3, 11, 5]), (1, 20, [4, 6])]),
    "decode-second-kv-block": dict(sequences=[(1, 127, [1, 2, 3, 4, 9, 6, 7, 12])]),
    "decode-runs": dict(sequences=[(1, 63, [4, 5, 9, 10]), (1, 60, [1, 2, 3, 14])]),
    "decode-contiguous": dict(
        sequences=[(1, 63, [1, 2, 3, 4]), (1, 100, [5, 6, 7, 8, 9, 10, 11])],
    ),
    # Mixed: the new-KV writeback over 3-4 non-contiguous pages, and the fetch.
    "mixed-chunk-continuation": dict(
        # The second chunk continues page 9; 3 and 11 come from the free list and
        # page 10 belongs to another request.
        sequences=[(40, 8, [9, 3, 11]), (1, 15, [10])],
    ),
    "mixed-gap-owner": dict(sequences=[(40, 8, [5, 1, 7]), (1, 15, [6])]),
    "mixed-radix-prefix-multi-bq": dict(
        # Three Q blocks over one KV block (resident bkv reuse); the writeback
        # covers pages [6, 2, 8].
        sequences=[(48, 16, [5, 6, 2, 8]), (1, 10, [7])],
    ),
    "mixed-fetch-and-update": dict(sequences=[(10, 54, [6, 1, 3, 9]), (17, 15, [2, 4])]),
    "mixed-multi-kv-block": dict(
        # KV block 0 fetches pages [12, 2, 3, 15]; KV block 1 writes back to
        # [5, 9, 7, 8], and page 6 belongs to the second request.
        sequences=[(70, 60, [12, 2, 3, 15, 5, 9, 7, 8, 20]), (1, 30, [4, 6])],
    ),
    "mixed-contiguous": dict(sequences=[(40, 8, [3, 4, 5]), (20, 30, [6, 7, 8, 9])]),
    # Writeback ranges that start mid-page (8 tokens of the first page are
    # cached), have their first and last pages n - 1 apart, and hold a run of
    # several contiguous pages, which is written by one DMA.
    "mixed-page-runs-head": dict(sequences=[(50, 8, [9, 10, 3, 12]), (1, 15, [11])]),
    "mixed-page-runs-middle": dict(sequences=[(50, 8, [9, 3, 4, 12]), (1, 20, [10, 11])]),
    "mixed-page-runs-second-kv-block": dict(
        # KV block 1 writes back to [20, 21, 14, 23] from token 8 of page 20;
        # page 22 belongs to the second request.
        sequences=[(56, 72, [1, 2, 5, 6, 20, 21, 14, 23]), (1, 30, [22, 7])],
    ),
    "chunk-prefill": dict(
        sequences=[(32, 8, [9, 3, 11]), (32, 40, [4, 13, 6, 7, 1])],
        chunk_prefill_size=32,
    ),
}


@pytest.mark.parametrize("coalesce_page_dma", [True, False], ids=["coalesce", "per-page"])
@pytest.mark.parametrize("name", list(CASES))
def test_non_increasing_page_tables(name, coalesce_page_dma):
    case = dict(CASES[name])
    batch = _make_batch(case.pop("sequences"))
    output, cache = _run(batch, coalesce_page_dma=coalesce_page_dma, **case)
    _check(batch, output, cache)


def _pages(token_indices):
    return list(dict.fromkeys((np.asarray(token_indices) // PAGE_SIZE).tolist()))


def _radix_prefix_layout():
    """Decode after a radix-cached prefix: page list [5, 6, 2, 8] from the allocator."""
    alloc = PagedTokenToKVPoolAllocator(
        size=NUM_PAGES * PAGE_SIZE, page_size=PAGE_SIZE, kvcache=None
    )
    finished = alloc.alloc(2 * PAGE_SIZE)  # pages 1, 2
    running = alloc.alloc(2 * PAGE_SIZE)  # pages 3, 4
    prefix = alloc.alloc(2 * PAGE_SIZE)  # pages 5, 6, kept by the radix cache
    running2 = alloc.alloc(PAGE_SIZE)  # page 7
    alloc.free(finished[PAGE_SIZE:])  # page 2 released
    alloc.merge_and_sort_free()
    # A new request reuses the 32-token prefix and extends it to 62 tokens.
    extend = alloc.alloc_extend([32], [62], [int(prefix[-1])], extend_num_tokens=30)
    pages = _pages(prefix) + _pages(extend)
    return [(1, 62, pages), (1, 20, _pages(running)), (1, 9, _pages(running2))], pages


def _chunk_continuation_layout():
    """Second prefill chunk that continues a partly filled page: [9, 3, 11]."""
    alloc = PagedTokenToKVPoolAllocator(
        size=NUM_PAGES * PAGE_SIZE, page_size=PAGE_SIZE, kvcache=None
    )
    held = alloc.alloc(8 * PAGE_SIZE)  # pages 1..8
    chunk1 = alloc.alloc(PAGE_SIZE)  # page 9: the first chunk wrote 8 tokens
    other = alloc.alloc(PAGE_SIZE)  # page 10, another running request
    alloc.free(held[2 * PAGE_SIZE : 3 * PAGE_SIZE])  # page 3 released
    alloc.merge_and_sort_free()
    extend = alloc.alloc_extend([8], [48], [int(chunk1[7])], extend_num_tokens=40)
    pages = list(dict.fromkeys(_pages(chunk1) + _pages(extend)))
    return [
        (40, 8, pages),
        (1, 15, _pages(other)),
        (1, 50, _pages(held[3 * PAGE_SIZE : 7 * PAGE_SIZE])),
    ], pages


@pytest.mark.parametrize(
    "layout",
    [_radix_prefix_layout, _chunk_continuation_layout],
    ids=["radix-prefix", "chunk-continuation"],
)
def test_allocator_page_tables(layout):
    sequences, pages = layout()
    # The allocator must still produce a non-contiguous list whose endpoints are
    # len - 1 apart, or this test no longer covers the case.
    assert pages[-1] - pages[0] == len(pages) - 1, pages
    assert np.any(np.diff(pages) != 1), pages
    batch = _make_batch(sequences)
    for coalesce_page_dma in (True, False):
        output, cache = _run(batch, coalesce_page_dma=coalesce_page_dma)
        _check(batch, output, cache)


def test_token_allocator_decode_page_table():
    """page_size 1 decode on a TokenToKVPoolAllocator page list: [1, 2, 21, 4, 5].

    The allocator keeps its free list sorted, so slots freed by finished requests
    are reused before higher slots handed out earlier.
    """
    alloc = TokenToKVPoolAllocator(size=64, kvcache=None)
    # Prefill a = [1, 2], x = [3], b = [4], c = [5..20], then one decode step.
    a, x, b, c = (alloc.alloc(n).tolist() for n in (2, 1, 1, 16))
    for seq, slot in zip((a, x, b, c), alloc.alloc(4).tolist()):
        seq.append(slot)
    # b and c finish; x keeps running, and a decodes two more tokens.
    alloc.free_group_begin()
    alloc.free(np.array(b, np.int32))
    alloc.free(np.array(c, np.int32))
    alloc.free_group_end()
    a += [int(alloc.alloc(1)[0]) for _ in range(2)]
    # The cached part of a is fetched as one KV block; it must still be a
    # non-contiguous list whose endpoints are len - 1 apart.
    cached = a[:-1]
    assert cached[-1] - cached[0] == len(cached) - 1, a
    assert np.any(np.diff(cached) != 1), a
    batch = _make_batch([(1, len(a) - 1, a), (1, len(x) - 1, x)], num_pages=65, page_size=1)
    for coalesce_page_dma in (True, False):
        output, cache = _run(
            batch, coalesce_page_dma=coalesce_page_dma, d_block_sizes=(1, 32, 1, 32)
        )
        _check(batch, output, cache)


V7_PAGE_SIZE = 128
# 64 pages of 128 tokens per sequence, so the v7 heuristic is not capped by max_kv.
V7_PAGES_PER_SEQ = 64
# KV block of 16 pages, 2048 tokens: list entries from 40 to 55 with pages of
# other requests (41..54) in between, so its first and last pages are 15 apart.
V7_SCRAMBLED_BLOCK = [40, *range(17, 24), 31, 32, *range(24, 29), 55]


def _v7_default_block_sizes(case):
    """The block sizes get_default_block_sizes picks on TPU v7 for these shapes."""
    with mock.patch.object(ragged_paged_attention_v3, "get_tpu_version", return_value=7):
        sizes = get_default_block_sizes(
            jnp.bfloat16,
            jnp.bfloat16,
            actual_num_q_heads=NUM_HEADS,
            actual_num_kv_heads=1,
            head_dim=HEAD_DIM,
            page_size=V7_PAGE_SIZE,
            max_num_tokens=64,
            max_num_seqs=2,
            pages_per_seq=V7_PAGES_PER_SEQ,
            case=case,
            vmem_limit_bytes=64 * 1024 * 1024,
        )
    return tuple(sizes[k] for k in ("bq_sz", "bkv_sz", "bq_csz", "bkv_csz"))


def test_v7_default_decode_tile():
    """Decode with the TPU v7 default tile of 2048 tokens (16 pages of 128).

    A long sequence is fetched in several 16-page KV blocks; its second block
    is the scrambled list above, and the other request owns pages 41..54.
    """
    d_block_sizes = _v7_default_block_sizes(RpaCase.DECODE)
    assert d_block_sizes == (1, 2048, 1, 2048), d_block_sizes
    long_seq = [*range(1, 17), *V7_SCRAMBLED_BLOCK, 29, 30, 33, 34]  # 36 pages
    sequences = [(1, 4499, long_seq), (1, 1699, list(range(41, 55)))]
    batch = _make_batch(sequences, num_pages=64, page_size=V7_PAGE_SIZE)
    for coalesce_page_dma in (True, False):
        output, cache = _run(
            batch,
            coalesce_page_dma=coalesce_page_dma,
            d_block_sizes=d_block_sizes,
            m_block_sizes=_v7_default_block_sizes(RpaCase.MIXED),
        )
        _check(batch, output, cache)


def test_v7_default_mixed_tile():
    """Mixed batch with the TPU v7 default tiles (one 2048-token compute tile).

    The new KV of the first request is written back over the scrambled
    16-page list from token 64 of page 40; the other request owns pages 41..54.
    """
    m_block_sizes = _v7_default_block_sizes(RpaCase.MIXED)
    assert m_block_sizes == (32, 2048, 32, 2048), m_block_sizes
    sequences = [(1984, 64, V7_SCRAMBLED_BLOCK), (1, 1699, list(range(41, 55)))]
    batch = _make_batch(sequences, num_pages=64, page_size=V7_PAGE_SIZE)
    for coalesce_page_dma in (True, False):
        output, cache = _run(
            batch, coalesce_page_dma=coalesce_page_dma, m_block_sizes=m_block_sizes
        )
        _check(batch, output, cache)
