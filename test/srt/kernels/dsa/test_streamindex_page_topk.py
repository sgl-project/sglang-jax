"""Parity: streamindex_page_topk (Pallas, page-pooled) vs ref general path.

TPU-only (same constraint as test_streamindex_topk.py).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels.dsa.ref import streamindex_page_topk_ref
from sgl_jax.srt.kernels.dsa.streamindex_topk import streamindex_page_topk

if jax.default_backend() != "tpu":
    pytest.skip("streamindex_page_topk kernel requires TPU", allow_module_level=True)


@pytest.mark.parametrize(
    "page_size, pages_per_seq, k_pages, bkv_p, bq_sz, q_lens, kv_lens",
    [
        # 2-seq packed extend, tail pages partially filled, causal within chunk
        (8, 16, 3, 128, 8, [5, 12], [37, 61]),
        # single long extend spanning many bkv blocks
        (8, 48, 5, 128, 16, [33], [301]),
        # chunked-prefill form: q_len < kv_len (prefix present)
        (8, 32, 4, 128, 8, [16], [200]),
        # decode-like degenerate rows inside MIXED (q_len 1) + extend
        (8, 16, 3, 128, 8, [1, 7], [90, 55]),
        # multi-block accumulation/prefetch (page mode pins bkv_p=128, so
        # exceeding one block needs pages_per_seq > 128): 138 pages -> 2 blocks
        (8, 144, 5, 128, 16, [33], [1100]),
        # multi-block + multi-seq: seq 1 spans 2 kv blocks (150 pages), seq 0 one
        (8, 160, 3, 128, 8, [5, 12], [900, 1200]),
    ],
)
def test_page_topk_parity(page_size, pages_per_seq, k_pages, bkv_p, bq_sz, q_lens, kv_lens):
    rng = np.random.default_rng(0)
    num_seqs = len(q_lens)
    H, D = 4, 128
    kv_packing = 2  # bf16
    total_pages = num_seqs * pages_per_seq + 8

    T = int(sum(q_lens))
    q = jnp.asarray(rng.standard_normal((T, H, D)), jnp.bfloat16)
    w = jnp.asarray(rng.standard_normal((T, H)), jnp.bfloat16)
    cache = jnp.asarray(
        rng.standard_normal((total_pages, page_size // kv_packing, kv_packing, D)),
        jnp.bfloat16,
    )
    pi = jnp.asarray(rng.permutation(total_pages)[: num_seqs * pages_per_seq], jnp.int32)
    seq_lens = jnp.asarray(kv_lens, jnp.int32)
    cu_q = jnp.asarray(np.concatenate([[0], np.cumsum(q_lens)]), jnp.int32)
    cu_kv = jnp.asarray(np.arange(num_seqs + 1) * pages_per_seq * page_size, jnp.int32)
    dist = jnp.asarray([0, 0, num_seqs], jnp.int32)

    ref_out = streamindex_page_topk_ref(
        q.astype(jnp.float32) if False else q,
        w,
        cache.reshape(total_pages, page_size, D),
        seq_lens,
        pi,
        cu_q,
        cu_kv,
        dist,
        k_pages=k_pages,
        pages_per_seq=pages_per_seq,
        one_token_per_seq=False,
    )

    ker_out = streamindex_page_topk(
        q,
        w,
        cache,
        seq_lens,
        pi,
        cu_q,
        jnp.int32(num_seqs),
        k_pages=k_pages,
        num_kv_pages_per_block=bkv_p,
        num_queries_per_block=bq_sz,
    )

    ref_np, ker_np = np.asarray(ref_out), np.asarray(ker_out)
    assert ker_np.shape == ref_np.shape == (T, k_pages)
    for t in range(T):
        ref_set = set(ref_np[t][ref_np[t] >= 0].tolist())
        ker_set = set(ker_np[t][ker_np[t] >= 0].tolist())
        assert ker_set == ref_set, f"row {t}: kernel {ker_set} != ref {ref_set}"


@pytest.mark.parametrize(
    "page_size, k_pages, bkv_p, q_lens, kv_lens",
    [
        # unequal per-seq page counts (5 vs 13): the production packing places
        # seq 1's pages at cu_kv_lens[1]//ps == 5, NOT at 1*pages_per_seq
        (8, 3, 128, [5, 12], [37, 101]),
        # 3 seqs, unequal, one spanning 2 kv blocks (188 pages > bkv_p=128)
        (8, 4, 128, [9, 3, 20], [70, 17, 1500]),
    ],
)
def test_page_topk_variable_stride_packing(page_size, k_pages, bkv_p, q_lens, kv_lens):
    """Production-layout contract: sglang packs seq i's pages starting at
    ``cu_kv_lens[i] // page_size`` (variable stride, page-aligned cumsum —
    see ``schedule_batch._merge_cache_loc``), while the kernel indexes
    ``page_indices[seq_id * pages_per_seq + p]`` (fixed stride). The call
    site must repack via ``_fixed_stride_pages`` — passing the packed table
    directly makes every query of an offset sequence read another sequence's
    pages (measured 12/17 wrong rows on the first case here).
    """
    from sgl_jax.srt.layers.attention.dsa_sparse_backend import _fixed_stride_pages

    rng = np.random.default_rng(1)
    num_seqs = len(q_lens)
    H, D = 4, 128
    kv_packing = 2
    T = int(sum(q_lens))
    n_pages = [(l + page_size - 1) // page_size for l in kv_lens]
    # window width the backend derives: len(page_indices) // num_seqs
    pages_per_seq = max(n_pages) + 3
    pi_len = num_seqs * pages_per_seq
    total_pages = pi_len + 8

    q = jnp.asarray(rng.standard_normal((T, H, D)), jnp.bfloat16)
    w = jnp.asarray(rng.standard_normal((T, H)), jnp.bfloat16)
    cache = jnp.asarray(
        rng.standard_normal((total_pages, page_size // kv_packing, kv_packing, D)),
        jnp.bfloat16,
    )
    # variable-stride packed table + page-aligned cumsum cu_kv_lens
    pi = jnp.asarray(rng.permutation(total_pages)[:pi_len], jnp.int32)
    cu_kv = jnp.asarray(
        np.concatenate([[0], np.cumsum([n * page_size for n in n_pages])]), jnp.int32
    )
    seq_lens = jnp.asarray(kv_lens, jnp.int32)
    cu_q = jnp.asarray(np.concatenate([[0], np.cumsum(q_lens)]), jnp.int32)
    dist = jnp.asarray([0, 0, num_seqs], jnp.int32)

    ref_out = streamindex_page_topk_ref(
        q,
        w,
        cache.reshape(total_pages, page_size, D),
        seq_lens,
        pi,
        cu_q,
        cu_kv,
        dist,
        k_pages=k_pages,
        pages_per_seq=pages_per_seq,
        one_token_per_seq=False,
    )
    ker_out = streamindex_page_topk(
        q,
        w,
        cache,
        seq_lens,
        _fixed_stride_pages(pi, cu_kv, page_size, pages_per_seq),
        cu_q,
        jnp.int32(num_seqs),
        k_pages=k_pages,
        num_kv_pages_per_block=bkv_p,
        num_queries_per_block=8,
    )
    ref_np, ker_np = np.asarray(ref_out), np.asarray(ker_out)
    for t in range(T):
        ref_set = set(ref_np[t][ref_np[t] >= 0].tolist())
        ker_set = set(ker_np[t][ker_np[t] >= 0].tolist())
        assert ker_set == ref_set, f"row {t}: kernel {ker_set} != ref {ref_set}"
