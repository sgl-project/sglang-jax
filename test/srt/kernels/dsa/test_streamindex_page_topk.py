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
