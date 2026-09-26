"""CPU parity: grouped (G queries per request) batched page-topk vs the loop reference.

The spec verify batch is S requests x G draft tokens (cu_q_lens step == G).
``streamindex_page_topk_ref_grouped`` must pick, per draft token, the same page
set as ``streamindex_page_topk_ref`` (general causal path), including the
intra-request causal rule: draft token i must not see keys of draft tokens > i.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels.dsa.ref import (
    streamindex_page_topk_ref,
    streamindex_page_topk_ref_grouped,
)

PAGE = 8
D = 32
H = 6
PPS = 8  # pages per seq
K = 3
G = 4


def _meta(base_lens, pps=PPS, g=G, t_pad=None):
    S = len(base_lens)
    valid = np.asarray(base_lens) > 0
    ext = np.where(valid, g, 0).astype(np.int32)
    seq_lens = np.where(valid, np.asarray(base_lens) + g, 0).astype(np.int32)
    cu_q = np.concatenate([[0], np.cumsum(ext)]).astype(np.int32)
    cu_kv = (np.arange(S + 1) * pps * PAGE).astype(np.int32)
    page_indices = (1 + np.arange(S * pps)).astype(np.int32)
    dist = np.array([0, 0, int(valid.sum())], np.int32)
    T = int(cu_q[-1]) if t_pad is None else t_pad
    return T, seq_lens, cu_q, cu_kv, page_indices, dist


def _inputs(key, T, S, pps=PPS):
    k1, k2, k3 = jax.random.split(key, 3)
    q = jax.random.normal(k1, (T, H, D), jnp.float32).astype(jnp.bfloat16)
    w = jax.random.uniform(k2, (T, H), jnp.float32).astype(jnp.bfloat16)
    cache = jax.random.normal(k3, (1 + S * pps, PAGE, D), jnp.float32).astype(jnp.bfloat16)
    return q, w, cache


def _sets(pages):
    pages = np.asarray(pages)
    return [frozenset(int(p) for p in row if p >= 0) for row in pages]


def _run_both(base_lens, t_pad=None, k_pages=K, seed=0):
    T, seq_lens, cu_q, cu_kv, pi, dist = _meta(base_lens, t_pad=t_pad)
    S = len(base_lens)
    q, w, cache = _inputs(jax.random.PRNGKey(seed), T, S)
    ref = streamindex_page_topk_ref(
        q, w, cache, seq_lens, pi, cu_q, cu_kv, dist, k_pages=k_pages, pages_per_seq=PPS
    )
    got = streamindex_page_topk_ref_grouped(
        q, w, cache, seq_lens, pi, cu_q, cu_kv, k_pages=k_pages, pages_per_seq=PPS, q_group=G
    )
    assert got.shape == (T, k_pages) and got.dtype == jnp.int32
    return ref, got, cu_q


def test_grouped_matches_loop_ref_per_token_sets():
    ref, got, cu_q = _run_both([5, 17, 40, 1, 60])
    assert _sets(got) == _sets(ref)


def test_padded_request_and_trailing_pad_rows_are_minus_one():
    # request 3 is padded (seq_len 0, q_len 0); 6 trailing pad rows beyond cu_q[-1]
    ref, got, cu_q = _run_both([9, 33, 20, 0, 12], t_pad=4 * G + 6)
    assert _sets(got) == _sets(ref)
    assert np.all(np.asarray(got)[int(cu_q[-1]) :] == -1)


def test_k_pages_clamped_to_pages_per_seq():
    ref, got, _ = _run_both([3, 50], k_pages=PPS)
    assert _sets(got) == _sets(ref)


def test_intra_request_causal_mask_hides_later_draft_keys():
    # One request: base 12 tokens + G=4 drafts -> kv_len 16 = pages 0,1; the
    # drafts sit at absolute positions 12..15 (page 1). Every key is zero except
    # the one at position 15 (= last draft token), so page 1 only scores > 0 for
    # the query that may see position 15. With k_pages=1 drafts 0..2 must pick
    # page 0 (ties -> lowest page) and draft 3 must pick page 1.
    T, seq_lens, cu_q, cu_kv, pi, dist = _meta([12])
    q = jnp.ones((T, H, D), jnp.bfloat16)
    w = jnp.ones((T, H), jnp.bfloat16)
    cache = jnp.zeros((1 + PPS, PAGE, D), jnp.float32)
    cache = cache.at[2, 7, :].set(1.0).astype(jnp.bfloat16)  # physical page 2 = seq page 1
    ref = streamindex_page_topk_ref(
        q, w, cache, seq_lens, pi, cu_q, cu_kv, dist, k_pages=1, pages_per_seq=PPS
    )
    got = streamindex_page_topk_ref_grouped(
        q, w, cache, seq_lens, pi, cu_q, cu_kv, k_pages=1, pages_per_seq=PPS, q_group=G
    )
    assert np.asarray(ref)[:, 0].tolist() == [0, 0, 0, 1]
    assert np.asarray(got)[:, 0].tolist() == [0, 0, 0, 1]


@pytest.mark.parametrize("unroll_heads", [True, False])
def test_head_chunk_boundary_and_unroll_modes(unroll_heads):
    T, seq_lens, cu_q, cu_kv, pi, dist = _meta([7, 30, 45])
    q, w, cache = _inputs(jax.random.PRNGKey(3), T, 3)
    ref = streamindex_page_topk_ref(
        q, w, cache, seq_lens, pi, cu_q, cu_kv, dist, k_pages=K, pages_per_seq=PPS
    )
    got = streamindex_page_topk_ref_grouped(
        q,
        w,
        cache,
        seq_lens,
        pi,
        cu_q,
        cu_kv,
        k_pages=K,
        pages_per_seq=PPS,
        q_group=G,
        head_chunk=4,
        unroll_heads=unroll_heads,
    )
    assert _sets(got) == _sets(ref)


def test_head_chunking_is_invariant_on_same_input():
    T, seq_lens, cu_q, cu_kv, pi, dist = _meta([11, 26, 3])
    q, w, cache = _inputs(jax.random.PRNGKey(5), T, 3)
    outs = [
        streamindex_page_topk_ref_grouped(
            q,
            w,
            cache,
            seq_lens,
            pi,
            cu_q,
            cu_kv,
            k_pages=K,
            pages_per_seq=PPS,
            q_group=G,
            head_chunk=hc,
        )
        for hc in (1, 4, 8)
    ]
    assert _sets(outs[0]) == _sets(outs[1]) == _sets(outs[2])
