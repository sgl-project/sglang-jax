"""Parity: batched decode page-topk vs the per-seq fori_loop reference.

Compares the selected page SETS per sequence (top_k tie order may differ
between the batched and looped accumulation), plus exact agreement whenever
the page scores are distinct.
"""

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.dsa.ref import (
    streamindex_page_topk_ref,
    streamindex_page_topk_ref_batched,
)

jax.config.update("jax_platform_name", "cpu")

PAGE, D, H = 8, 32, 6
PPS = 8
K_PAGES = 3


def _case(seed, S, T_pad):
    rng = np.random.default_rng(seed)
    num_pages = 1 + S * PPS + 4
    cache = jnp.asarray(rng.standard_normal((num_pages, PAGE, D)), jnp.float32)
    q = jnp.asarray(rng.standard_normal((T_pad, H, D)), jnp.float32)
    w = jnp.asarray(rng.random((T_pad, H)), jnp.float32)
    seq_lens = jnp.asarray(rng.integers(1, PPS * PAGE, (S,)), jnp.int32)
    pi = jnp.asarray(rng.permutation(np.arange(1, num_pages))[: S * PPS], jnp.int32)
    cu_q = jnp.asarray(np.arange(T_pad + 1), jnp.int32)
    cu_kv = jnp.asarray(np.arange(S + 1) * PPS * PAGE, jnp.int32)
    dist = jnp.asarray([S, S, S], jnp.int32)
    ref = streamindex_page_topk_ref(
        q,
        w,
        cache,
        seq_lens,
        pi,
        cu_q,
        cu_kv,
        dist,
        k_pages=K_PAGES,
        pages_per_seq=PPS,
        one_token_per_seq=True,
    )
    outs = [
        streamindex_page_topk_ref_batched(
            q,
            w,
            cache,
            seq_lens,
            pi,
            cu_kv,
            k_pages=K_PAGES,
            pages_per_seq=PPS,
            head_chunk=4,
            unroll_heads=un,
        )
        for un in (False, True)
    ]
    return np.asarray(ref), [np.asarray(o) for o in outs], S


def _check(ref, outs, S):
    for out in outs:
        assert ref.shape == out.shape
        for i in range(S):
            assert set(ref[i].tolist()) == set(out[i].tolist()), (i, ref[i], out[i])
        # padding rows past S stay -1 on both
        assert (out[S:] == -1).all() and (ref[S:] == -1).all()


def test_batched_matches_loop_small():
    _check(*_case(0, S=5, T_pad=8))


def test_batched_matches_loop_full_batch():
    _check(*_case(1, S=16, T_pad=16))


def test_batched_head_chunk_boundary():
    # H=6 with head_chunk=4 exercises the padded second chunk
    _check(*_case(2, S=7, T_pad=8))
