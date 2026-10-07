"""KPool indexer cache (GLM-5.3-Flash): pooled-key writes and token expansion.

Checks the paged write path against a direct NumPy pooling of the whole sequence
(pools spanning several forwards) and the token expansion
against the reference ``select_tokens`` rule (selected pools + unfinished tail).
"""

import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.layers.attention.dsa_sparse_backend import (
    _kpool_blocks,
    _kpool_expand,
    _kpool_write,
    kpool_indexer_dim,
)

PS, D, KP = 128, 128, 4


def _pool_ref(keys, gates, kp=KP):
    n = len(keys) // kp
    k = keys[: n * kp].reshape(n, kp, -1).astype(np.float64)
    g = gates[: n * kp].reshape(n, kp, -1).astype(np.float64)
    w = np.exp(g - g.max(1, keepdims=True))
    w /= w.sum(1, keepdims=True)
    return (w * k).sum(1)


def _new_cache(n_pages):
    from sgl_jax.srt.mem_cache.memory_pool import MLATokenToKVPool

    shape = MLATokenToKVPool._indexer_cache_shape(
        total_num_pages=n_pages,
        page_size=PS,
        dtype=jnp.bfloat16,
        indexer_key_dim=kpool_indexer_dim(D, KP),
    )
    assert shape[-1] == 128, shape  # minor dim stays 128-wide (bitcast block views)
    return jnp.zeros(shape, jnp.bfloat16)


def _write(cache, keys, gates, starts, lens, page_tables):
    """One forward: seq i appends keys[i] at positions starts[i] .. starts[i] + lens[i] - 1."""
    page_size = PS
    blocks, E, bpp = _kpool_blocks(cache, D, KP, PS)
    pages_per_seq = page_tables.shape[1]
    seq_lens = np.asarray(starts) + np.asarray(lens)
    cu_q = np.r_[0, np.cumsum(lens)].astype(np.int32)
    cu_kv = (np.arange(len(lens) + 1) * pages_per_seq * page_size).astype(np.int32)
    blocks = _kpool_write(
        blocks,
        jnp.asarray(np.concatenate(keys), jnp.bfloat16),
        jnp.asarray(np.concatenate(gates), jnp.bfloat16),
        jnp.asarray(seq_lens, jnp.int32),
        jnp.asarray(page_tables.reshape(-1), jnp.int32),
        jnp.asarray(cu_q),
        jnp.asarray(cu_kv),
        page_size=page_size,
        E=E,
        bpp=bpp,
        kpool=KP,
    )
    return blocks.reshape(cache.shape)


def _read_pools(cache, page_table, n_pools):
    blocks, E, bpp = _kpool_blocks(cache, D, KP, PS)
    blocks = blocks.reshape(blocks.shape[0], E, D)
    e = np.arange(n_pools)
    return np.asarray(blocks[page_table[e // E] * bpp, e % E], np.float64)


def test_layout_width():
    assert kpool_indexer_dim(128, 4) == 384
    assert kpool_indexer_dim(128, 1) == 128


def test_chunked_writes_match_whole_sequence_pooling():
    rng = np.random.default_rng(0)
    L = 3 * PS + 37  # crosses pages; last pool incomplete
    keys = rng.normal(size=(L, D)).astype(np.float32)
    gates = rng.normal(size=(L, D)).astype(np.float32)
    keys_b = np.asarray(jnp.asarray(keys, jnp.bfloat16), np.float32)
    gates_b = np.asarray(jnp.asarray(gates, jnp.bfloat16), np.float32)
    pt = np.array([[5, 2, 7, 3]], np.int32)
    cache = _new_cache(9)
    # chunks of odd sizes, then single-token decode steps
    bounds = [0, 5, 130, 131, 300, L - 3, L - 2, L - 1, L]
    for a, b in zip(bounds[:-1], bounds[1:]):
        cache = _write(cache, [keys[a:b]], [gates[a:b]], [a], [b - a], pt)
    n = L // KP
    got = _read_pools(cache, pt[0], n)
    want = _pool_ref(keys_b, gates_b)
    np.testing.assert_allclose(got, want, rtol=2e-2, atol=2e-2)


def _select_tokens_ref(pools, valid, qpos, kp, topk):
    """Reference (engine/glm53/model.py select_tokens) for one query."""
    tok = [p * kp + j if v else -1 for p, v in zip(pools, valid) for j in range(kp)]
    tok += [-1] * (topk - len(tok))
    vis = qpos + 1
    ts, tc = vis - vis % kp, vis % kp
    tok += [ts + j if j < tc else -1 for j in range(kp - 1)]
    return tok


def test_expand_matches_reference():
    rng = np.random.default_rng(2)
    topk = 32
    k_pools = topk // KP
    for qpos in [0, 1, 2, 3, 4, 7, 29, 30, 31, 32, 100, 1001]:
        n_vis = (qpos + 1) // KP
        sel = rng.permutation(max(n_vis, 1))[: min(k_pools, n_vis)]
        pools = np.full(k_pools, -1, np.int32)
        pools[: len(sel)] = sel
        got = np.asarray(
            _kpool_expand(
                jnp.asarray(pools[None]), jnp.asarray([qpos], jnp.int32), kpool=KP, k_tokens=topk
            )
        )[0]
        want = _select_tokens_ref(pools, pools >= 0, qpos, KP, topk)
        assert got.tolist() == want, qpos
    # padding query (seq_len 0 -> q_pos -1) selects nothing
    got = _kpool_expand(
        jnp.full((1, k_pools), -1, jnp.int32), jnp.asarray([-1]), kpool=KP, k_tokens=topk
    )
    assert (np.asarray(got) == -1).all()
