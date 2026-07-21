import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels.dsa.ref import (
    build_index_share_map,
    sparse_mla_ref,
    streamindex_topk_ref,
)

jax.config.update("jax_platform_name", "cpu")


def test_build_index_share_map_glm52():
    """GLM-5.2 pattern: [full×3, shared×3, full, shared×3, ...] for 78 layers."""
    types = ["full"] * 3
    for _ in range(75 // 4):
        types += ["shared"] * 3 + ["full"]
    types += ["shared"] * (78 - len(types))
    assert len(types) == 78

    full_slot, src_slot, num_full = build_index_share_map(types, skip_offset=3, num_layers=78)

    assert num_full == types.count("full")
    assert full_slot[0] == 0 and full_slot[1] == 1 and full_slot[2] == 2
    assert src_slot[3] == 2 and src_slot[4] == 2 and src_slot[5] == 2
    assert full_slot[6] == 3
    assert src_slot[7] == 3
    assert len(src_slot) == 78


def test_build_index_share_map_none_is_all_full():
    full_slot, src_slot, num_full = build_index_share_map(None, skip_offset=0, num_layers=4)
    assert num_full == 4
    assert full_slot == {0: 0, 1: 1, 2: 2, 3: 3}
    assert src_slot == full_slot


def test_build_index_share_map_shared_first_raises():
    with pytest.raises(AssertionError):
        build_index_share_map(["shared", "full"], skip_offset=0, num_layers=2)


def _make_paged(keys_flat: np.ndarray, page_size: int):
    """Pack [N, D] into [P, page_size, D] with linear page_indices."""
    n, d = keys_flat.shape
    n_pad = ((n + page_size - 1) // page_size) * page_size
    padded = np.zeros((n_pad, d), dtype=keys_flat.dtype)
    padded[:n] = keys_flat
    pages = padded.reshape(-1, page_size, d)
    return pages, np.arange(pages.shape[0], dtype=np.int32)


def test_streamindex_topk_matches_numpy():
    rng = np.random.default_rng(0)
    T, H, D, KV, page_size, k = 4, 2, 8, 32, 8, 5
    q = rng.normal(size=(T, H, D)).astype(np.float32)
    weights = rng.normal(size=(T, H)).astype(np.float32)
    keys = rng.normal(size=(KV, D)).astype(np.float32)
    cache, page_idx = _make_paged(keys, page_size)

    seq_lens = np.array([KV], np.int32)
    cu_q = np.array([0, T], np.int32)
    dist = np.array([0, 1, 1], np.int32)

    got = np.asarray(
        streamindex_topk_ref(
            jnp.array(q),
            jnp.array(weights),
            jnp.array(cache),
            jnp.array(seq_lens),
            jnp.array(page_idx),
            jnp.array(cu_q),
            jnp.array([0, cache.shape[0] * page_size], np.int32),
            jnp.array(dist),
            k=k,
            pages_per_seq=cache.shape[0],
        )
    )

    w = weights.astype(np.float32)
    for t in range(T):
        abs_t = KV - T + t
        logits = np.maximum(np.einsum("hd,kd->hk", q[t], keys), 0)
        s = np.einsum("h,hk->k", w[t], logits)
        s[abs_t + 1 :] = -np.inf
        n_valid = abs_t + 1
        want = set(np.argsort(-s)[: min(k, n_valid)].tolist())
        got_t = set(x for x in got[t].tolist() if x >= 0)
        recall = len(got_t & want) / max(len(want), 1)
        assert recall >= 0.9, f"t={t}: recall {recall:.2%} got {got_t} want {want}"
        assert (got[t] == -1).sum() == max(0, k - n_valid)


def test_sparse_mla_full_topk_equals_dense():
    rng = np.random.default_rng(1)
    T, H, Dq, KV, page_size, v_dim = 3, 4, 16, 24, 8, 12
    q = rng.normal(size=(T, H, Dq)).astype(np.float32)
    kv = rng.normal(size=(KV, Dq)).astype(np.float32)
    cache, page_idx = _make_paged(kv, page_size)
    kv_lens = np.array([KV], np.int32)
    cu_q = np.array([0, T], np.int32)
    dist = np.array([0, 1, 1], np.int32)

    topk_full = np.tile(np.arange(KV, dtype=np.int32), (T, 1))

    o_sparse = np.asarray(
        sparse_mla_ref(
            jnp.array(q),
            jnp.array(cache),
            jnp.array(kv_lens),
            jnp.array(topk_full),
            jnp.array(page_idx),
            jnp.array(cu_q),
            jnp.array([0, cache.shape[0] * page_size], np.int32),
            jnp.array(dist),
            sm_scale=1.0,
            pages_per_seq=cache.shape[0],
            v_dim=v_dim,
        )
    )

    logits = np.einsum("thd,kd->thk", q, kv)
    p = np.exp(logits - logits.max(-1, keepdims=True))
    p = p / p.sum(-1, keepdims=True)
    o_dense = np.einsum("thk,kd->thd", p, kv[:, :v_dim])

    np.testing.assert_allclose(o_sparse, o_dense, rtol=1e-3, atol=1e-4)


def test_sparse_mla_respects_mask():
    """topk containing only position 0 → output == v[0] for every head."""
    T, H, Dq, KV, page_size, v_dim = 2, 2, 8, 16, 8, 6
    rng = np.random.default_rng(2)
    q = rng.normal(size=(T, H, Dq)).astype(np.float32)
    kv = rng.normal(size=(KV, Dq)).astype(np.float32)
    cache, page_idx = _make_paged(kv, page_size)

    topk = np.full((T, 4), -1, np.int32)
    topk[:, 0] = 0

    o = np.asarray(
        sparse_mla_ref(
            jnp.array(q),
            jnp.array(cache),
            jnp.array([KV], np.int32),
            jnp.array(topk),
            jnp.array(page_idx),
            jnp.array([0, T], np.int32),
            jnp.array([0, cache.shape[0] * page_size], np.int32),
            jnp.array([0, 1, 1], np.int32),
            sm_scale=1.0,
            pages_per_seq=cache.shape[0],
            v_dim=v_dim,
        )
    )
    for t in range(T):
        for h in range(H):
            np.testing.assert_allclose(o[t, h], kv[0, :v_dim], rtol=1e-3, atol=1e-4)


def test_scatter_paged_padding_seq_no_leak():
    """Regression: DECODE cu_q_lens=arange gives padding seqs q_len=1 but
    kv_len=0 → abs_pos=-1 → page_indices[seq*pps-1] wraps into the previous
    seq's page slots. Guard with kv_len>0 so padding writes go to sentinel."""
    from sgl_jax.srt.layers.attention.dsa_sparse_backend import _scatter_paged

    P, ps, D, pps = 4, 4, 8, 2
    cache = jnp.zeros((P, ps, D), jnp.float32)
    seq_lens = jnp.asarray([3, 0], jnp.int32)  # seq1 = padding
    cu_q_lens = jnp.asarray([0, 1, 2], jnp.int32)  # DECODE arange
    cu_kv_lens = jnp.asarray([0, ps, ps], jnp.int32)  # seq0 aligned=4, seq1 aligned=0
    page_indices = jnp.asarray([0, 1, 2, 3], jnp.int32)
    new_tokens = jnp.asarray([[1.0] * D, [99.0] * D], jnp.float32)

    out = np.asarray(
        _scatter_paged(cache, new_tokens, seq_lens, page_indices, cu_q_lens, cu_kv_lens, pps)
    )
    assert out[0, 2, 0] == 1.0  # real seq0 write
    # padding seq1 must NOT leak into any non-sentinel page
    assert not np.any(out[: P - 1] == 99.0), f"leaked: {np.argwhere(out[:P-1]==99.0)}"


def test_sparse_mla_multi_seq_packed_layout():
    """Regression: page_indices is packed at cu_kv_lens[i]//page_size (variable
    stride via cumsum(aligned_lens)), NOT seq_id*pages_per_seq. With 2 seqs of
    unequal aligned length, the fixed-stride assumption made seq1 gather seq0's
    (or padding) pages → cross-seq attention corruption at batch>1."""
    rng = np.random.default_rng(3)
    H, Dq, ps, v_dim = 2, 8, 4, 6
    # seq0: kv_len=7 (aligned=8, 2 pages); seq1: kv_len=3 (aligned=4, 1 page)
    kv0 = rng.normal(size=(7, Dq)).astype(np.float32)
    kv1 = rng.normal(size=(3, Dq)).astype(np.float32)
    q = rng.normal(size=(2, H, Dq)).astype(np.float32)  # DECODE T=2

    # 4 physical pages: page0,1=seq0; page2=seq1; page3=pad
    cache = np.zeros((4, ps, Dq), np.float32)
    cache[0, :4] = kv0[:4]
    cache[1, :3] = kv0[4:7]
    cache[2, :3] = kv1
    page_idx = np.array([0, 1, 2, 3], np.int32)  # packed: seq0@[0:2], seq1@[2:3], pad@[3:]
    seq_lens = np.array([7, 3], np.int32)
    cu_q = np.array([0, 1, 2], np.int32)
    cu_kv = np.array([0, 8, 12], np.int32)  # aligned cumsum → seq1 starts at page 2

    topk = np.array([[0, 3, 6, -1], [0, 1, 2, -1]], np.int32)  # seq-relative

    o = np.asarray(
        sparse_mla_ref(
            jnp.array(q),
            jnp.array(cache),
            jnp.array(seq_lens),
            jnp.array(topk),
            jnp.array(page_idx),
            jnp.array(cu_q),
            jnp.array(cu_kv),
            jnp.array([2, 2, 2], np.int32),
            sm_scale=1.0,
            pages_per_seq=2,
            v_dim=v_dim,
        )
    )
    # Oracle: per-seq dense attention over its own topk kv rows
    for t, (kv, tk) in enumerate([(kv0, [0, 3, 6]), (kv1, [0, 1, 2])]):
        sel = kv[tk]
        logits = np.einsum("hd,kd->hk", q[t], sel)
        p = np.exp(logits - logits.max(-1, keepdims=True))
        p = p / p.sum(-1, keepdims=True)
        want = np.einsum("hk,kd->hd", p, sel[:, :v_dim])
        np.testing.assert_allclose(o[t], want, rtol=1e-3, atol=1e-4)
