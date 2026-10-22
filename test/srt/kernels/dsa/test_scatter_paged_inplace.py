"""CPU interpret parity: 4D-native ``_scatter_paged_inplace`` vs jnp ``_scatter_paged``.

Live pages (1..N-1) must match the jnp reference bitwise; padding rows may
only ever dirty reserved page 0 (jnp collapses padding to (0, 0), the kernel
spreads it over distinct page-0 slabs to keep RMW DMAs collision-free).

Same-slab merge (fix of the verify pre-write row loss): with kv packing 2 two
consecutive positions of one request share a slab. The kernel read-modify-writes
each slab once, so ``_scatter_paged_inplace`` merges the partner rows into the
slab owner's ``[packing, D]`` payload; the second half of this file covers the
speculative-verify shapes (G consecutive positions per request, odd start, page
crossing, ragged draft-extend, the cc64 T=256 / G=4 shape) against the same
sequential oracle, plus the 8 cases of the window-kernel reference suite.
"""

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.dsa.scatter_paged_inplace import scatter_rows_inplace
from sgl_jax.srt.layers.attention.dsa_sparse_backend import (
    _dense_arm_idx_write_pages,
    _paged_landing,
    _scatter_paged,
    _scatter_paged_inplace,
)

jax.config.update("jax_platform_name", "cpu")

PAGE_SIZE = 8
PACK = 2
NUM_PAGES = 24  # >= 1 reserved + max(S) * PPS live pages: seqs must never share a page
D = 128
PPS = 4  # pages_per_seq


def _mk_batch(rng, seq_lens, q_lens, T_pad):
    S = len(seq_lens)
    cu_q = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
    cu_kv = np.concatenate([[0], np.cumsum(np.array([PPS] * S) * PAGE_SIZE)]).astype(np.int32)
    page_indices = rng.permutation(np.arange(1, NUM_PAGES))[: S * PPS].astype(np.int32)
    new_tokens = rng.standard_normal((T_pad, D)).astype(np.float32)
    return (
        jnp.asarray(new_tokens),
        jnp.asarray(np.array(seq_lens, np.int32)),
        jnp.asarray(page_indices),
        jnp.asarray(cu_q),
        jnp.asarray(cu_kv),
    )


def _check_parity(cache4d, new_tokens, seq_lens, page_indices, cu_q, cu_kv):
    cache3d = cache4d.reshape(NUM_PAGES, PAGE_SIZE, D)
    ref = np.asarray(
        _scatter_paged(cache3d, new_tokens, seq_lens, page_indices, cu_q, cu_kv, PPS)
    ).reshape(cache4d.shape)
    out = np.asarray(
        _scatter_paged_inplace(cache4d, new_tokens, seq_lens, page_indices, cu_q, cu_kv, PPS)
    )
    base = np.asarray(cache4d)
    np.testing.assert_array_equal(out[1:], ref[1:])
    live_dirty = (out[1:] != base[1:]) & (ref[1:] == base[1:])
    assert not live_dirty.any(), "in-place scatter dirtied a live page the ref did not"


def _cache(rng):
    return jnp.asarray(
        rng.standard_normal((NUM_PAGES, PAGE_SIZE // PACK, PACK, D)).astype(np.float32)
    )


def test_decode_one_token_per_seq():
    rng = np.random.default_rng(0)
    seq_lens = [5, PAGE_SIZE * PPS, 1, 9, 2]  # incl. exactly-full, fresh, odd/even slots
    new_tokens, sl, pi, cuq, cukv = _mk_batch(rng, seq_lens, [1] * 5, T_pad=8)
    _check_parity(_cache(rng), new_tokens, sl, pi, cuq, cukv)


def test_padding_and_invalid_seq():
    rng = np.random.default_rng(1)
    seq_lens = [7, 12, 0]  # kv_len == 0 seq is invalid
    new_tokens, sl, pi, cuq, cukv = _mk_batch(rng, seq_lens, [1, 1, 0], T_pad=16)
    _check_parity(_cache(rng), new_tokens, sl, pi, cuq, cukv)


def test_both_rows_of_a_slab_across_seqs():
    rng = np.random.default_rng(2)
    # positions 4 (pair 0) and 5 (pair 1) on different seqs' pages: same sub, different slabs
    new_tokens, sl, pi, cuq, cukv = _mk_batch(rng, [5, 6], [1, 1], T_pad=2)
    _check_parity(_cache(rng), new_tokens, sl, pi, cuq, cukv)


def test_dense_arm_write_parity_pages():
    """``_dense_arm_idx_write_pages`` (exact-dense prefill arm key write) vs ``_scatter_paged``: sequences starting
    mid-page (chunked continuation), page-aligned ones, a padded/empty one; live pages bitwise equal.
    """
    for seed, seq_lens, q_lens in [
        (
            5,
            [PAGE_SIZE * 3, PAGE_SIZE * 2 + 3, PAGE_SIZE * 4, 0],
            [PAGE_SIZE * 3, PAGE_SIZE, PAGE_SIZE * 2, 0],
        ),
        (6, [PAGE_SIZE * 2 + 5, PAGE_SIZE * 4, 7, 0], [PAGE_SIZE + 2, PAGE_SIZE * 4, 7, 0]),
        (7, [PAGE_SIZE * 4, PAGE_SIZE * 4, PAGE_SIZE * 4, PAGE_SIZE * 4], [PAGE_SIZE * 4] * 4),
        # every sequence straddles a page boundary with a 2-token extend: 2 pages per sequence,
        # 8 pages for 8 tokens (the page-unit budget must be T // page_size + 2 * S, not + S)
        (8, [PAGE_SIZE + 1] * 4, [2] * 4),
        (9, [PAGE_SIZE * 2 + 5] * 4, [6] * 4),
    ]:
        rng = np.random.default_rng(seed)
        cache4d = _cache(rng)
        T_pad = sum(q_lens) + (6 if seed < 8 else 0)
        new_tokens, sl, pi, cu_q, cu_kv = _mk_batch(rng, seq_lens, q_lens, T_pad)
        cache3d = cache4d.reshape(NUM_PAGES, PAGE_SIZE, D)
        ref = np.asarray(_scatter_paged(cache3d, new_tokens, sl, pi, cu_q, cu_kv, PPS)).reshape(
            cache4d.shape
        )
        out = np.asarray(
            jax.jit(
                lambda c, nt, a, b, cq, ck: _dense_arm_idx_write_pages(
                    c, nt, PAGE_SIZE, a, b, cq, ck, PPS
                )
            )(cache4d, new_tokens, sl, pi, cu_q, cu_kv)
        )
        base = np.asarray(cache4d)
        np.testing.assert_array_equal(out[1:], ref[1:])
        assert not ((out[1:] != base[1:]) & (ref[1:] == base[1:])).any(), seed


# ---------------------------------------------------------------------------
# Same-slab merge: speculative-verify shapes (G consecutive positions per request).
# Oracle = jnp ``_scatter_paged`` on the 3D view (sequential semantics), bf16 cache.
# ---------------------------------------------------------------------------

V_PAGE_SIZE = 16
V_S = V_PAGE_SIZE // PACK


def _mk_verify_batch(rng, seq_lens, q_lens, T_pad, page_size, pps, num_pages):
    R = len(seq_lens)
    assert sum(q_lens) <= T_pad
    cu_q = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
    cu_kv = np.concatenate([[0], np.cumsum(np.array([pps] * R) * page_size)]).astype(np.int32)
    assert R * pps <= num_pages - 1, "requests must never share a page"
    page_indices = rng.permutation(np.arange(1, num_pages))[: R * pps].astype(np.int32)
    new_tokens = rng.standard_normal((T_pad, D)).astype(np.float32)
    cache4d = jnp.asarray(
        rng.standard_normal((num_pages, page_size // PACK, PACK, D)), jnp.bfloat16
    )
    return (
        cache4d,
        jnp.asarray(new_tokens, jnp.bfloat16),
        jnp.asarray(np.array(seq_lens, np.int32)),
        jnp.asarray(page_indices),
        jnp.asarray(cu_q),
        jnp.asarray(cu_kv),
    )


def _check_verify(rng, seq_lens, q_lens, T_pad, page_size=V_PAGE_SIZE, pps=4, num_pages=40):
    cache4d, new_tokens, sl, pi, cu_q, cu_kv = _mk_verify_batch(
        rng, seq_lens, q_lens, T_pad, page_size, pps, num_pages
    )
    cache3d = cache4d.reshape(num_pages, page_size, D)
    ref = np.asarray(_scatter_paged(cache3d, new_tokens, sl, pi, cu_q, cu_kv, pps)).reshape(
        cache4d.shape
    )
    out = np.asarray(_scatter_paged_inplace(cache4d, new_tokens, sl, pi, cu_q, cu_kv, pps))
    base = np.asarray(cache4d)
    # live pages bitwise equal to the sequential reference
    np.testing.assert_array_equal(out[1:], ref[1:])
    # nothing on a live page dirtied that the reference did not write
    live_dirty = (out[1:] != base[1:]) & (ref[1:] == base[1:])
    assert not live_dirty.any()
    # merged partners / padding fetch and store reserved page 0 unchanged
    np.testing.assert_array_equal(out[0], base[0])
    # every valid token row landed (the defect was a lost row per shared slab)
    page, off, valid = (
        np.asarray(a) for a in _paged_landing(T_pad, page_size, sl, pi, cu_q, cu_kv)
    )
    rows = np.asarray(new_tokens)
    for t in np.flatnonzero(valid):
        np.testing.assert_array_equal(out[page[t], off[t] // PACK, off[t] % PACK], rows[t])
    return out


def test_verify_aligned_pairs():
    # kv_len - 4 even for every request: positions (2k, 2k+1), (2k+2, 2k+3) -> two full slabs
    rng = np.random.default_rng(10)
    _check_verify(rng, [8, 12, 20, 36, 4], [4] * 5, T_pad=20)


def test_verify_odd_start():
    # odd start: partial (row 1 only) + full + partial (row 0 only) slab per request
    rng = np.random.default_rng(11)
    _check_verify(rng, [9, 13, 21, 37, 5], [4] * 5, T_pad=20)


def test_verify_page_crossing():
    # positions [14,18) / [15,19) straddle the 16-token page boundary
    rng = np.random.default_rng(12)
    _check_verify(rng, [18, 19, 17, 20 + V_PAGE_SIZE, 15 + V_PAGE_SIZE], [4] * 5, T_pad=20)


def test_verify_mixed_lengths():
    # ragged: 1..4 new tokens per request (draft-extend / partially accepted), odd+even starts
    rng = np.random.default_rng(13)
    _check_verify(rng, [7, 12, 23, 30, 19, 1, 2], [1, 2, 3, 4, 2, 1, 1], T_pad=16)


def test_verify_cc64_shape_t256_g4():
    # production cc64 verify: T = 256 = 64 requests x 4 consecutive positions, page 64, bf16
    rng = np.random.default_rng(14)
    page_size, pps, R, G = 64, 4, 64, 4
    seq_lens = rng.integers(G, page_size * pps + 1, size=R).tolist()
    seq_lens[0] = page_size + 2  # crosses page 0->1 of the request: [62, 66)
    seq_lens[1] = page_size + 3  # odd start crossing
    seq_lens[2] = G  # fresh request, positions [0, 4)
    seq_lens[3] = page_size * pps  # exactly full
    _check_verify(
        rng, seq_lens, [G] * R, T_pad=R * G, page_size=page_size, pps=pps, num_pages=R * pps + 1
    )


def test_verify_cc64_shape_with_padding_requests():
    # cc64 shape with padded requests (kv_len 0 -> invalid rows) and padding tail rows
    rng = np.random.default_rng(15)
    page_size, pps, R, G = 64, 4, 60, 4
    seq_lens = rng.integers(G, page_size * pps + 1, size=R).tolist()
    seq_lens[5] = 0
    seq_lens[17] = 0
    _check_verify(
        rng, seq_lens, [G] * R, T_pad=256, page_size=page_size, pps=pps, num_pages=R * pps + 1
    )


def test_per_token_slots_lose_shared_slab_rows_kernel_contract():
    """Documents the contract the fix relies on: feeding the kernel one entry per token
    (the pre-fix slot construction, two entries on one slab) loses a row per shared slab,
    while the merged entries from ``_scatter_paged_inplace`` land all of them."""
    rng = np.random.default_rng(16)
    seq_lens = [8, 12, 20, 36, 4]
    T_pad = 20
    cache4d, new_tokens, sl, pi, cu_q, cu_kv = _mk_verify_batch(
        rng, seq_lens, [4] * 5, T_pad, V_PAGE_SIZE, 4, 40
    )
    page, off, valid = _paged_landing(T_pad, V_PAGE_SIZE, sl, pi, cu_q, cu_kv)
    t = jnp.arange(T_pad)
    pair = off % PACK
    per_token_slots = jnp.stack(
        [
            jnp.where(valid, page, 0),
            jnp.where(valid, off // PACK, t % V_S),
            jnp.where(valid, pair, 0),
            jnp.where(valid, pair + 1, 0),
        ]
    ).astype(jnp.int32)
    per_token_payload = jnp.broadcast_to(new_tokens[:, None, :], (T_pad, PACK, D))
    out_old = np.asarray(
        scatter_rows_inplace(per_token_slots, per_token_payload, cache4d, interpret=True)
    )
    p, o, v = (np.asarray(a) for a in (page, off, valid))
    rows = np.asarray(new_tokens)
    landed_old = sum(
        np.array_equal(out_old[p[i], o[i] // PACK, o[i] % PACK], rows[i])
        for i in range(T_pad)
        if v[i]
    )
    assert landed_old < int(v.sum()), "two entries per slab unexpectedly landed every row"
    out_new = np.asarray(_scatter_paged_inplace(cache4d, new_tokens, sl, pi, cu_q, cu_kv, 4))
    landed_new = sum(
        np.array_equal(out_new[p[i], o[i] // PACK, o[i] % PACK], rows[i])
        for i in range(T_pad)
        if v[i]
    )
    assert landed_new == int(v.sum())


# --- the 8 cases of the window-kernel reference suite (wt-k5b
# test_scatter_window_inplace.py), run against the merged per-slab path ---


def test_k5b_verify_aligned_even_start():
    rng = np.random.default_rng(0)
    _check_verify(rng, [8, 12, 20, 36, 4], [4] * 5, T_pad=20)


def test_k5b_verify_odd_start():
    rng = np.random.default_rng(1)
    _check_verify(rng, [9, 13, 21, 37, 5], [4] * 5, T_pad=20)


def test_k5b_verify_page_crossing():
    rng = np.random.default_rng(2)
    _check_verify(rng, [18, 19, 17, 20 + V_PAGE_SIZE, 15 + V_PAGE_SIZE], [4] * 5, T_pad=20)


def test_k5b_verify_padding_requests_and_tail():
    rng = np.random.default_rng(3)
    _check_verify(rng, [10, 0, 27, 0], [4, 4, 4, 4], T_pad=24)


def test_k5b_draft_extend_variable_n():
    rng = np.random.default_rng(4)
    _check_verify(rng, [7, 12, 23, 30, 16 + 3], [1, 2, 3, 4, 2], T_pad=16)


def test_k5b_decode_one_token_per_request():
    rng = np.random.default_rng(5)
    _check_verify(rng, [5, V_PAGE_SIZE * 4, 1, 9, 2, 16], [1] * 6, T_pad=8)


def test_k5b_block_eight():
    # 8 consecutive positions per request (block_max 8 in the window suite): 4-5 slabs per request
    rng = np.random.default_rng(6)
    _check_verify(rng, [9, 30, 41], [8, 5, 8], T_pad=24)


def test_k5b_shared_slab_rows_all_landed():
    # window-suite case 8 (per-token kernel loses shared-slab rows): the merged path lands all 20
    rng = np.random.default_rng(7)
    seq_lens = [8, 12, 20, 36, 4]
    out = _check_verify(rng, seq_lens, [4] * 5, T_pad=20)
    assert out.shape == (40, V_S, PACK, D)


def test_verify_pseudo_decode_metadata_defeats_merge_request_metadata_lands_all():
    """#1811 review (DP-attend grouped verify): the decode-form verify path hands the kernel
    per-token pseudo-decode metadata (every token its own one-query sequence). Fed to
    ``_scatter_paged_inplace`` that metadata makes every token a slab owner, so the two tokens
    of one request sharing a packed slab become two read-modify-writes of that slab and one
    row is reverted. The replica pre-write therefore has to use the ORIGINAL request
    metadata (``_run_sparse(prewrite_md=md)``), which merges them and lands every row."""
    from sgl_jax.srt.layers.attention.dsa_sparse_backend import (
        _spec_pseudo_decode_metadata,
    )

    rng = np.random.default_rng(17)
    seq_lens = [8, 12, 20, 36, 4, 9, 15, 33]
    G = 4
    T_pad = 32
    cache4d, new_tokens, sl, pi, cu_q, cu_kv = _mk_verify_batch(
        rng, seq_lens, [G] * len(seq_lens), T_pad, V_PAGE_SIZE, 4, 40
    )
    page, off, valid = _paged_landing(T_pad, V_PAGE_SIZE, sl, pi, cu_q, cu_kv)
    p, o, v = (np.asarray(a) for a in (page, off, valid))
    rows = np.asarray(new_tokens)

    def landed(out):
        return sum(
            np.array_equal(out[p[i], o[i] // PACK, o[i] % PACK], rows[i])
            for i in range(T_pad)
            if v[i]
        )

    kv_len, pcu_q, pcu_kv, ppi, _ = _spec_pseudo_decode_metadata(
        sl, cu_q, cu_kv, pi, T_pad, V_PAGE_SIZE
    )
    # same landing slots either way ...
    pp, po, pv = (
        np.asarray(a) for a in _paged_landing(T_pad, V_PAGE_SIZE, kv_len, ppi, pcu_q, pcu_kv)
    )
    np.testing.assert_array_equal(pp[v], p[v])
    np.testing.assert_array_equal(po[v], o[v])
    np.testing.assert_array_equal(pv, v)
    # ... but the pseudo-sequences carry no request bounds: rows are lost
    out_pseudo = np.asarray(
        _scatter_paged_inplace(cache4d, new_tokens, kv_len, ppi, pcu_q, pcu_kv, 0)
    )
    assert landed(out_pseudo) < int(v.sum()), "per-token metadata unexpectedly landed every row"
    # the request metadata merges same-slab rows and lands all of them
    out_req = np.asarray(_scatter_paged_inplace(cache4d, new_tokens, sl, pi, cu_q, cu_kv, 4))
    assert landed(out_req) == int(v.sum())
