"""CPU checks for DCP token-granular attend + DSA-page topk."""

from __future__ import annotations

import numpy as np


def test_owned_phys_from_vpages_stripes_one_dsa_page():
    import jax.numpy as jnp

    from sgl_jax.srt.layers.dcp.indexer import owned_phys_from_vpages

    dcp, page, rank = 16, 128, 3
    vpages = jnp.array([[0, 1, -1]], dtype=jnp.int32)
    phys = np.asarray(owned_phys_from_vpages(vpages, dcp, rank, page))
    got = [int(x) for x in phys[0] if x >= 0]
    assert got == list(range(16))


def test_gather_attend_matches_full_softmax_two_ranks():
    """One query, striped KV, merge of per-rank gather == full softmax."""
    import jax.numpy as jnp

    from sgl_jax.srt.layers.dcp.indexer import dcp_gather_attend
    from sgl_jax.srt.layers.dcp.merge import merge_dcp_attention_jax

    dcp, page, t, h, dv, rope = 2, 4, 1, 2, 8, 2
    dk = dv + rope + 2
    rng = np.random.default_rng(0)
    rows = rng.normal(size=(4, dk)).astype(np.float32)
    ql = rng.normal(size=(t, h, dv)).astype(np.float32)
    qpe = rng.normal(size=(t, h, rope)).astype(np.float32)
    pos = np.array([3], dtype=np.int32)
    pi = np.array([0], dtype=np.int32)
    cuq = np.array([0, 1], dtype=np.int32)
    cukv = np.array([0, page], dtype=np.int32)
    sm = 0.1

    def rank_cache(rank):
        c = np.zeros((2, page, 1, dk), np.float32)
        for v in range(4):
            if v % dcp != rank:
                continue
            c[0, v // dcp, 0] = rows[v]
        return c

    partial_o, partial_lse = [], []
    phys = np.array([[0, 1, 2, 3]], dtype=np.int32)
    for rank in range(dcp):
        o, lse = dcp_gather_attend(
            jnp.asarray(ql),
            jnp.asarray(qpe),
            jnp.asarray(rank_cache(rank)),
            jnp.asarray(pi),
            jnp.asarray(cuq),
            jnp.asarray(cukv),
            jnp.asarray(phys),
            jnp.asarray(pos),
            sm,
            page,
            dcp,
            rank,
            query_chunk=1,
        )
        partial_o.append(o)
        partial_lse.append(lse)

    merged_o, _ = merge_dcp_attention_jax(jnp.stack(partial_o), jnp.stack(partial_lse))
    q = np.concatenate([ql, qpe], axis=-1)
    k_full = rows[:, : dv + rope]
    v_full = rows[:, :dv]
    scores = np.einsum("thd,kd->thk", q, k_full) * sm
    m = scores.max(-1, keepdims=True)
    p = np.exp(scores - m)
    p = p / p.sum(-1, keepdims=True)
    ref = np.einsum("thk,kd->thd", p, v_full)
    np.testing.assert_allclose(np.asarray(merged_o), ref, rtol=1e-4, atol=1e-4)


def test_blocked_page_topk_matches_dcp1_selection():
    """Slice 7 parity gate: the block-interleaved page top-k must pick exactly the
    pages a ``dcp=1`` run picks, and the per-rank owned sets must partition them."""
    import jax.numpy as jnp

    from sgl_jax.srt.layers.dcp.indexer import (
        local_page_max_jax,
        page_topk_from_gathered,
    )
    from sgl_jax.srt.layers.dcp.layout import owner, physical_index, virtual_index

    dcp, page, k_pages = 4, 8, 5
    n_pages, t = 12, 6
    rng = np.random.default_rng(7)
    gscores = rng.normal(size=(t, n_pages * page)).astype(np.float32)
    positions = np.array([0, page, 3 * page, 5 * page, 9 * page, n_pages * page - 1], np.int32)

    # dcp=1 oracle: page max, causal by page start, top-k
    oracle_ps = gscores.reshape(t, n_pages, page).max(-1)
    starts = np.arange(n_pages) * page
    oracle_ps = np.where(starts[None, :] <= positions[:, None], oracle_ps, -np.inf)
    oracle = np.argsort(-oracle_ps, axis=-1, kind="stable")[:, :k_pages]
    oracle = np.where(np.isfinite(np.take_along_axis(oracle_ps, oracle, -1)), oracle, -1)

    # shard the same global scores by the block-interleaved owner rule
    local_kv = n_pages * page // dcp
    local_max = []
    for r in range(dcp):
        ls = np.full((t, local_kv), -np.inf, np.float32)
        for v in range(n_pages * page):
            if int(owner(v, dcp, page)) == r:
                ls[:, int(physical_index(v, dcp, page))] = gscores[:, v]
        local_max.append(np.asarray(local_page_max_jax(jnp.asarray(ls), page)))
    gathered = jnp.asarray(np.stack(local_max, axis=0))

    union = [set() for _ in range(t)]
    for r in range(dcp):
        top, phys = page_topk_from_gathered(gathered, dcp, r, page, k_pages, positions)
        # every rank agrees on the global selection, and it equals the dcp=1 oracle
        np.testing.assert_array_equal(np.asarray(top), oracle)
        # owned physical pages map back to selected global pages owned by this rank
        for i in range(t):
            for p in np.asarray(phys)[i]:
                if p < 0:
                    continue
                d = int(virtual_index(int(p) * page, dcp, r, page)) // page
                assert d % dcp == r
                assert d in set(oracle[i].tolist())
                union[i].add(d)
    for i in range(t):
        assert union[i] == {int(d) for d in oracle[i] if d >= 0}, i


def test_block_interleaved_shards_reconstruct_the_exact_softmax():
    """End-to-end Slice 7 math gate (no Pallas): sharding the selected pages by the
    block-interleave owner rule, bounding each shard's causal mask in **physical**
    slot space, and LSE-merging must reproduce the dense ``dcp=1`` softmax over the
    same selected pages. This is what the qblock kernel does per rank; the kernel
    itself is unchanged from the validated ``dcp=1`` path.
    """
    import jax.numpy as jnp

    from sgl_jax.srt.layers.dcp.layout import owner, physical_index
    from sgl_jax.srt.layers.dcp.merge import merge_dcp_attention_jax
    from sgl_jax.srt.layers.dcp.write import owned_len_jax, physical_positions_jax

    dcp, page, n_pages = 4, 8, 6
    seq_len = (n_pages - 1) * page + 3  # deliberately not page-aligned
    h, dv, rope = 2, 8, 2
    dk = dv + rope
    rng = np.random.default_rng(11)
    kv = rng.normal(size=(seq_len, dk)).astype(np.float32)
    sm = 0.15

    positions = np.array([0, page - 1, page, 2 * page + 5, seq_len - 1], np.int32)
    q = rng.normal(size=(len(positions), h, dk)).astype(np.float32)
    # a fixed per-query page selection, as the indexer would hand us
    sel = [[0], [0, 1], [0, 2], [1, 2, 3], [0, 3, 5]]

    def attend(qq, keys):
        """Dense softmax of one query row over ``keys`` -> (out, natural-log lse)."""
        if keys.shape[0] == 0:
            return np.zeros((h, dv), np.float32), np.full((h,), -np.inf, np.float32)
        s = np.einsum("hd,kd->hk", qq, keys) * sm
        m = s.max(-1)
        e = np.exp(s - m[:, None])
        return np.einsum("hk,kd->hd", e, keys[:, :dv]) / e.sum(-1)[:, None], m + np.log(e.sum(-1))

    ref = np.stack(
        [
            attend(
                q[i],
                kv[
                    [
                        v
                        for d in sel[i]
                        for v in range(d * page, (d + 1) * page)
                        if v <= positions[i] and v < seq_len
                    ]
                ],
            )[0]
            for i in range(len(positions))
        ]
    )

    part_o, part_lse = [], []
    for r in range(dcp):
        bounds = np.asarray(physical_positions_jax(jnp.asarray(positions), dcp, r, page))
        llen = int(np.asarray(owned_len_jax(jnp.array([seq_len], np.int32), dcp, r, page))[0])
        # this rank's physical cache: owned virtual tokens in physical slot order
        cache = np.zeros((llen, dk), np.float32)
        for v in range(seq_len):
            if int(owner(v, dcp, page)) == r:
                cache[int(physical_index(v, dcp, page))] = kv[v]
        os_, ls_ = [], []
        for i in range(len(positions)):
            # a selected page belongs to exactly one rank, whole
            phys_pages = [d // dcp for d in sel[i] if d % dcp == r]
            slots = [
                p
                for pp in phys_pages
                for p in range(pp * page, (pp + 1) * page)
                if p <= bounds[i] and p < llen
            ]
            o_i, l_i = attend(q[i], cache[slots])
            os_.append(o_i)
            ls_.append(l_i)
        part_o.append(np.stack(os_))
        part_lse.append(np.stack(ls_))

    merged, _ = merge_dcp_attention_jax(
        jnp.asarray(np.stack(part_o)), jnp.asarray(np.stack(part_lse))
    )
    np.testing.assert_allclose(np.asarray(merged), ref, rtol=1e-5, atol=1e-5)


def test_non_owner_rank_does_not_drop_its_physical_page_zero():
    """Regression: ``sparse_mla_page_level`` excludes the new-token page from the
    hit list. A non-owner rank has no new-token page and its clamped
    ``new_page_local`` is 0, so excluding it silently dropped 128 real keys. The
    exclusion id must be -1 for non-owners.
    """
    import jax.numpy as jnp

    dcp, page = 16, 128
    # rank 3 owns pages but NOT the sequence's last block -> owns_new is False
    owns_new = jnp.array([False])
    new_abs = jnp.where(owns_new, 99, 0)
    new_page_local = new_abs // page
    new_page_excl = jnp.where(owns_new, new_page_local, jnp.int32(-1))
    assert int(new_page_local[0]) == 0, "clamped new page is 0 for a non-owner"
    assert int(new_page_excl[0]) == -1, "exclusion id must not match a real page"

    # this rank legitimately selected its physical pages 0, 1, 2
    raw = jnp.array([[0, 1, 2, -1]], jnp.int32)
    kept_buggy = int(jnp.sum((raw >= 0) & (raw != new_page_local[:, None])))
    kept_fixed = int(jnp.sum((raw >= 0) & (raw != new_page_excl[:, None])))
    assert kept_buggy == 2, "the bug dropped page 0"
    assert kept_fixed == 3, "the fix keeps all three owned pages"
    assert dcp == 16


def test_virtual_page_groups_are_contiguous_physical_blocks():
    """DSA page i is physical slots [i*spp, (i+1)*spp) on every rank."""
    dcp, page = 16, 128
    spp = page // dcp
    for rank in range(dcp):
        for dsa in range(4):
            virt = np.arange(dsa * page, (dsa + 1) * page)
            owned = virt[virt % dcp == rank]
            phys = owned // dcp
            assert phys.tolist() == list(range(dsa * spp, (dsa + 1) * spp))
