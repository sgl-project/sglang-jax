import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.models.minimax_m3 import msa_block_topk


def _ref_msa_block_topk(iq, ik_hist, seq_len, q_pos, block_size, topk, local_blocks):
    """NumPy reference of the corrected M3 indexer (per index head; HF
    transformers #46719, MiniMax MSA reference): block max-pool over tokens,
    local-block boost, top-k per head. Returns (list of per-head index arrays,
    n_valid)."""
    H, D = iq.shape
    L = ik_hist.shape[0]
    n_blocks = L // block_size
    scores = iq.astype(np.float64) @ ik_hist.astype(np.float64).T  # [H, L]
    scores[:, seq_len:] = -np.inf  # causal: only past tokens
    block_scores = scores.reshape(H, n_blocks, block_size).max(-1)  # [H, n_blocks]
    q_block = q_pos // block_size
    for j in range(local_blocks):
        block_scores[:, max(q_block - j, 0)] = np.inf
    k = min(topk, n_blocks)
    n_valid = min((seq_len + block_size - 1) // block_size, topk)
    out = []
    for h in range(H):
        idx = np.argpartition(-block_scores[h], k - 1)[:k]
        idx = idx[np.argsort(-block_scores[h][idx])]
        out.append(idx[:n_valid])
    return out, n_valid


def _run(iq, ik, seq_len, q_pos, B, K, LB):
    return jax.jit(msa_block_topk, static_argnames=("block_size", "topk", "local_blocks"))(
        jnp.asarray(iq), jnp.asarray(ik), seq_len, q_pos, block_size=B, topk=K, local_blocks=LB
    )


@pytest.mark.unit
@pytest.mark.parametrize("seq_len,q_pos", [(384, 383), (1024, 1023), (130, 129)])
def test_msa_block_topk_matches_ref_per_head(seq_len, q_pos):
    rng = np.random.default_rng(42 + seq_len)
    H, D, L_pad, B, K, LB = 4, 128, 1024, 128, 16, 1
    iq = rng.standard_normal((H, D)).astype(np.float32)
    ik = rng.standard_normal((L_pad, D)).astype(np.float32)
    ik[seq_len:] = 0
    out_idx, out_nv = _run(iq, ik, seq_len, q_pos, B, K, LB)
    ref, ref_nv = _ref_msa_block_topk(iq, ik, seq_len, q_pos, B, K, LB)
    assert out_idx.shape == (H, K)
    assert int(out_nv) == ref_nv
    for h in range(H):
        got = sorted(np.asarray(out_idx)[h, :ref_nv].tolist())
        assert got == sorted(ref[h].tolist()), f"head {h}: jax={got} ref={sorted(ref[h].tolist())}"


@pytest.mark.unit
def test_msa_per_head_selection_not_shared():
    """Heads with distinct preferences must get distinct block sets; a
    head-collapsed (max over heads) selection would hand every head the same
    list and drop each head's own preferred blocks (HF #46762 symptom)."""
    H, D, B, LB = 4, 64, 128, 1
    n_blocks, K = 18, 16
    L = n_blocks * B
    ik = np.zeros((L, D), np.float32)
    # block b carries a unit key along dim b (b < D); block 0..17 -> dims 0..17
    for b in range(n_blocks):
        ik[b * B : (b + 1) * B, b] = 1.0
    # head h strongly prefers blocks {h, h+4, h+8, h+12} (disjoint across heads)
    # and mildly dislikes everything else, so the sets cannot coincide.
    iq = np.full((H, D), -0.1, np.float32)
    for h in range(H):
        iq[h, [h, h + 4, h + 8, h + 12]] = 1.0
    seq_len, q_pos = L, L - 1
    out_idx, out_nv = _run(iq, ik, seq_len, q_pos, B, K, LB)
    ref, ref_nv = _ref_msa_block_topk(iq, ik, seq_len, q_pos, B, K, LB)
    assert int(out_nv) == ref_nv == K
    sel = [set(np.asarray(out_idx)[h, :K].tolist()) for h in range(H)]
    for h in range(H):
        assert sel[h] == set(ref[h].tolist()), f"head {h}"
        assert {h, h + 4, h + 8, h + 12} <= sel[h], f"head {h} lost its preferred blocks"
    # distinct heads, distinct selections (K=16 of 18 blocks leaves room to differ)
    assert len({frozenset(x) for x in sel}) > 1
    # and the head-collapsed variant is NOT what per-head selection produces
    shared = set(np.argsort(-(iq @ ik.T).reshape(H, n_blocks, B).max(-1).max(0))[:K].tolist()) | {
        q_pos // B
    }
    assert any(sel[h] != shared for h in range(H))


@pytest.mark.unit
def test_msa_degenerate_selects_all():
    """seq_len <= topk*block_size: topk should select all valid blocks (= dense), every head."""
    rng = np.random.default_rng(7)
    iq = rng.standard_normal((4, 128)).astype(np.float32)
    ik = rng.standard_normal((2048, 128)).astype(np.float32)
    seq_len = 640  # 5 blocks < topk=16
    out_idx, out_nv = msa_block_topk(
        jnp.asarray(iq),
        jnp.asarray(ik),
        seq_len,
        seq_len - 1,
        block_size=128,
        topk=16,
        local_blocks=1,
    )
    assert int(out_nv) == 5
    for h in range(4):
        assert sorted(np.asarray(out_idx)[h, :5].tolist()) == [0, 1, 2, 3, 4]


@pytest.mark.unit
def test_msa_local_block_always_selected():
    rng = np.random.default_rng(11)
    iq = rng.standard_normal((4, 128)).astype(np.float32)
    ik = rng.standard_normal((4096, 128)).astype(np.float32)
    ik[3000:] = 0
    q_pos = 2999
    out_idx, _ = msa_block_topk(
        jnp.asarray(iq), jnp.asarray(ik), 3000, q_pos, block_size=128, topk=16, local_blocks=1
    )
    for h in range(4):
        assert (q_pos // 128) in np.asarray(out_idx)[h].tolist(), f"head {h}"


@pytest.mark.unit
@pytest.mark.parametrize(
    "seq_lens",
    [
        [59969],  # bs=1: cumsum offset=0, reshape would also work
        [40, 59969],  # bs=2 short+long
        [59000, 59969],  # bs=2 long+long (different aligned page counts)
        [2048] * 8 + [59969],  # bs=9 cc=8-style
    ],
)
def test_pi_2d_ragged_layout(seq_lens):
    """flashattention_backend.py _msa_inner pi_2d construction.

    page_indices from schedule_batch._merge_cache_loc is cumsum-packed ragged
    (per-req start = cumsum(aligned_lens[:r])), NOT [bs, P] rectangular.
    `page_indices.reshape(bs, P)` misaligns req[k>0] when seq_lens are
    heterogeneous and reads stale cache_loc_host_buf entries. The correct
    construction gathers via cu_kv_lens[:bs]//page_size offsets.
    """
    page_size, pages_per_seq = 128, 512
    bs = len(seq_lens)
    aligned = ((np.asarray(seq_lens) + page_size - 1) // page_size) * page_size
    cu_kv = np.zeros(bs + 1, dtype=np.int32)
    cu_kv[1:] = np.cumsum(aligned)
    # ragged page_indices: req r owns distinct page range [1000+r*600, ...)
    page_indices = np.full(bs * pages_per_seq, -8, dtype=np.int32)  # stale buf
    for r in range(bs):
        off = cu_kv[r] // page_size
        npg = aligned[r] // page_size
        page_indices[off : off + npg] = np.arange(1000 + r * 600, 1000 + r * 600 + npg)
    # === code under test (mirrors fa_backend.py _msa_inner pi_2d gather) ===
    cu_pages = jnp.asarray(cu_kv)[:bs] // page_size
    col = jnp.arange(pages_per_seq, dtype=jnp.int32)
    gidx = jnp.minimum(cu_pages[:, None] + col[None, :], page_indices.shape[0] - 1)
    pi_2d = np.asarray(jnp.asarray(page_indices)[gidx])
    # === verify ===
    for r in range(bs):
        npg = aligned[r] // page_size
        expect = np.arange(1000 + r * 600, 1000 + r * 600 + npg)
        np.testing.assert_array_equal(pi_2d[r, :npg], expect)
    if bs > 1 and aligned[0] // page_size != pages_per_seq:
        broken = page_indices.reshape(bs, pages_per_seq)
        assert not np.array_equal(
            broken[1, : aligned[1] // page_size], np.arange(1600, 1600 + aligned[1] // page_size)
        ), "reshape should be wrong for bs>1 ragged (regression sentinel)"
