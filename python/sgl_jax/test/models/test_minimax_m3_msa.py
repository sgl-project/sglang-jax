import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.models.minimax_m3 import msa_block_topk


def _ref_msa_block_topk(iq, ik_hist, seq_len, q_pos, block_size, topk, local_blocks):
    """NumPy reference matching HF modular_minimax_m3_vl.py Indexer.forward
    (single query, single batch). Returns sorted topk block indices."""
    H, D = iq.shape
    L = ik_hist.shape[0]
    n_blocks = L // block_size
    scores = iq.astype(np.float64) @ ik_hist.astype(np.float64).T  # [H, L]
    scores[:, seq_len:] = -np.inf  # causal: only past tokens
    block_scores = scores.reshape(H, n_blocks, block_size).max(-1).max(0)  # [n_blocks]
    q_block = q_pos // block_size
    for j in range(local_blocks):
        block_scores[max(q_block - j, 0)] = np.inf
    k = min(topk, n_blocks)
    idx = np.argpartition(-block_scores, k - 1)[:k]
    idx = idx[np.argsort(-block_scores[idx])]
    n_valid = min((seq_len + block_size - 1) // block_size, topk)
    return idx[:n_valid], n_valid


@pytest.mark.unit
@pytest.mark.parametrize("seq_len,q_pos", [(384, 383), (1024, 1023), (130, 129)])
def test_msa_block_topk_matches_ref(seq_len, q_pos):
    rng = np.random.default_rng(42 + seq_len)
    H, D, L_pad, B, K, LB = 4, 128, 1024, 128, 16, 1
    iq = rng.standard_normal((H, D)).astype(np.float32)
    ik = rng.standard_normal((L_pad, D)).astype(np.float32)
    ik[seq_len:] = 0

    out_idx, out_nv = jax.jit(
        msa_block_topk, static_argnames=("block_size", "topk", "local_blocks")
    )(jnp.asarray(iq), jnp.asarray(ik), seq_len, q_pos, block_size=B, topk=K, local_blocks=LB)
    ref_idx, ref_nv = _ref_msa_block_topk(iq, ik, seq_len, q_pos, B, K, LB)

    assert int(out_nv) == ref_nv
    out_valid = sorted(np.asarray(out_idx)[:ref_nv].tolist())
    ref_valid = sorted(ref_idx.tolist())
    assert out_valid == ref_valid, f"jax={out_valid} ref={ref_valid}"


@pytest.mark.unit
def test_msa_degenerate_selects_all():
    """seq_len <= topk*block_size: topk should select all valid blocks (= dense)."""
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
    assert sorted(np.asarray(out_idx)[:5].tolist()) == [0, 1, 2, 3, 4]


@pytest.mark.unit
def test_msa_local_block_always_selected():
    rng = np.random.default_rng(9)
    iq = rng.standard_normal((4, 128)).astype(np.float32) * 0.01  # tiny scores
    ik = rng.standard_normal((4096, 128)).astype(np.float32)
    ik[3800:3900] *= 1000  # one block dominates
    seq_len, q_pos = 4096, 4095
    out_idx, _ = msa_block_topk(
        jnp.asarray(iq),
        jnp.asarray(ik),
        seq_len,
        q_pos,
        block_size=128,
        topk=16,
        local_blocks=1,
    )
    assert (q_pos // 128) in np.asarray(out_idx).tolist()


def _pooled_topk(iq, ik_hist, seq_len, *, block_size, topk, local_blocks):
    n_blocks = ik_hist.shape[0] // block_size
    ik_pooled = ik_hist.reshape(n_blocks, block_size, -1).max(1)
    bscores = np.einsum("hd,nd->hn", iq, ik_pooled).max(0)
    n_valid = (seq_len + block_size - 1) // block_size
    bscores[n_valid:] = -np.inf
    q_block = (seq_len - 1) // block_size
    for j in range(local_blocks):
        bscores[max(q_block - j, 0)] = np.inf
    return set(np.argsort(-bscores)[:topk].tolist())


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


def test_pooled_topk_recovers_outlier_block():
    """P0 approximation: element-wise-max-pooled ik must still rank a block
    containing one strongly-correlated outlier token (the needle) into topk.
    This is the property needle@64K relies on, not mean-overlap."""
    rng = np.random.default_rng(42)
    H, D, BS, L, TOPK = 4, 128, 128, 8192, 16
    for trial in range(10):
        iq = rng.standard_normal((H, D)).astype(np.float32)
        ik = rng.standard_normal((L, D)).astype(np.float32) * 0.3
        needle_tok = int(rng.integers(0, L - BS))
        ik[needle_tok] = iq[0] * 5.0  # strongly correlated with q (the needle)
        exact_idx, _ = msa_block_topk(
            jnp.asarray(iq),
            jnp.asarray(ik),
            jnp.int32(L),
            jnp.int32(L - 1),
            block_size=BS,
            topk=TOPK,
            local_blocks=1,
        )
        exact = set(np.asarray(exact_idx).tolist())
        pooled = _pooled_topk(iq, ik, L, block_size=BS, topk=TOPK, local_blocks=1)
        needle_block = needle_tok // BS
        assert needle_block in exact, f"trial {trial}: exact missed needle (sanity)"
        assert needle_block in pooled, (
            f"trial {trial}: pooled approx missed needle block {needle_block}, "
            f"overlap={len(exact & pooled)}/{TOPK}"
        )
