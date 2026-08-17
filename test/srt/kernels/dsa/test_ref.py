from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels.dsa.indexer import (
    _mask_and_compact_topk_indices,
    _map_packed_logical_topk_to_physical_slots,
    _pipeline_score_topk_and_mapping_tiles,
    _select_topk_indices,
    compute_scores_and_select_topk_indices,
)
from sgl_jax.srt.kernels.dsa.ref import (
    build_index_share_map,
    sparse_mla_ref,
    streamindex_topk_ref,
)


def test_mask_and_compact_topk_indices_preserves_unordered_valid_set():
    values = jnp.asarray([[3.0, -jnp.inf, 1.0, -jnp.inf], [4.0, 2.0, 3.0, 1.0]])
    indices = jnp.asarray([[9, 99, 5, 88], [7, 2, 6, 4]], dtype=jnp.int32)

    got = np.asarray(jax.jit(_mask_and_compact_topk_indices)(values, indices))

    np.testing.assert_array_equal(got[0], np.asarray([9, 5, -1, -1]))
    np.testing.assert_array_equal(got[1], np.asarray([7, 2, 6, 4]))


@pytest.mark.parametrize(
    ("device_count", "expected_query_block_size"),
    [(1, 16), (16, 16), (31, 16), (32, 24), (64, 24)],
)
def test_default_extend_query_block_size_uses_global_device_count(
    monkeypatch, device_count, expected_query_block_size
):
    import sgl_jax.srt.kernels.dsa.indexer as indexer

    monkeypatch.setattr(indexer.jax, "device_count", lambda: device_count)

    assert indexer._default_extend_query_block_size() == expected_query_block_size


@pytest.mark.parametrize("num_tiles", [0, 1, 2, 3, 6])
def test_score_topk_mapping_pipeline_fill_steady_state_and_drain(num_tiles):
    max_tiles, rows, width, topk = 6, 2, 4, 2

    def run(runtime_num_tiles):
        def score_tile(tile_id):
            row = jnp.arange(rows, dtype=jnp.int32)[:, None]
            col = jnp.arange(width, dtype=jnp.int32)[None, :]
            return tile_id * 100 + row * 10 + col

        def select_tile(tile_id, scores):
            del tile_id
            return scores[:, :topk]

        def map_tile(tile_id, logical):
            return logical + 1000 + tile_id

        def score_and_map_tile(score_tile_id, mapping_tile_id, logical):
            return score_tile(score_tile_id), map_tile(mapping_tile_id, logical)

        def write_tile(tile_id, values, out):
            return jax.lax.dynamic_update_slice_in_dim(
                out, values[None, ...], tile_id, axis=0
            )

        return _pipeline_score_topk_and_mapping_tiles(
            runtime_num_tiles,
            score_tile,
            score_and_map_tile,
            select_tile,
            map_tile,
            write_tile,
            jnp.full((max_tiles, rows, topk), -1, jnp.int32),
        )

    got = np.asarray(jax.jit(run)(jnp.int32(num_tiles)))
    expected = np.full((max_tiles, rows, topk), -1, np.int32)
    for tile_id in range(num_tiles):
        row = np.arange(rows, dtype=np.int32)[:, None]
        col = np.arange(topk, dtype=np.int32)[None, :]
        expected[tile_id] = tile_id * 100 + row * 10 + col + 1000 + tile_id
    np.testing.assert_array_equal(got, expected)


def test_radix_indices_only_uses_valid_lengths_without_score_gather(monkeypatch):
    import sgl_jax.srt.kernels.dsa.indexer as indexer

    selected = jnp.asarray([[2, 7, 0, 6], [7, 1, 4, 3]], dtype=jnp.int32)
    monkeypatch.setattr(
        indexer,
        "select_indexer_radix_topk_indices",
        lambda scores, *, k: selected,
    )

    got = _select_topk_indices(
        jnp.zeros((2, 8), dtype=jnp.float32),
        jnp.asarray([3, 8], dtype=jnp.int32),
        k=4,
        topk_impl="radix",
    )

    np.testing.assert_array_equal(np.asarray(got[0]), np.asarray([2, 0, -1, -1]))
    np.testing.assert_array_equal(np.asarray(got[1]), np.asarray([7, 1, 4, 3]))


def test_packed_logical_topk_mapping_runs_after_selection_for_ragged_sequences():
    logical_topk = jnp.asarray(
        [
            [0, 4, 5],
            [1, 5, -1],
            [0, 7, 8],
            [2, 8, 9],
            [0, 1, 2],
        ],
        dtype=jnp.int32,
    )

    got = _map_packed_logical_topk_to_physical_slots(
        logical_topk,
        jnp.asarray([5, 2, 9, 7, 1, 4], dtype=jnp.int32),
        jnp.asarray([6, 9], dtype=jnp.int32),
        jnp.asarray([0, 2, 4], dtype=jnp.int32),
        jnp.asarray([0, 12, 24], dtype=jnp.int32),
        jnp.int32(2),
        pages_per_seq=3,
        page_size=4,
    )

    np.testing.assert_array_equal(
        np.asarray(got),
        np.asarray(
            [
                [20, 8, -1],
                [21, 9, -1],
                [28, 7, -1],
                [30, 16, -1],
                [-1, -1, -1],
            ],
            dtype=np.int32,
        ),
    )


jax.config.update("jax_platform_name", "cpu")


def _glm52_indexer_types():
    types = ["full"] * 3
    for _ in range(75 // 4):
        types += ["shared"] * 3 + ["full"]
    types += ["shared"] * (78 - len(types))
    return types


def test_build_index_share_map_glm52():
    """GLM-5.2 pattern: [full×3, shared×3, full, shared×3, ...] for 78 layers."""
    types = _glm52_indexer_types()
    assert len(types) == 78

    full_slot, src_slot, num_full = build_index_share_map(
        types, skip_offset=3, num_layers=78
    )

    assert num_full == types.count("full")
    assert full_slot[0] == 0 and full_slot[1] == 1 and full_slot[2] == 2
    assert src_slot[3] == 2 and src_slot[4] == 2 and src_slot[5] == 2
    assert full_slot[6] == 3
    assert src_slot[7] == 3
    assert len(src_slot) == 78


def test_build_index_share_map_none_is_all_full():
    full_slot, src_slot, num_full = build_index_share_map(
        None, skip_offset=0, num_layers=4
    )
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


def test_streamindex_topk_ref_matches_numpy_exactly():
    rng = np.random.default_rng(0)
    T, H, D, KV, page_size, k = 4, 2, 8, 32, 8, 5
    q = rng.normal(size=(T, H, D)).astype(np.float32)
    weights = rng.normal(size=(T, H)).astype(np.float32)
    keys = rng.normal(size=(KV, D)).astype(np.float32)
    cache, page_idx = _make_paged(keys, page_size)

    got = np.asarray(
        streamindex_topk_ref(
            jnp.asarray(q),
            jnp.asarray(weights),
            jnp.asarray(cache),
            jnp.asarray([KV], np.int32),
            jnp.asarray(page_idx),
            jnp.asarray([0, T], np.int32),
            jnp.asarray([0, cache.shape[0] * page_size], np.int32),
            jnp.asarray([0, 1, 1], np.int32),
            k=k,
            pages_per_seq=cache.shape[0],
        )
    )

    for token_id in range(T):
        abs_pos = KV - T + token_id
        scores = np.einsum(
            "h,hk->k",
            weights[token_id],
            np.maximum(np.einsum("hd,kd->hk", q[token_id], keys), 0),
        )
        scores[abs_pos + 1 :] = -np.inf
        np.testing.assert_array_equal(got[token_id], np.argsort(-scores)[:k])


def test_compute_scores_and_select_topk_indices_extend_matches_numpy():
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
        compute_scores_and_select_topk_indices(
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
            topk_impl="approx",
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
        got_t = {x for x in got[t].tolist() if x >= 0}
        recall = len(got_t & want) / max(len(want), 1)
        assert recall >= 0.9, f"t={t}: recall {recall:.2%} got {got_t} want {want}"
        assert (got[t] == -1).sum() == max(0, k - n_valid)


@pytest.mark.parametrize("score_query_block_size", [1, 14])
def test_compute_scores_and_select_topk_indices_extend_exact_topk_matches_numpy(
    score_query_block_size,
):
    rng = np.random.default_rng(11)
    T, H, D, KV, page_size, k = 3, 2, 8, 24, 8, 7
    q = rng.normal(size=(T, H, D)).astype(np.float32)
    weights = rng.normal(size=(T, H)).astype(np.float32)
    keys = rng.normal(size=(KV, D)).astype(np.float32)
    cache, page_idx = _make_paged(keys, page_size)

    got = np.asarray(
        compute_scores_and_select_topk_indices(
            jnp.array(q),
            jnp.array(weights),
            jnp.array(cache),
            jnp.array([KV], np.int32),
            jnp.array(page_idx),
            jnp.array([0, T], np.int32),
            jnp.array([0, cache.shape[0] * page_size], np.int32),
            jnp.array([0, 1, 1], np.int32),
            k=k,
            pages_per_seq=cache.shape[0],
            topk_impl="exact_lax",
            score_query_block_size=score_query_block_size,
        )
    )

    for t in range(T):
        abs_t = KV - T + t
        scores = np.einsum(
            "h,hk->k",
            weights[t],
            np.maximum(np.einsum("hd,kd->hk", q[t], keys), 0),
        )
        scores[abs_t + 1 :] = -np.inf
        expected = np.argsort(-scores)[:k]
        np.testing.assert_array_equal(got[t], expected)


def test_compute_scores_and_select_topk_indices_extend_packed_multiseq_matches_numpy():
    """Ragged extend scores each packed query once against its own sequence."""
    rng = np.random.default_rng(17)
    H, D, page_size, k = 2, 8, 4, 5
    q_lens = np.array([2, 3], np.int32)
    kv_lens = np.array([8, 12], np.int32)
    T = int(q_lens.sum())

    q = rng.normal(size=(T, H, D)).astype(np.float32)
    weights = rng.normal(size=(T, H)).astype(np.float32)
    seq_keys = [
        rng.normal(size=(int(kv_len), D)).astype(np.float32) for kv_len in kv_lens
    ]

    # Ragged-packed page table: seq0 owns two pages, seq1 owns three, and the
    # final page is static padding so page_indices.shape[0] / S == 3.
    padding_page = np.zeros((page_size, D), np.float32)
    cache = np.concatenate([*seq_keys, padding_page], axis=0).reshape(-1, page_size, D)
    page_indices = np.arange(cache.shape[0], dtype=np.int32)
    cu_q_lens = np.array([0, 2, 5], np.int32)
    cu_kv_lens = np.array([0, 8, 20], np.int32)

    got = np.asarray(
        compute_scores_and_select_topk_indices(
            jnp.asarray(q),
            jnp.asarray(weights),
            jnp.asarray(cache),
            jnp.asarray(kv_lens),
            jnp.asarray(page_indices),
            jnp.asarray(cu_q_lens),
            jnp.asarray(cu_kv_lens),
            jnp.asarray([0, 0, 2], np.int32),
            k=k,
            pages_per_seq=cache.shape[0] // len(kv_lens),
            topk_impl="exact_lax",
        )
    )
    physical = np.asarray(
        compute_scores_and_select_topk_indices(
            jnp.asarray(q),
            jnp.asarray(weights),
            jnp.asarray(cache),
            jnp.asarray(kv_lens),
            jnp.asarray(page_indices),
            jnp.asarray(cu_q_lens),
            jnp.asarray(cu_kv_lens),
            jnp.asarray([0, 0, 2], np.int32),
            k=k,
            pages_per_seq=cache.shape[0] // len(kv_lens),
            topk_impl="exact_lax",
            output_physical_slots=True,
        )
    )

    for seq_id, keys in enumerate(seq_keys):
        q_start = int(cu_q_lens[seq_id])
        q_len = int(q_lens[seq_id])
        prefix_len = int(kv_lens[seq_id] - q_len)
        for local_q in range(q_len):
            token_id = q_start + local_q
            scores = np.einsum(
                "h,hk->k",
                weights[token_id],
                np.maximum(np.einsum("hd,kd->hk", q[token_id], keys), 0),
            )
            scores[prefix_len + local_q + 1 :] = -np.inf
            expected = np.argsort(-scores)[:k]
            np.testing.assert_array_equal(got[token_id], expected)
            expected_slots = (
                page_indices[cu_kv_lens[seq_id] // page_size + expected // page_size]
                * page_size
                + expected % page_size
            )
            np.testing.assert_array_equal(physical[token_id], expected_slots)


def test_compute_scores_and_select_topk_indices_extend_multiblock_ragged_matches_numpy():
    """A sequence crossing the fixed query-block boundary is scored once."""
    rng = np.random.default_rng(19)
    H, D, page_size, k = 2, 4, 8, 4
    q_lens = np.array([257, 3], np.int32)
    kv_lens = np.array([264, 16], np.int32)
    T = int(q_lens.sum())

    q = rng.normal(size=(T, H, D)).astype(np.float32)
    weights = (np.abs(rng.normal(size=(T, H))) + 0.1).astype(np.float32)
    seq_keys = [
        rng.normal(size=(int(kv_len), D)).astype(np.float32) for kv_len in kv_lens
    ]

    pages_per_seq = 33
    total_pages = pages_per_seq * len(kv_lens)
    actual_pages = sum(int(kv_len) // page_size for kv_len in kv_lens)
    padding = np.zeros(((total_pages - actual_pages) * page_size, D), np.float32)
    cache = np.concatenate([*seq_keys, padding], axis=0).reshape(-1, page_size, D)
    page_indices = np.arange(total_pages, dtype=np.int32)
    cu_q_lens = np.array([0, 257, 260], np.int32)
    cu_kv_lens = np.array([0, 264, 280], np.int32)

    got = np.asarray(
        compute_scores_and_select_topk_indices(
            jnp.asarray(q),
            jnp.asarray(weights),
            jnp.asarray(cache),
            jnp.asarray(kv_lens),
            jnp.asarray(page_indices),
            jnp.asarray(cu_q_lens),
            jnp.asarray(cu_kv_lens),
            jnp.asarray([0, 0, 2], np.int32),
            k=k,
            pages_per_seq=pages_per_seq,
            topk_impl="exact_lax",
        )
    )

    for seq_id, keys in enumerate(seq_keys):
        q_start = int(cu_q_lens[seq_id])
        q_len = int(q_lens[seq_id])
        prefix_len = int(kv_lens[seq_id] - q_len)
        for local_q in range(q_len):
            token_id = q_start + local_q
            scores = np.einsum(
                "h,hk->k",
                weights[token_id],
                np.maximum(np.einsum("hd,kd->hk", q[token_id], keys), 0),
            )
            scores[prefix_len + local_q + 1 :] = -np.inf
            expected = np.argsort(-scores)[:k]
            np.testing.assert_array_equal(
                got[token_id], expected, err_msg=f"seq={seq_id}, token={token_id}"
            )


def test_compute_scores_and_select_topk_indices_extend_three_block_pipeline_matches_numpy():
    """The score/top-k pipeline handles fill, repeated overlap, and drain."""
    rng = np.random.default_rng(23)
    T, H, D, KV, page_size, k = 513, 2, 4, 520, 8, 7
    q = rng.normal(size=(T, H, D)).astype(np.float32)
    weights = (np.abs(rng.normal(size=(T, H))) + 0.1).astype(np.float32)
    keys = rng.normal(size=(KV, D)).astype(np.float32)
    cache, page_idx = _make_paged(keys, page_size)

    got = np.asarray(
        compute_scores_and_select_topk_indices(
            jnp.asarray(q),
            jnp.asarray(weights),
            jnp.asarray(cache),
            jnp.asarray([KV], np.int32),
            jnp.asarray(page_idx),
            jnp.asarray([0, T], np.int32),
            jnp.asarray([0, cache.shape[0] * page_size], np.int32),
            jnp.asarray([0, 0, 1], np.int32),
            k=k,
            pages_per_seq=cache.shape[0],
            topk_impl="exact_lax",
        )
    )

    prefix_len = KV - T
    for token_id in range(T):
        scores = np.einsum(
            "h,hk->k",
            weights[token_id],
            np.maximum(np.einsum("hd,kd->hk", q[token_id], keys), 0),
        )
        scores[prefix_len + token_id + 1 :] = -np.inf
        np.testing.assert_array_equal(got[token_id], np.argsort(-scores)[:k])


@pytest.mark.parametrize("active_num_seqs", [0, 1, 2, 3])
def test_compute_scores_and_select_topk_indices_decode_matches_extend(
    active_num_seqs,
):
    """Decode batched top-k matches extend scoring for one query per sequence."""
    rng = np.random.default_rng(29)
    H, D, page_size, pages_per_seq, k = 2, 8, 8, 4, 4
    kv_lens = [24, 31, 0]
    num_seqs = len(kv_lens)
    q = rng.normal(size=(num_seqs, H, D)).astype(np.float32)
    weights = np.abs(rng.normal(size=(num_seqs, H))).astype(np.float32)

    pages = []
    page_indices = []
    for kv_len in kv_lens:
        keys = rng.normal(size=(max(kv_len, 1), D)).astype(np.float32)
        seq_pages, _ = _make_paged(keys, page_size)
        padding = np.zeros(
            (pages_per_seq - seq_pages.shape[0], page_size, D),
            np.float32,
        )
        seq_pages = np.concatenate([seq_pages, padding]) if padding.size else seq_pages
        page_start = len(pages) * pages_per_seq
        pages.append(seq_pages)
        page_indices.append(
            np.arange(page_start, page_start + pages_per_seq, dtype=np.int32)
        )

    q_array = jnp.asarray(q)
    weights_array = jnp.asarray(weights)
    cache = jnp.asarray(np.concatenate(pages))
    seq_lens = jnp.asarray(kv_lens, np.int32)
    page_indices = jnp.asarray(np.concatenate(page_indices))
    cu_q_lens = jnp.asarray(np.arange(num_seqs + 1), np.int32)
    cu_kv_lens = jnp.asarray(
        [i * pages_per_seq * page_size for i in range(num_seqs + 1)],
        np.int32,
    )
    distribution = jnp.asarray([0, active_num_seqs, active_num_seqs], np.int32)
    extend = np.asarray(
        compute_scores_and_select_topk_indices(
            q_array,
            weights_array,
            cache,
            seq_lens,
            page_indices,
            cu_q_lens,
            cu_kv_lens,
            distribution,
            k=k,
            pages_per_seq=pages_per_seq,
            topk_impl="exact_lax",
        )
    )
    decode = np.asarray(
        compute_scores_and_select_topk_indices(
            q_array,
            weights_array,
            cache,
            seq_lens,
            page_indices,
            cu_q_lens,
            cu_kv_lens,
            distribution,
            k=k,
            pages_per_seq=pages_per_seq,
            one_token_per_seq=True,
            topk_impl="exact_lax",
        )
    )
    extend_slots = np.asarray(
        compute_scores_and_select_topk_indices(
            q_array,
            weights_array,
            cache,
            seq_lens,
            page_indices,
            cu_q_lens,
            cu_kv_lens,
            distribution,
            k=k,
            pages_per_seq=pages_per_seq,
            topk_impl="exact_lax",
            output_physical_slots=True,
        )
    )
    decode_slots = np.asarray(
        compute_scores_and_select_topk_indices(
            q_array,
            weights_array,
            cache,
            seq_lens,
            page_indices,
            cu_q_lens,
            cu_kv_lens,
            distribution,
            k=k,
            pages_per_seq=pages_per_seq,
            one_token_per_seq=True,
            topk_impl="exact_lax",
            output_physical_slots=True,
        )
    )

    np.testing.assert_array_equal(decode, extend)
    np.testing.assert_array_equal(decode_slots, extend_slots)
    assert (decode[-1] == -1).all()


def test_compute_scores_and_select_topk_indices_extend_skips_inactive_sequences():
    """Rows beyond distribution.num_seqs stay padded in the extend pipeline."""
    rng = np.random.default_rng(31)
    T, H, D, page_size, pages_per_seq, k = 4, 2, 4, 4, 2, 3
    q = rng.normal(size=(T, H, D)).astype(np.float32)
    weights = rng.normal(size=(T, H)).astype(np.float32)
    cache = rng.normal(size=(4, page_size, D)).astype(np.float32)

    got = np.asarray(
        compute_scores_and_select_topk_indices(
            jnp.asarray(q),
            jnp.asarray(weights),
            jnp.asarray(cache),
            jnp.asarray([8, 8], np.int32),
            jnp.asarray([0, 1, 2, 3], np.int32),
            jnp.asarray([0, 2, 4], np.int32),
            jnp.asarray([0, 8, 16], np.int32),
            jnp.asarray([0, 0, 1], np.int32),
            k=k,
            pages_per_seq=pages_per_seq,
            topk_impl="exact_lax",
        )
    )

    assert (got[:2] >= 0).all()
    assert (got[2:] == -1).all()


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
    from sgl_jax.srt.layers.attention.dsa_cache_ops import scatter_paged_cache

    P, ps, D = 4, 4, 8
    cache = jnp.zeros((P, ps, D), jnp.float32)
    seq_lens = jnp.asarray([3, 0], jnp.int32)  # seq1 = padding
    cu_q_lens = jnp.asarray([0, 1, 2], jnp.int32)  # DECODE arange
    cu_kv_lens = jnp.asarray([0, ps, ps], jnp.int32)  # seq0 aligned=4, seq1 aligned=0
    page_indices = jnp.asarray([0, 1, 2, 3], jnp.int32)
    new_tokens = jnp.asarray([[1.0] * D, [99.0] * D], jnp.float32)

    out = np.asarray(
        scatter_paged_cache(
            cache, new_tokens, seq_lens, page_indices, cu_q_lens, cu_kv_lens
        )
    )
    assert out[0, 2, 0] == 1.0  # real seq0 write
    # padding seq1 must NOT leak into any non-sentinel page
    assert not np.any(out[: P - 1] == 99.0), (
        f"leaked: {np.argwhere(out[: P - 1] == 99.0)}"
    )


def test_logical_topk_to_physical_slots_uses_ragged_page_offsets():
    from sgl_jax.srt.layers.attention.dsa_sparse_backend import (
        _logical_topk_to_physical_slots,
    )

    topk = jnp.asarray([[0, 5, 6, -1], [0, 2, -1, -1]], jnp.int32)
    seq_lens = jnp.asarray([7, 3], jnp.int32)
    cu_q = jnp.asarray([0, 1, 2], jnp.int32)
    cu_kv = jnp.asarray([0, 8, 12], jnp.int32)
    page_indices = jnp.asarray([5, 9, 3, 12], jnp.int32)

    slots, counts = _logical_topk_to_physical_slots(
        topk, seq_lens, page_indices, cu_q, cu_kv, page_size=4
    )

    np.testing.assert_array_equal(np.asarray(slots), [[20, 37, 38, 0], [12, 14, 0, 0]])
    np.testing.assert_array_equal(np.asarray(counts), [3, 2])


def test_scatter_fused_kv_matches_mla_cache_layout():
    from sgl_jax.srt.layers.attention.dsa_sparse_backend import _scatter_fused_kv_paged

    cache = jnp.zeros((3, 4, 640), jnp.bfloat16)
    latent = jnp.arange(2 * 512, dtype=jnp.float32).reshape(2, 512)
    rope = jnp.arange(2 * 64, dtype=jnp.float32).reshape(2, 64) + 2000
    out = _scatter_fused_kv_paged(
        cache,
        latent,
        rope,
        jnp.asarray([6], jnp.int32),
        jnp.asarray([1, 0, 2], jnp.int32),
        jnp.asarray([0, 2], jnp.int32),
        jnp.asarray([0, 8], jnp.int32),
        kv_lora_rank=512,
    )
    out = np.asarray(out)
    np.testing.assert_array_equal(
        out[0, 0, :512], np.asarray(latent[0].astype(jnp.bfloat16))
    )
    np.testing.assert_array_equal(
        out[0, 1, 512:576], np.asarray(rope[1].astype(jnp.bfloat16))
    )
    assert not np.any(out[0, :2, 576:])


def test_exact_dsa_pool_uses_native_flat_page_layout():
    from jax.sharding import Mesh

    from sgl_jax.srt.mem_cache.memory_pool import MLATokenToKVPool

    mesh = Mesh(np.asarray(jax.devices()).reshape(1, 1), ("data", "tensor"))
    common = {
        "size": 128,
        "page_size": 64,
        "dtype": jnp.bfloat16,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "layer_num": 1,
        "mesh": mesh,
        "indexer_key_dim": 128,
        "num_indexer_layers": 1,
    }
    mla_pool = MLATokenToKVPool(**common)
    dsa_pool = MLATokenToKVPool(**common, page_layout="flat")

    assert mla_pool.kv_buffer[0].shape == (3, 32, 2, 640)
    assert dsa_pool.kv_buffer[0].shape == (3, 64, 640)
    assert dsa_pool.indexer_key_buffer[0].shape == (3, 64, 128)
    assert dsa_pool.kv_sharding.spec == jax.sharding.PartitionSpec("data", None, None)
    assert dsa_pool.get_kv_size_bytes() == mla_pool.get_kv_size_bytes()
    assert dsa_pool.get_kv_size_bytes() == 294_912

    leaves, tree = jax.tree_util.tree_flatten(dsa_pool)
    restored = jax.tree_util.tree_unflatten(tree, leaves)
    assert restored.page_layout == "flat"
    assert restored.kv_buffer[0].shape == dsa_pool.kv_buffer[0].shape


def test_exact_dsa_flat_layout_uses_logical_page_size_without_mla_packing_padding():
    from jax.sharding import Mesh

    from sgl_jax.srt.mem_cache.memory_pool import MLATokenToKVPool

    mesh = Mesh(np.asarray(jax.devices()).reshape(1, 1), ("data", "tensor"))
    common = {
        "size": 128,
        "page_size": 1,
        "dtype": jnp.bfloat16,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "layer_num": 1,
        "mesh": mesh,
        "indexer_key_dim": 128,
        "num_indexer_layers": 1,
    }
    mla_pool = MLATokenToKVPool(**common)
    dsa_pool = MLATokenToKVPool(**common, page_layout="flat")

    assert mla_pool.kv_buffer[0].shape == (129, 1, 2, 640)
    assert dsa_pool.kv_buffer[0].shape == (129, 1, 640)
    assert dsa_pool.indexer_key_buffer[0].shape == (129, 1, 128)
    assert dsa_pool.get_kv_size_bytes() * 2 == mla_pool.get_kv_size_bytes()


def test_exact_dsa_cell_size_includes_indexer_cache():
    from sgl_jax.srt.model_executor.model_runner_kv_cache_mixin import (
        ModelRunnerKVCacheMixin,
    )

    config = SimpleNamespace(
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        index_head_dim=128,
        index_skip_topk_offset=3,
        indexer_types=_glm52_indexer_types(),
    )
    runner = SimpleNamespace(
        kv_cache_dtype=jnp.bfloat16,
        use_mla_backend=True,
        server_args=SimpleNamespace(
            attention_backend="dsa_sparse",
            dsa_sparse_impl="exact",
        ),
        model_config=SimpleNamespace(hf_text_config=config),
        page_size=64,
        _kv_pool_layer_count=lambda: 78,
    )

    main_kv_bytes = 78 * 640 * 2
    indexer_kv_bytes = 21 * 128 * 2
    assert ModelRunnerKVCacheMixin._compute_cell_size(runner) == (
        main_kv_bytes + indexer_kv_bytes
    )


def test_exact_dsa_backend_preserves_native_cache_layout(monkeypatch):
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    import sgl_jax.srt.layers.attention.dsa_sparse_backend as dsa_backend
    from sgl_jax.srt.layers.attention.mla_backend import MLAAttentionMetadata

    mesh = Mesh(np.asarray(jax.devices()).reshape(1, 1), ("data", "tensor"))

    def put(value, spec):
        return jax.device_put(value, NamedSharding(mesh, spec))

    backend = dsa_backend.DSASparseAttentionBackend(
        sparse_impl="exact",
        num_attn_heads=1,
        kv_lora_rank=4,
        qk_nope_head_dim=4,
        qk_rope_head_dim=2,
        v_head_dim=4,
        page_size=4,
        mesh=mesh,
        vmem_limit_bytes=1 << 20,
    )

    def fake_exact_attention(q, qpe, cache, slots, counts, scale, **kwargs):
        del qpe, cache, slots, counts, scale, kwargs
        return jnp.zeros_like(q)

    monkeypatch.setattr(
        dsa_backend, "sparse_core_tensor_core_dsa", fake_exact_attention
    )
    metadata = MLAAttentionMetadata(
        cu_q_lens=put(jnp.array([0, 1], jnp.int32), P("data")),
        cu_kv_lens=put(jnp.array([0, 4], jnp.int32), P("data")),
        page_indices=put(jnp.array([0], jnp.int32), P("data")),
        seq_lens=put(jnp.array([1], jnp.int32), P("data")),
        distribution=put(jnp.array([0, 1, 1], jnp.int32), P("data")),
    )
    common_args = (
        put(jnp.ones((1, 1, 4), jnp.bfloat16), P("data", "tensor", None)),
        put(jnp.ones((1, 1, 2), jnp.bfloat16), P("data", "tensor", None)),
        put(jnp.arange(4, dtype=jnp.bfloat16).reshape(1, 4), P("data", None)),
        put(jnp.arange(2, dtype=jnp.bfloat16).reshape(1, 2), P("data", None)),
    )
    topk = put(jnp.array([[0]], jnp.int32), P("data", None))

    with jax.set_mesh(mesh):
        flat = put(jnp.zeros((2, 4, 256), jnp.bfloat16), P("data", None, None))
        packed = put(
            jnp.zeros((2, 2, 2, 256), jnp.bfloat16),
            P("data", None, None, None),
        )
        _, flat_out = backend._run_exact(
            *common_args, flat, topk, 1.0, "data", metadata
        )
        _, packed_out = backend._run_exact(
            *common_args, packed, topk, 1.0, "data", metadata
        )

    assert flat_out.shape == flat.shape
    assert packed_out.shape == packed.shape
    np.testing.assert_array_equal(
        np.asarray(flat_out), np.asarray(packed_out).reshape(flat.shape)
    )


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
    page_idx = np.array(
        [0, 1, 2, 3], np.int32
    )  # packed: seq0@[0:2], seq1@[2:3], pad@[3:]
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
