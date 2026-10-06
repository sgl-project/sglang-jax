"""Unit tests for the rank-3 verify-mask repack (pure numpy, no TPU).

The repack in ``flashattention_backend`` is the only production producer of the
attention kernel's ``custom_mask``, and it had no coverage at all. Its failure
mode is silent: a mis-placed row means one sequence reads another's mask, and
the model just accepts wrong draft tokens.

The invariant under test is the one the kernel relies on: **row index ==
per-DP-rank cumulative q-token index**, i.e. the same thing ``_per_dp_cumsum``
produces for ``cu_q_lens``.
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.attention.flashattention_backend import (
    FlashAttention,
    _expand_verify_tree_mask,
    _pack_verify_mask,
    _per_dp_cumsum,
    mask_row_width,
)
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode


def _build(seq_lens, q, page_size, dp_size, per_dp_bs):
    """Mimic the host-side inputs of the repack, plus a per-slot oracle."""
    seq_lens = np.asarray(seq_lens, dtype=np.int32)
    aligned = ((seq_lens + page_size - 1) // page_size) * page_size
    # Flat tree-mask layout: per slot q*kl entries, pad slots get q*(q-1).
    cm_kl = np.where(seq_lens > 0, seq_lens, q - 1).astype(np.int64)
    cm_off = np.concatenate([[0], np.cumsum(q * cm_kl)])
    cm = np.zeros(int(cm_off[-1]), dtype=np.int32)
    oracle = {}
    rng = np.random.default_rng(0)
    for s, kl in enumerate(seq_lens):
        n = int(q * cm_kl[s])
        block = rng.integers(0, 2, size=n, dtype=np.int64).astype(np.int32)
        cm[cm_off[s] : cm_off[s] + n] = block
        if kl > 0:
            oracle[s] = block.reshape(q, int(kl))
    return seq_lens, aligned, cm, cm_off, oracle


def _pack(seq_lens, q=8, page_size=128, dp_size=1, per_dp_bs=None):
    per_dp_bs = per_dp_bs if per_dp_bs is not None else len(seq_lens) // dp_size
    sl, aligned, cm, cm_off, oracle = _build(seq_lens, q, page_size, dp_size, per_dp_bs)
    packed = _pack_verify_mask(cm, sl, aligned, cm_off, q, dp_size, per_dp_bs)
    return packed, sl, aligned, oracle, per_dp_bs


@pytest.mark.parametrize(
    "w_max,expected", [(1, 128), (128, 128), (129, 256), (2048, 2048), (2049, 4096)]
)
def test_row_width_is_pow2_and_covers_max(w_max, expected):
    assert mask_row_width(np.array([w_max, 1], dtype=np.int32)) == expected


def test_row_width_always_lane_aligned_and_wide_enough():
    for lens in ([7], [900, 130], [4096, 1], [8193]):
        w = mask_row_width(np.array(lens, dtype=np.int32))
        assert w % 128 == 0
        assert w >= max(lens)


def _check_rows_match_cu_q_lens(packed, seq_lens, oracle, q, dp_size, per_dp_bs):
    """The kernel indexes a row as cu_q_lens[slot] (per-DP-rank). Assert that."""
    extend = np.where(np.asarray(seq_lens) > 0, q, 0).astype(np.int32)
    cu = _per_dp_cumsum(extend, dp_size, per_dp_bs).reshape(dp_size, per_dp_bs + 1)
    rows_per_rank = per_dp_bs * q
    seen = 0
    for r in range(dp_size):
        for j in range(per_dp_bs):
            s = r * per_dp_bs + j
            if seq_lens[s] <= 0:
                continue
            row0 = r * rows_per_rank + int(cu[r, j])
            kl = int(seq_lens[s])
            np.testing.assert_array_equal(
                packed[row0 : row0 + q, 0, :kl],
                oracle[s],
                err_msg=f"slot {s} landed at the wrong row",
            )
            seen += 1
    assert seen == sum(1 for x in seq_lens if x > 0)


def test_dense_batch_dp1():
    seq_lens = [1000, 512, 2000, 128]
    packed, sl, aligned, oracle, per_dp_bs = _pack(seq_lens)
    assert packed.shape == (len(seq_lens) * 8, 1, mask_row_width(aligned))
    assert packed.shape[2] % 128 == 0
    _check_rows_match_cu_q_lens(packed, sl, oracle, 8, 1, per_dp_bs)


def test_pad_slot_between_live_slots():
    """The desync case: a padding slot must consume no rows.

    If the packer advanced its cursor for pad slots, every sequence after the
    pad would read the previous one's mask -- and nothing else in the stack
    would notice.
    """
    seq_lens = [1000, 0, 700, 0]
    packed, sl, aligned, oracle, per_dp_bs = _pack(seq_lens)
    _check_rows_match_cu_q_lens(packed, sl, oracle, 8, 1, per_dp_bs)
    # Rows past the two live sequences must be all zero (= masked).
    assert not packed[16:].any()


def test_dp2_segments_are_independent():
    seq_lens = [1000, 0, 700, 640]  # rank0: 1 live, rank1: 2 live
    packed, sl, aligned, oracle, per_dp_bs = _pack(seq_lens, dp_size=2)
    assert packed.shape[0] == 2 * per_dp_bs * 8
    _check_rows_match_cu_q_lens(packed, sl, oracle, 8, 2, per_dp_bs)
    # Each rank owns a contiguous, equally sized block of the leading dim.
    assert packed.shape[0] % 2 == 0


def test_all_pad_batch_is_all_zero():
    packed, sl, aligned, oracle, per_dp_bs = _pack([0, 0])
    assert packed.shape[0] == 2 * 8
    assert not packed.any()


def test_columns_past_kv_len_are_masked():
    seq_lens = [300]
    packed, sl, aligned, oracle, per_dp_bs = _pack(seq_lens)
    assert not packed[:, 0, 300:].any(), "padding columns must be 0 (= masked)"


def test_width_bucket_is_stable_across_nearby_batches():
    """Shape churn guard: batches inside one power-of-two bucket share W."""
    widths = {mask_row_width(np.array([n], dtype=np.int32)) for n in (1100, 1500, 2048)}
    assert widths == {2048}


def _tree_blocks(context_lens, q, seed):
    """Per-slot ``q x q`` tree blocks and the flat full-mask layout they imply.

    Real slots get a random tree (node ``k`` hangs off some ``j < k``); row ``i``
    marks the root-to-``i`` path. Padding slots get noise in both layouts.
    """
    rng = np.random.default_rng(seed)
    blocks, full = [], []
    for ctx in context_lens:
        if ctx < 0:
            blocks.append(rng.integers(0, 2, (q, q)))
            full.append(rng.integers(0, 2, q * (q - 1)))
            continue
        parent = [0] + [int(rng.integers(0, k)) for k in range(1, q)]
        block = np.zeros((q, q), dtype=np.int64)
        for i in range(q):
            node = i
            while node:
                block[i, node] = 1
                node = parent[node]
            block[i, 0] = 1
        blocks.append(block)
        full.append(np.concatenate([np.ones((q, ctx), np.int64), block], axis=1).reshape(-1))
    return np.stack(blocks).astype(np.int32).reshape(-1), np.concatenate(full).astype(np.int32)


@pytest.mark.parametrize(
    "context_lens, q, page_size, dp_size",
    [
        ([1000, 512, -1, -1], 4, 64, 1),
        ([126, 3], 4, 1, 1),  # one row crosses the 128-lane boundary
        ([5, 300, -1, 129, -1, -1], 8, 1, 2),
        ([-1, -1], 4, 1, 1),
    ],
)
def test_device_expansion_matches_the_host_repack(context_lens, q, page_size, dp_size):
    context_lens = np.asarray(context_lens, dtype=np.int32)
    per_dp_bs = len(context_lens) // dp_size
    seq_lens = np.where(context_lens >= 0, context_lens + q, 0).astype(np.int32)
    aligned = ((seq_lens + page_size - 1) // page_size) * page_size
    tree, full = _tree_blocks(context_lens, q, seed=int(seq_lens.sum()))
    cm_off = np.concatenate([[0], np.cumsum(q * np.where(seq_lens > 0, seq_lens, q - 1))])

    expected = _pack_verify_mask(full, seq_lens, aligned, cm_off, q, dp_size, per_dp_bs)
    expanded = _expand_verify_tree_mask(
        tree, context_lens, draft_token_num=q, width=mask_row_width(aligned)
    )
    np.testing.assert_array_equal(np.asarray(expanded), expected)


def test_verify_metadata_expands_the_tree_blocks():
    """The target-verify metadata turns the tree builder's blocks into the
    kernel's rectangle, as the host repack of the full mask would."""
    q, page_size = 4, 64
    context_lens = np.array([70, 3, -1], dtype=np.int32)
    tree, full = _tree_blocks(context_lens, q, seed=1)
    mesh = Mesh(
        np.array(jax.devices()[:1]).reshape(1, 1),
        ("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 2,
    )
    backend = FlashAttention(8, 8, 128, page_size=page_size, mesh=mesh)
    batch = SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        cache_loc=np.arange(3 * 128, dtype=np.int32),
        seq_lens=np.maximum(context_lens, 0),
        logits_indices_selector=np.array([0, 1]),
        spec_info_padded=SimpleNamespace(custom_mask=jnp.asarray(tree), draft_token_num=q),
        dp_size=1,
        per_dp_bs_size=3,
    )
    metadata = backend.get_eagle_forward_metadata(batch)

    seq_lens = np.where(context_lens >= 0, context_lens + q, 0)
    aligned = ((seq_lens + page_size - 1) // page_size) * page_size
    cm_off = np.concatenate([[0], np.cumsum(q * np.where(seq_lens > 0, seq_lens, q - 1))])
    expected = _pack_verify_mask(full, seq_lens, aligned, cm_off, q, 1, 3)
    np.testing.assert_array_equal(np.asarray(metadata.custom_mask), expected)
    assert metadata.custom_mask.sharding.spec == P("data")


@pytest.mark.parametrize("page_size", [1, 64])
@pytest.mark.parametrize("draft_alloc", [4, 16])  # chain: == q; tree: steps * topk > q
def test_verify_page_table_follows_the_kv_window(page_size, draft_alloc):
    """cache_loc gives each request its allocated length, which a tree round
    makes longer than the verify window. Each request's pages must still start
    where cu_kv_lens puts them."""
    q = 4
    seq_lens = np.array([100, 50, 70, 0], dtype=np.int32)
    allocate_lens = seq_lens[:3] + draft_alloc
    bases = [8192 * (k + 1) for k in range(3)]
    aligned = -(-allocate_lens // page_size) * page_size
    cache_loc = np.concatenate([b + np.arange(n) for b, n in zip(bases, aligned)])
    cache_loc = np.pad(cache_loc, (0, 1024 - len(cache_loc))).astype(np.int32)
    mesh = Mesh(
        np.array(jax.devices()[:1]).reshape(1, 1),
        ("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 2,
    )
    backend = FlashAttention(8, 8, 128, page_size=page_size, mesh=mesh)
    batch = SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        cache_loc=cache_loc,
        seq_lens=seq_lens,
        logits_indices_selector=np.array([0, 1, 2]),
        spec_info_padded=SimpleNamespace(
            custom_mask=None, draft_token_num=q, allocate_lens=allocate_lens
        ),
        dp_size=1,
        per_dp_bs_size=4,
    )
    metadata = backend.get_eagle_forward_metadata(batch)

    pages, cu_kv_lens = np.asarray(metadata.page_indices), np.asarray(metadata.cu_kv_lens)
    for k, base in enumerate(bases):
        num_pages = -(-(int(seq_lens[k]) + q) // page_size)
        first = cu_kv_lens[k] // page_size
        np.testing.assert_array_equal(
            pages[first : first + num_pages],
            base // page_size + np.arange(num_pages),
            err_msg=f"request {k} reads another request's pages",
        )
