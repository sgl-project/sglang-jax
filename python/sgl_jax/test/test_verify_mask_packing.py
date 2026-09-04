"""Unit tests for the rank-3 verify-mask repack (pure numpy, no TPU).

The repack in ``flashattention_backend`` is the only production producer of the
attention kernel's ``custom_mask``, and it had no coverage at all. Its failure
mode is silent: a mis-placed row means one sequence reads another's mask, and
the model just accepts wrong draft tokens.

The invariant under test is the one the kernel relies on: **row index ==
per-DP-rank cumulative q-token index**, i.e. the same thing ``_per_dp_cumsum``
produces for ``cu_q_lens``.
"""

import numpy as np
import pytest

from sgl_jax.srt.layers.attention.flashattention_backend import (
    _pack_verify_mask,
    _per_dp_cumsum,
    mask_row_width,
)


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
