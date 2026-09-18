"""DP-attention layout regressions for speculative scheduler sequence lengths."""

import numpy as np
import pytest

from sgl_jax.srt.managers.scheduler import _split_spec_new_seq_lens


def test_split_compact_spec_sequence_lengths_across_unbalanced_dp_ranks():
    values = np.arange(31, dtype=np.int32)

    rank0, rank1 = _split_spec_new_seq_lens(
        values,
        real_bs_per_dp=[15, 16],
        per_dp_bs=32,
    )

    np.testing.assert_array_equal(rank0, np.arange(15, dtype=np.int32))
    np.testing.assert_array_equal(rank1, np.arange(15, 31, dtype=np.int32))


def test_split_padded_spec_sequence_lengths_across_unbalanced_dp_ranks():
    values = np.concatenate(
        (
            np.arange(15, dtype=np.int32),
            np.full(17, -1, dtype=np.int32),
            np.arange(100, 116, dtype=np.int32),
            np.full(16, -1, dtype=np.int32),
        )
    )

    rank0, rank1 = _split_spec_new_seq_lens(
        values,
        real_bs_per_dp=[15, 16],
        per_dp_bs=32,
    )

    np.testing.assert_array_equal(rank0, np.arange(15, dtype=np.int32))
    np.testing.assert_array_equal(rank1, np.arange(100, 116, dtype=np.int32))


def test_split_spec_sequence_lengths_rejects_unknown_layout():
    with pytest.raises(ValueError, match="neither compact nor DP-padded"):
        _split_spec_new_seq_lens(
            np.arange(30, dtype=np.int32),
            real_bs_per_dp=[15, 16],
            per_dp_bs=32,
        )


@pytest.mark.parametrize("real_bs_per_dp", ([33, 0], [-1, 1]))
def test_split_spec_sequence_lengths_rejects_invalid_dp_geometry(real_bs_per_dp):
    with pytest.raises(ValueError, match="Invalid speculative DP batch geometry"):
        _split_spec_new_seq_lens(
            np.arange(32, dtype=np.int32),
            real_bs_per_dp=real_bs_per_dp,
            per_dp_bs=32,
        )
