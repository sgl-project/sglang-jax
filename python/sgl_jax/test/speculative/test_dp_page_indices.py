import numpy as np
import pytest

from sgl_jax.srt.layers.attention.flashattention_backend import _pad_page_indices


@pytest.mark.parametrize("dp_size", [1, 2, 8])
@pytest.mark.parametrize("fixed", [False, True])
def test_page_padding_preserves_each_dp_rank(dp_size, fixed):
    # 17 pages per rank require a 32-page bucket. A flat pad would move
    # rank 1's first page into rank 0's shard and leave later ranks empty.
    pages = np.arange(1, dp_size * 17 + 1, dtype=np.int32).reshape(dp_size, 17)
    padded = _pad_page_indices(
        pages.ravel(),
        max_num_seqs=dp_size,
        fixed_capacity=dp_size * 32 if fixed else None,
        dp_size=dp_size,
    ).reshape(dp_size, 32)
    np.testing.assert_array_equal(padded[:, :17], pages)
    np.testing.assert_array_equal(padded[:, 17:], 0)


def test_empty_fixed_capacity_dp_page_table():
    padded = _pad_page_indices(np.array([], dtype=np.int32), 8, 256, dp_size=8)
    np.testing.assert_array_equal(padded, np.zeros(256, dtype=np.int32))


@pytest.mark.parametrize("pages,capacity", [(np.arange(7), 16), (np.arange(8), 17)])
def test_invalid_dp_page_table_layout(pages, capacity):
    with pytest.raises(ValueError):
        _pad_page_indices(pages, 8, capacity, dp_size=8)
