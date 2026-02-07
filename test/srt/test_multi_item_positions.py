from types import SimpleNamespace

import numpy as np
import pytest

from sgl_jax.srt.managers.schedule_batch import (
    _build_extend_positions_for_req,
    _build_multi_item_extend_positions,
)


def test_build_multi_item_extend_positions_resets_each_item_block():
    delimiter = 99
    # query=[10, 11], then <d>item1(2 tokens)<d>item2(1 token)<d>
    tokens = np.array([10, 11, 99, 21, 22, 99, 31, 99], dtype=np.int32)

    positions = _build_multi_item_extend_positions(tokens, delimiter, np.int32)

    # query positions: 0,1
    # first delimiter resets to query_len=2
    # item1: 3,4
    # second delimiter resets to 2
    # item2: 3
    # final delimiter resets to 2
    assert positions.tolist() == [0, 1, 2, 3, 4, 2, 3, 2]


def test_build_multi_item_extend_positions_requires_query_prefix():
    delimiter = 99
    tokens = np.array([99, 1, 2, 99], dtype=np.int32)

    with pytest.raises(ValueError, match="non-empty query prefix"):
        _build_multi_item_extend_positions(tokens, delimiter, np.int32)


def test_build_extend_positions_for_req_multi_item_requires_no_cached_prefix():
    req = SimpleNamespace(
        is_multi_item_scoring=True,
        multi_item_scoring_delimiter=99,
        fill_ids=[10, 11, 99, 21, 99],
    )

    with pytest.raises(ValueError, match="without cached prefix"):
        _build_extend_positions_for_req(req, seq_len=5, prefix_len=1, dtype=np.int32)
