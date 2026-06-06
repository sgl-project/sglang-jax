from unittest.mock import MagicMock

import numpy as np

from sgl_jax.srt.managers.schedule_batch import ScheduleBatch, ScheduleReqsInfo


class _ReqToTokenPool:
    def __init__(self):
        self.mapping = np.array([0, 11, 22, 33, 44], dtype=np.int32)

    def get_linear_recurrent_indices(self, req_pool_indices):
        return self.mapping[np.asarray(req_pool_indices, dtype=np.int32)]


def _req(rid):
    req = MagicMock()
    req.rid = rid
    return req


def _batch(reqs_info):
    return ScheduleBatch(
        reqs_info=reqs_info,
        req_to_token_pool=_ReqToTokenPool(),
        is_hybrid_recurrent=True,
        dp_size=len(reqs_info),
    )


def test_merge_batch_refreshes_recurrent_indices_for_nonempty_rank():
    left = _batch(
        [
            ScheduleReqsInfo(
                reqs=[_req("a")],
                req_pool_indices=np.array([1], dtype=np.int32),
                seq_lens=np.array([5], dtype=np.int32),
                recurrent_indices=np.array([99], dtype=np.int32),
            )
        ]
    )
    right = _batch(
        [
            ScheduleReqsInfo(
                reqs=[_req("b")],
                req_pool_indices=np.array([3], dtype=np.int32),
                seq_lens=np.array([7], dtype=np.int32),
                recurrent_indices=np.array([88], dtype=np.int32),
            )
        ]
    )

    left.merge_batch(right)

    np.testing.assert_array_equal(
        left.reqs_info[0].recurrent_indices,
        np.array([11, 33], dtype=np.int32),
    )


def test_merge_batch_refreshes_recurrent_indices_when_left_rank_empty():
    left = _batch([ScheduleReqsInfo(reqs=[])])
    right = _batch(
        [
            ScheduleReqsInfo(
                reqs=[_req("b")],
                req_pool_indices=np.array([2], dtype=np.int32),
                seq_lens=np.array([7], dtype=np.int32),
                recurrent_indices=np.array([88], dtype=np.int32),
            )
        ]
    )

    left.merge_batch(right)

    np.testing.assert_array_equal(
        left.reqs_info[0].recurrent_indices,
        np.array([22], dtype=np.int32),
    )


def test_filter_batch_refreshes_recurrent_indices():
    batch = _batch(
        [
            ScheduleReqsInfo(
                reqs=[_req("a"), _req("b"), _req("c")],
                req_pool_indices=np.array([1, 2, 3], dtype=np.int32),
                seq_lens=np.array([5, 6, 7], dtype=np.int32),
                recurrent_indices=np.array([99, 98, 97], dtype=np.int32),
            )
        ]
    )

    batch.filter_batch(keep_indices={0: [0, 2]})

    np.testing.assert_array_equal(
        batch.reqs_info[0].recurrent_indices,
        np.array([11, 33], dtype=np.int32),
    )
