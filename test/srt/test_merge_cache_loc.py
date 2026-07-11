"""Correctness tests for the vectorized ScheduleBatch._merge_cache_loc.

The paged fast path (page_size > 1) reconstructs token-level slot indices from
page-start values instead of memcpy-ing every token from req_to_token. These
tests verify byte-identity against the reference per-req-loop implementation
at every real-token position, plus the page-start slice that FA/MLA backends
actually consume (cache_loc.reshape(dp, -1)[:, ::page_size]).
"""

import unittest
from unittest.mock import MagicMock

import numpy as np

from sgl_jax.srt.managers.schedule_batch import ScheduleBatch, ScheduleReqsInfo
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode


def _make_paged_req_to_token(pool_size, max_ctx, page_size, seed=0):
    """Build a req_to_token with the same page-contiguous layout the
    PagedTokenToKVPoolAllocator produces: within each page, slot indices are
    consecutive integers starting at page_id * page_size."""
    rng = np.random.default_rng(seed)
    n_pages = pool_size * (max_ctx // page_size)
    page_ids = rng.permutation(n_pages).astype(np.int32).reshape(pool_size, -1)
    return (page_ids[:, :, None] * page_size + np.arange(page_size, dtype=np.int32)).reshape(
        pool_size, max_ctx
    )


def _reference_loop(req_to_token, reqs_info, dp_size, per_dp_size, page_size, total_size):
    """The pre-vectorization per-req loop (kept as ground truth)."""
    out = np.zeros(total_size, dtype=np.int32)
    offset_bs = 0
    for dp_rank in range(dp_size):
        info = reqs_info[dp_rank]
        if info.seq_lens is None or len(info.seq_lens) == 0:
            offset_bs += per_dp_size
            continue
        seq_lens = info.seq_lens
        aligned = ((seq_lens + page_size - 1) // page_size) * page_size
        offsets = np.concatenate([[0], np.cumsum(aligned[:-1])]).astype(np.int64)
        for r in range(len(seq_lens)):
            sl = int(seq_lens[r])
            d0 = int(offsets[r]) + offset_bs
            out[d0 : d0 + sl] = req_to_token[int(info.req_pool_indices[r]), :sl]
        offset_bs += per_dp_size
    return out


class TestMergeCacheLoc(unittest.TestCase):
    def _run_merge(self, reqs_info, dp_size, req_to_token, page_size, total_cache_loc_size):
        pool = MagicMock()
        pool.req_to_token = req_to_token
        pool.cache_loc_host_buf = np.zeros(total_cache_loc_size, dtype=np.int32)
        batch = ScheduleBatch(
            reqs_info=reqs_info,
            dp_size=dp_size,
            req_to_token_pool=pool,
            forward_mode=ForwardMode.EXTEND,  # extend picks cache_loc_paddings[-1]
        )
        return batch._merge_cache_loc(
            bs_paddings=[dp_size * 128],
            cache_loc_paddings=[total_cache_loc_size],
            page_size=page_size,
            per_dp_bs_size=128,
        )

    def _assert_real_tokens_match(self, got, ref, reqs_info, dp_size, per_dp, page_size, msg=""):
        offset_bs = 0
        for dp_rank in range(dp_size):
            info = reqs_info[dp_rank]
            if info.seq_lens is None or len(info.seq_lens) == 0:
                offset_bs += per_dp
                continue
            aligned = ((info.seq_lens + page_size - 1) // page_size) * page_size
            offsets = np.concatenate([[0], np.cumsum(aligned[:-1])]).astype(np.int64)
            for r in range(len(info.seq_lens)):
                sl = int(info.seq_lens[r])
                d0 = int(offsets[r]) + offset_bs
                np.testing.assert_array_equal(
                    got[d0 : d0 + sl],
                    ref[d0 : d0 + sl],
                    err_msg=f"{msg} dp={dp_rank} req={r} sl={sl}",
                )
            offset_bs += per_dp

    def test_paged_matches_reference(self):
        """Vectorized paged path must be byte-identical to the loop at every
        real-token position, across page sizes / dp / boundary alignments."""
        pool_size, max_ctx = 64, 4096
        rng = np.random.default_rng(123)
        for page_size in (16, 64, 256):
            req_to_token = _make_paged_req_to_token(pool_size, max_ctx, page_size)
            for dp_size in (1, 2, 4):
                for _ in range(3):
                    reqs_info = []
                    for _dp in range(dp_size):
                        n = int(rng.integers(0, 9))  # includes empty-DP case
                        if n == 0:
                            reqs_info.append(ScheduleReqsInfo(reqs=[], seq_lens=None))
                            continue
                        # Mix of page-boundary-exact and off-boundary seq_lens
                        seq = rng.integers(1, max_ctx, size=n).astype(np.int32)
                        seq[::3] = (seq[::3] // page_size) * page_size
                        seq = np.maximum(seq, 1)
                        idx = rng.choice(pool_size, size=n, replace=False).astype(np.int32)
                        reqs_info.append(
                            ScheduleReqsInfo(reqs=[], seq_lens=seq, req_pool_indices=idx)
                        )
                    per_dp = ((max_ctx * 8 + page_size - 1) // page_size) * page_size
                    total = per_dp * dp_size
                    got = self._run_merge(reqs_info, dp_size, req_to_token, page_size, total)
                    ref = _reference_loop(
                        req_to_token, reqs_info, dp_size, per_dp, page_size, total
                    )
                    self._assert_real_tokens_match(
                        got,
                        ref,
                        reqs_info,
                        dp_size,
                        per_dp,
                        page_size,
                        msg=f"page={page_size} dp={dp_size}",
                    )
                    # Page-start slice (what FA/MLA backends read) must match
                    # exactly — this is the correctness-critical view.
                    np.testing.assert_array_equal(
                        got.reshape(dp_size, per_dp)[:, ::page_size],
                        ref.reshape(dp_size, per_dp)[:, ::page_size],
                    )

    def test_page_size_1_full_identity(self):
        """page_size==1 keeps the per-req loop; must be fully byte-identical
        even without the page-contiguous invariant."""
        pool_size, max_ctx = 32, 512
        rng = np.random.default_rng(7)
        req_to_token = rng.integers(0, 1 << 20, size=(pool_size, max_ctx), dtype=np.int32)
        seq = np.array([5, 128, 1, 511, 42], dtype=np.int32)
        idx = np.array([3, 0, 17, 9, 22], dtype=np.int32)
        reqs_info = [ScheduleReqsInfo(reqs=[], seq_lens=seq, req_pool_indices=idx)]
        total = 4096
        got = self._run_merge(reqs_info, 1, req_to_token, 1, total)
        ref = _reference_loop(req_to_token, reqs_info, 1, total, 1, total)
        np.testing.assert_array_equal(got, ref)

    def test_all_empty_dp(self):
        """All-empty batch must not crash and returns the buffer view."""
        req_to_token = np.zeros((8, 256), dtype=np.int32)
        reqs_info = [
            ScheduleReqsInfo(reqs=[], seq_lens=None),
            ScheduleReqsInfo(reqs=[], seq_lens=np.empty(0, dtype=np.int32)),
        ]
        got = self._run_merge(reqs_info, 2, req_to_token, 64, 1024)
        self.assertEqual(got.shape, (1024,))
        np.testing.assert_array_equal(got, np.zeros(1024, dtype=np.int32))

    def test_single_req_single_token(self):
        """Minimal edge: bs=1, seq_len=1 (one page)."""
        req_to_token = _make_paged_req_to_token(4, 256, 64)
        reqs_info = [
            ScheduleReqsInfo(
                reqs=[],
                seq_lens=np.array([1], dtype=np.int32),
                req_pool_indices=np.array([2], dtype=np.int32),
            )
        ]
        got = self._run_merge(reqs_info, 1, req_to_token, 64, 256)
        ref = _reference_loop(req_to_token, reqs_info, 1, 256, 64, 256)
        np.testing.assert_array_equal(got[:1], ref[:1])
        self.assertEqual(got[0], req_to_token[2, 0])


if __name__ == "__main__":
    unittest.main()
