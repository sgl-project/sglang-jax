"""CPU-only shape/layout tests for spec-decode DP plumbing (#1053 P1-5b).

Run: JAX_PLATFORMS=cpu pytest test_spec_dp_shapes.py -v

Catches the Python-layer shim/shape/index bugs (P1-5b r4-r9) without a 15-min
v6e-64 reload. Multi-host sharding errors (process_allgather, P("data")
shard_shape on real mesh) are NOT covered — those still need hardware.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

from sgl_jax.srt.managers.schedule_batch import ScheduleBatch, ScheduleReqsInfo
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.speculative.eagle_util import EagleDraftInput
from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm

HIDDEN = 8
TOPK = 1
DRAFT_N = 4


def _mk_spec_info(real_bs: int) -> EagleDraftInput:
    return EagleDraftInput(
        topk_p=np.ones((real_bs, TOPK), np.float32),
        topk_index=np.zeros((real_bs, TOPK), np.int32),
        hidden_states=np.zeros((real_bs, HIDDEN), np.float32),
        verified_id=np.zeros((real_bs,), np.int32),
        allocate_lens=np.full((real_bs,), 8, np.int32),
    )


def _mk_batch(dp_size: int, bs_per_rank: list[int]) -> ScheduleBatch:
    """Minimal ScheduleBatch for spec-decode DP shape tests.

    Only fills fields read by `_get_spec_decode_mwb_dp` /
    `_merge_batch_metadata`. Model/pool/tree_cache left None.
    """
    assert len(bs_per_rank) == dp_size
    real_bs = sum(bs_per_rank)
    reqs_info = []
    for r, bs in enumerate(bs_per_rank):
        info = ScheduleReqsInfo()
        if bs > 0:
            info.reqs = [object()] * bs  # placeholder; only len() is read
            info.seq_lens = np.full((bs,), 6 + r, np.int32)
            info.req_pool_indices = np.arange(r * 10, r * 10 + bs, dtype=np.int32)
        else:
            info.reqs = []
            info.seq_lens = None
            info.req_pool_indices = None
        info.out_cache_loc = None
        info.spec_info = None
        info.sampling_info = None
        reqs_info.append(info)
    reqs_info[0].spec_info = _mk_spec_info(real_bs)
    reqs_info[0].out_cache_loc = np.arange(real_bs * DRAFT_N, dtype=np.int32)

    sb = ScheduleBatch.__new__(ScheduleBatch)
    sb.__dict__.update(
        dict(
            reqs_info=reqs_info,
            dp_size=dp_size,
            forward_mode=ForwardMode.DECODE,
            return_logprob=False,
            return_output_logprob_only=False,
            return_hidden_states=False,
            spec_algorithm=SpeculativeAlgorithm.NEXTN,
            tree_cache=None,
            launch_done=None,
            has_grammar=False,
            per_dp_bs_size=0,
        )
    )
    return sb


@pytest.fixture(autouse=True)
def _stub_sampling(monkeypatch):
    monkeypatch.setattr(
        ScheduleBatch, "_merge_sampling_info", lambda self, per_dp, total: None
    )


BS_BUCKETS = [1, 2, 4, 8, 16]


@pytest.mark.parametrize(
    "dp,bs_per_rank",
    [
        (1, [1]),
        (1, [3]),
        (2, [1, 0]),
        (2, [1, 1]),
        (2, [2, 1]),
        (2, [0, 2]),
        (4, [1, 0, 1, 0]),
        (4, [1, 1, 1, 1]),
        (4, [2, 1, 0, 3]),
    ],
)
def test_get_spec_decode_mwb_dp_shapes(dp, bs_per_rank):
    sb = _mk_batch(dp, bs_per_rank)
    real_bs = sum(bs_per_rank)
    buckets = [b for b in BS_BUCKETS if b >= dp]
    mwb = sb._get_spec_decode_mwb_dp(buckets, enable_static_lora=False)

    assert mwb.dp_size == dp
    assert mwb.real_bs == real_bs
    assert mwb.real_bs_per_dp == bs_per_rank
    assert mwb.per_dp_bs_size * dp == len(mwb.seq_lens)
    # all P("data")-sharded fields must be dp-divisible
    for name in ("seq_lens", "req_pool_indices", "out_cache_loc"):
        arr = getattr(mwb, name)
        assert len(arr) % dp == 0, f"{name} len={len(arr)} not %dp={dp}"
    # seq_lens slot layout: rank r's reqs at [r*per_dp_bs : r*per_dp_bs+bs_r]
    per_dp = mwb.per_dp_bs_size
    for r, bs in enumerate(bs_per_rank):
        seg = mwb.seq_lens[r * per_dp : r * per_dp + per_dp]
        assert np.all(seg[:bs] > 0), f"rank {r} real slots zero: {seg}"
        assert np.all(seg[bs:] == 0), f"rank {r} pad slots nonzero: {seg}"
    assert mwb.spec_info is not None
