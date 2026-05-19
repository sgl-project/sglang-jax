"""CPU-only shape/layout tests for spec-decode DP plumbing (#1053 P1-5b).

Run: JAX_PLATFORMS=cpu pytest test_spec_dp_shapes.py -v

Catches the Python-layer shim/shape/index bugs (P1-5b r4-r9) without a 15-min
v6e-64 reload. Multi-host sharding errors (process_allgather, P("data")
shard_shape on real mesh) are NOT covered — those still need hardware.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from types import SimpleNamespace

import numpy as np
import pytest

from sgl_jax.srt.managers.schedule_batch import ScheduleBatch, ScheduleReqsInfo
from sgl_jax.srt.managers.scheduler_output_processor_mixin import (
    SchedulerOutputProcessorMixin,
)
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.speculative.eagle_util import EagleDraftInput
from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm

HIDDEN = 8
TOPK = 1
DRAFT_N = 4


class _Req:
    def __init__(self, rid: int):
        self.rid = rid
        self._done = False
        self.output_ids: list[int] = []
        self.origin_input_ids = [1, 2, 3, 4, 5]
        self.spec_verify_ct = 0
        self.spec_accepted_tokens = 0
        self.is_retracted = False
        self.return_logprob = False
        self.return_output_logprob_only = False
        self.stream = False
        self.grammar = None
        self.return_hidden_states = False

    def finished(self) -> bool:
        return self._done


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
            info.reqs = [_Req(r * 100 + j) for j in range(bs)]
            info.seq_lens = np.full((bs,), 6 + r, np.int32)
            info.req_pool_indices = np.arange(r * 10, r * 10 + bs, dtype=np.int32)
        else:
            info.reqs = []
            info.seq_lens = None
            info.req_pool_indices = None
        info.out_cache_loc = (
            np.arange(r * 1000, r * 1000 + bs * DRAFT_N, dtype=np.int32) if bs > 0 else None
        )
        info.spec_info = None
        info.sampling_info = None
        reqs_info.append(info)
    reqs_info[0].spec_info = _mk_spec_info(real_bs)

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
    monkeypatch.setattr(ScheduleBatch, "_merge_sampling_info", lambda self, per_dp, total: None)


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
    # out_cache_loc DP-segmented: rank r's section = its own slots then -1 pad
    ocl = np.asarray(mwb.out_cache_loc).reshape(dp, -1)
    for r, bs in enumerate(bs_per_rank):
        n = bs * DRAFT_N
        assert list(ocl[r, :n]) == list(range(r * 1000, r * 1000 + n)), (r, ocl[r])
        assert np.all(ocl[r, n:] == -1), f"rank {r} pad not -1: {ocl[r, n:]}"
    # seq_lens slot layout: rank r's reqs at [r*per_dp_bs : r*per_dp_bs+bs_r]
    per_dp = mwb.per_dp_bs_size
    for r, bs in enumerate(bs_per_rank):
        seg = mwb.seq_lens[r * per_dp : r * per_dp + per_dp]
        assert np.all(seg[:bs] > 0), f"rank {r} real slots zero: {seg}"
        assert np.all(seg[bs:] == 0), f"rank {r} pad slots nonzero: {seg}"
    assert mwb.spec_info is not None


@pytest.mark.parametrize(
    "dp,bs_per_rank,finish",
    [
        (2, [1, 1], [(0, 0)]),  # rank 0 finishes → rank 1 survives (r9 crash repro)
        (2, [1, 1], [(1, 0)]),  # rank 1 finishes → rank 0 survives
        (2, [2, 1], [(0, 0)]),  # rank 0 partial → keep rank0[1] + rank1[0]
        (4, [1, 1, 1, 1], [(0, 0), (2, 0)]),  # ranks 0,2 finish
        (1, [3], [(0, 1)]),  # dp=1 regression
    ],
)
def test_filter_batch_preserves_global_spec_info(dp, bs_per_rank, finish):
    sb = _mk_batch(dp, bs_per_rank)
    real_bs = sum(bs_per_rank)
    # Tag spec_info arrays so we can verify which entries survive.
    spec = sb.reqs_info[0].spec_info
    spec.allocate_lens = np.arange(100, 100 + real_bs, dtype=np.int32)
    spec.verified_id = np.arange(200, 200 + real_bs, dtype=np.int32)
    spec.hidden_states = np.arange(real_bs, dtype=np.float32).reshape(real_bs, 1).repeat(HIDDEN, 1)
    # Compute expected survivors in global-flat order BEFORE marking finished.
    finish_set = set(finish)
    expected_keep = []
    flat = 0
    for r, bs in enumerate(bs_per_rank):
        for j in range(bs):
            if (r, j) not in finish_set:
                expected_keep.append(flat)
            flat += 1
    for r, j in finish:
        sb.reqs_info[r].reqs[j]._done = True

    sb.filter_batch()

    new_spec = sb.reqs_info[0].spec_info
    assert new_spec is not None, "global spec_info dropped after partial-finish"
    assert new_spec.allocate_lens.shape == (len(expected_keep),)
    assert list(new_spec.allocate_lens) == [100 + i for i in expected_keep]
    assert list(new_spec.verified_id) == [200 + i for i in expected_keep]
    assert new_spec.hidden_states.shape == (len(expected_keep), HIDDEN)
    # Surviving reqs per rank should match.
    for r, bs in enumerate(bs_per_rank):
        kept = bs - sum(1 for fr, fj in finish if fr == r)
        assert len(sb.reqs_info[r].reqs or []) == kept


def test_filter_batch_then_decode_mwb_round_trip():
    """Regression for 2-req partial-finish → next-round decode mwb (r9 crash).

    After rank 0 empties, the next `_get_spec_decode_mwb_dp` must still produce
    a dp-divisible mwb whose spec_info aligns with the DP-padded seq_lens slots.
    """
    dp, bs_per_rank = 2, [1, 1]
    sb = _mk_batch(dp, bs_per_rank)
    spec = sb.reqs_info[0].spec_info
    spec.allocate_lens = np.array([110, 120], dtype=np.int32)
    sb.reqs_info[0].reqs[0]._done = True
    sb.filter_batch()
    # rank 0 empty, rank 1 has 1 req; spec_info global-flat now [120].
    buckets = [b for b in BS_BUCKETS if b >= dp]
    mwb = sb._get_spec_decode_mwb_dp(buckets, enable_static_lora=False)
    assert mwb.real_bs == 1
    assert mwb.real_bs_per_dp == [0, 1]
    assert len(mwb.seq_lens) % dp == 0
    per_dp = mwb.per_dp_bs_size
    # rank 0 slot(s) all-zero, rank 1 slot 0 has data.
    assert np.all(mwb.seq_lens[:per_dp] == 0)
    assert mwb.seq_lens[per_dp] > 0
    # spec_info is now DP-padded (total_bs,); rank 1's data must be at slot per_dp.
    al = np.asarray(mwb.spec_info.allocate_lens)
    assert al.shape[0] == per_dp * dp
    assert int(al[per_dp]) == 120
    assert int(al[0]) == 0  # rank-0 padding slot


@pytest.mark.parametrize(
    "dp,bs_per_rank",
    [
        (2, [0, 1]),  # rank 0 empty (post-partial-finish shape)
        (2, [1, 2]),  # unbalanced, rank 1 heavier
        (4, [0, 1, 0, 2]),
    ],
)
def test_spec_info_aligns_with_dp_padded_slots(dp, bs_per_rank):
    """Core layout invariant: after `_get_spec_decode_mwb_dp`, spec_info arrays
    must index-align with DP-padded mwb.seq_lens — i.e., spec_info[i] is the
    data for the req at slot i (or padding). `padding_for_decode` then uses
    `valid_mask = seq_lens > 0` to index spec_info.allocate_lens; if spec_info
    is global-flat (no inter-rank padding) this picks wrong entries.

    Current impl stores spec_info global-flat → this test FAILS for unbalanced
    bs_per_rank → drives the layout fix (scatter via logits_indices_selector,
    PR #1108 design question option A).
    """
    sb = _mk_batch(dp, bs_per_rank)
    real_bs = sum(bs_per_rank)
    spec = sb.reqs_info[0].spec_info
    # Tag allocate_lens with global-flat indices so we know which req each is.
    spec.allocate_lens = np.arange(100, 100 + real_bs, dtype=np.int32)
    buckets = [b for b in BS_BUCKETS if b >= dp]
    mwb = sb._get_spec_decode_mwb_dp(buckets, enable_static_lora=False)
    per_dp = mwb.per_dp_bs_size
    total_bs = per_dp * dp

    # Simulate padding_for_decode's pad-then-valid_mask gather:
    al = np.asarray(mwb.spec_info.allocate_lens)
    if len(al) < total_bs:
        al = np.pad(al, (0, total_bs - len(al)))
    valid_mask = mwb.seq_lens > 0
    picked = al[valid_mask]

    # Expected: picked[k] == 100 + k (global-flat order of survivors).
    assert list(picked) == list(range(100, 100 + real_bs)), (
        f"layout mismatch: valid_mask picked {list(picked)} from "
        f"allocate_lens (DP-padded)={list(al)}, seq_lens={list(mwb.seq_lens)}; "
        f"expected {list(range(100, 100 + real_bs))}. "
        f"spec_info is global-flat but mwb is DP-padded — needs scatter."
    )


@pytest.mark.parametrize(
    "dp,bs_per_rank",
    [
        (1, [3]),
        (2, [1, 1]),
        (2, [2, 1]),
        (2, [0, 2]),
        (4, [1, 1, 1, 1]),
        (4, [2, 1, 0, 3]),
    ],
)
def test_draft_page_indices_dp_segmented(dp, bs_per_rank):
    """multi_step page_indices must be DP-segmented so the P("data") shard at
    [r*per_dp_dst:(r+1)*per_dp_dst) gives rank r exactly its own reqs' pages.

    Regression for the rank>0-reads-padding bug (dp=4 garbage output +
    accept-len ~1): pre-fix the gather wrote contiguous-then-pad, so rank>0's
    shard landed in the all-zero padding region.
    """
    sb = _mk_batch(dp, bs_per_rank)
    real_bs = sum(bs_per_rank)
    spec = sb.reqs_info[0].spec_info
    # Distinct seq_len per req so page boundaries are observable.
    spec.allocate_lens = np.asarray([256 * (k + 1) + 4 for k in range(real_bs)], dtype=np.int32)
    buckets = [b for b in BS_BUCKETS if b >= dp]
    mwb = sb._get_spec_decode_mwb_dp(buckets, enable_static_lora=False)
    per_dp = mwb.per_dp_bs_size
    sel = np.asarray(mwb.logits_indices_selector)
    assert sel.shape == (real_bs,)
    rank_of = sel // per_dp
    # cache_loc layout: rank r section = [r*L : (r+1)*L), tokens tagged with
    # global-flat req id k so we can verify which req each page belongs to.
    L_tok, page = 8192, 256
    cache_loc = np.full(L_tok * dp, -1, dtype=np.int32)
    al = np.asarray(mwb.spec_info.allocate_lens)
    aligned = ((al + page - 1) // page) * page
    intra = np.zeros(dp, dtype=np.int64)
    for s in np.where(mwb.seq_lens > 0)[0]:
        r = int(s) // per_dp
        k = int(np.where(sel == s)[0][0])
        b = r * L_tok + intra[r]
        cache_loc[b : b + al[s]] = k
        intra[r] += aligned[s]
    # multi_step gather (mirror of get_eagle_multi_step_metadata step 0).
    src_locs = cache_loc[::page]
    L_pg = L_tok // page
    alloc_pg = ((al[sel] + page - 1) // page).astype(np.int64)
    spec_pg = ((mwb.seq_lens[sel] + page - 1) // page).astype(np.int64)

    def starts(pages, base):
        out = np.zeros(len(pages), dtype=np.int64)
        for r in range(dp):
            m = rank_of == r
            if np.any(m):
                c = np.cumsum(pages[m])
                out[m] = r * base + np.concatenate(([0], c[:-1]))
        return out

    DST, per_dst = 16384, 16384 // dp
    flat_cum = np.concatenate(([0], np.cumsum(spec_pg)[:-1]))
    off = np.arange(int(spec_pg.sum())) - np.repeat(flat_cum, spec_pg)
    gi = np.repeat(starts(alloc_pg, L_pg), spec_pg) + off
    wi = np.repeat(starts(spec_pg, per_dst), spec_pg) + off
    result = np.full(DST, -1, dtype=np.int32)
    result[wi] = src_locs[gi]
    # Invariant: rank r's section contains exactly rank r's req-ids (in
    # global-flat order), then -1 padding; never another rank's id or 0-from-pad.
    for r in range(dp):
        seg = result[r * per_dst : (r + 1) * per_dst]
        ks = [k for k in range(real_bs) if rank_of[k] == r]
        want = np.concatenate([np.full(int(spec_pg[k]), k) for k in ks] or [np.array([], int)])
        assert list(seg[: len(want)]) == list(want), (r, seg[: len(want) + 2], want)
        assert np.all(seg[len(want) :] == -1), f"rank {r} pad region nonempty"


@pytest.mark.parametrize(
    "dp,bs_per_rank,accept_per_slot",
    [
        (1, [2], [3, 2]),
        (2, [1, 1], [3, 2]),
        (2, [1, 2], [3, 0, 2, 1]),  # per_dp=2: slot 1 is rank-0 pad (accept=0)
        (4, [1, 0, 1, 0], [2, 0, 3, 0]),  # per_dp=1
    ],
)
def test_resolve_spec_decode_token_ids(dp, bs_per_rank, accept_per_slot):
    """`_resolve_spec_decode_token_ids` must slice next_token_ids by DP-padded
    slot (not contiguous req index) and return a (total_bs,)-length list with
    [] at padding slots so the per-rank slice in process_batch_result_decode
    lands on the right reqs."""
    sb = _mk_batch(dp, bs_per_rank)
    total_bs = len(accept_per_slot)
    sb.per_dp_bs_size = total_bs // dp
    # Tag next_token_ids so slot s tokens are [s*1000+0, s*1000+1, ...]
    nt = np.concatenate(
        [np.arange(s * 1000, s * 1000 + DRAFT_N, dtype=np.int32) for s in range(total_bs)]
    )
    result = SimpleNamespace(
        next_token_ids=nt,
        accept_lens=np.asarray(accept_per_slot, dtype=np.int32),
        num_accepted_tokens=None,
    )
    sched = SimpleNamespace(draft_worker=SimpleNamespace(speculative_num_draft_tokens=DRAFT_N))
    out = SchedulerOutputProcessorMixin._resolve_spec_decode_token_ids(sched, result, sb)
    assert len(out) == total_bs
    per_dp = sb.per_dp_bs_size
    for r, bs in enumerate(bs_per_rank):
        for j in range(per_dp):
            slot = r * per_dp + j
            if j < bs:
                a = accept_per_slot[slot]
                assert out[slot] == list(
                    range(slot * 1000, slot * 1000 + a)
                ), f"slot {slot}: got {out[slot]}, want {a} tokens from {slot*1000}"
                assert sb.reqs_info[r].reqs[j].spec_accepted_tokens == a
            else:
                assert out[slot] == [], f"pad slot {slot} should be []"
