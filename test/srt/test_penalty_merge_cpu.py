"""CPU penalty merge equivalence, including DP padding and asynchronous ownership."""

from types import SimpleNamespace as NS

import numpy as np
import pytest

from sgl_jax.srt.managers.schedule_batch import ScheduleBatch, ScheduleReqsInfo
from sgl_jax.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sgl_jax.srt.sampling.penaltylib import (
    BatchedPenalizerOrchestrator,
    BatchedFrequencyPenalizer,
    BatchedPresencePenalizer,
    BatchedMinNewTokensPenalizer,
)


def make_batch(counts=(4, 4), vocab=64, mixed=False, enabled=True):
    infos = []
    for count in counts:
        reqs = [
            NS(
                sampling_params=NS(
                    min_new_tokens=(2 + i) if enabled else 0,
                    frequency_penalty=(0.25 if i % 2 else -0.5) if mixed else 0.0,
                    presence_penalty=(-0.75 if i % 2 else 0.5) if mixed else 0.0,
                    stop_token_ids={3, vocab + 7},
                ),
                tokenizer=NS(additional_stop_token_ids={4}, eos_token_id=5),
                grammar=None,
            )
            for i in range(count)
        ]
        info = ScheduleReqsInfo(reqs=reqs, seq_lens=np.ones(count, np.int32))
        orch = BatchedPenalizerOrchestrator(
            vocab,
            info,
            {
                BatchedFrequencyPenalizer,
                BatchedPresencePenalizer,
                BatchedMinNewTokensPenalizer,
            },
        )
        info.sampling_info = SamplingBatchInfo(
            temperatures=np.ones((count, 1), np.float32),
            top_ps=np.ones(count, np.float32),
            top_ks=np.ones(count, np.int32),
            min_ps=np.zeros(count, np.float32),
            vocab_size=vocab,
            penalizer_orchestrator=orch,
        )
        infos.append(info)
    return NS(reqs_info=infos, dp_size=len(infos), has_grammar=False)


def reference(batch, per_dp):
    """Independent dense reference to the original formulas."""
    vocab = batch.reqs_info[0].sampling_info.vocab_size
    out = np.zeros((len(batch.reqs_info) * per_dp, vocab), np.float32)
    required = False
    for rank, info in enumerate(batch.reqs_info):
        o = info.sampling_info.penalizer_orchestrator
        required |= o.is_required and len(info.seq_lens) > 0
        dst = out[rank * per_dp : rank * per_dp + len(info.seq_lens)]
        for kind in (BatchedPresencePenalizer, BatchedFrequencyPenalizer):
            p = o.penalizers[kind]
            if p.is_prepared():
                values = (
                    p.token_presence if kind is BatchedPresencePenalizer else p.token_frequencies
                )
                scale = (
                    p.presence_penalties
                    if kind is BatchedPresencePenalizer
                    else p.frequency_penalties
                )
                dst += values.astype(np.float32) * -scale
        p = o.penalizers[BatchedMinNewTokensPenalizer]
        if p.is_prepared():
            stops = np.zeros_like(dst)
            for i, tokens in enumerate(p.stop_token_sequences):
                stops[i, tokens[tokens < vocab]] = -np.inf
            dst += np.where(p.len_output_tokens < p.min_new_tokens, stops, 0.0)
    return out if required else None


@pytest.mark.parametrize("counts", [(4, 4), (1, 3), (0, 4), (2, 0)])
@pytest.mark.parametrize("mixed", [False, True])
def test_merge_matches_dense_reference_and_preserves_previous(counts, mixed):
    batch = make_batch(counts, mixed=mixed)
    previous = []
    for step in range(7):
        expected = reference(batch, 4)
        merged = ScheduleBatch._merge_sampling_info(batch, 4, 8)
        np.testing.assert_array_equal(merged.linear_penalty, expected)
        assert merged.linear_penalty.dtype == np.float32
        previous.append((merged.linear_penalty, expected.copy()))
        for info in batch.reqs_info:
            ids = np.full(len(info.seq_lens), 3 if step % 2 else 7, np.int32)
            info.sampling_info.penalizer_orchestrator.cumulate_output_tokens(ids)
        for old, snapshot in previous:
            np.testing.assert_array_equal(old, snapshot)


def test_no_penalties_remains_none():
    batch = make_batch(enabled=False)
    assert ScheduleBatch._merge_sampling_info(batch, 4, 8).linear_penalty is None


def test_direct_min_penalty_does_not_construct_dense_intermediate(monkeypatch):
    batch = make_batch()
    for info in batch.reqs_info:
        p = info.sampling_info.penalizer_orchestrator.penalizers[BatchedMinNewTokensPenalizer]
        monkeypatch.setattr(p, "compute_penalty", lambda: pytest.fail("dense intermediate"))
    result = ScheduleBatch._merge_sampling_info(batch, 4, 8)
    np.testing.assert_array_equal(result.linear_penalty, reference(batch, 4))


def test_penalties_after_request_filter_and_merge():
    batch = make_batch((4, 2), mixed=True)
    left, right = batch.reqs_info
    orch = left.sampling_info.penalizer_orchestrator
    orch.cumulate_output_tokens(np.array([3, 4, 7, 7], np.int32))
    keep = np.array([3, 1], np.int32)
    left.reqs = [left.reqs[i] for i in keep]
    orch.filter(keep)
    orch.merge(right.sampling_info.penalizer_orchestrator)
    left.reqs += right.reqs
    left.seq_lens = np.ones(4, np.int32)
    batch.reqs_info = [left]
    batch.dp_size = 1
    np.testing.assert_array_equal(
        ScheduleBatch._merge_sampling_info(batch, 4, 4).linear_penalty,
        reference(batch, 4),
    )


def test_precomputed_worker_penalties():
    from sgl_jax.srt.managers.schedule_batch import ModelWorkerSamplingInfo

    batch = make_batch((2, 0))
    info = batch.reqs_info[0]
    old = info.sampling_info
    penalty = np.arange(128, dtype=np.float32).reshape(2, 64)
    info.sampling_info = ModelWorkerSamplingInfo(
        temperatures=old.temperatures,
        top_ps=old.top_ps,
        top_ks=old.top_ks,
        min_ps=old.min_ps,
        vocab_size=64,
        linear_penalty=penalty,
    )
    merged = ScheduleBatch._merge_sampling_info(batch, 4, 8)
    np.testing.assert_array_equal(merged.linear_penalty[:2], penalty)
    assert not merged.linear_penalty[2:].any()
    penalty[:] = 0
    assert merged.linear_penalty[0, 1] == 1
