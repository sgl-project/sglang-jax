"""The linear penalty must reach the sampler with exactly as many rows as the padded sampling batch.

ScheduleBatch._merge_sampling_info pads the penalty into per-DP slots. from_model_worker_batch used to pad it
again on the speculative prefill path (pad_size = len(seq_lens) - real_bs), which crashed the sampler's
logits + penalty add.
"""

from types import SimpleNamespace

import jax
import numpy as np
import pytest
from jax.sharding import Mesh

from sgl_jax.srt.managers.schedule_batch import ScheduleBatch, ScheduleReqsInfo
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.sampling.sampling_batch_info import SamplingBatchInfo, SamplingMetadata
from sgl_jax.srt.sampling.sampling_params import SamplingParams

DP, PER_DP_BS, VOCAB = 2, 8, 8
TOTAL_BS = DP * PER_DP_BS
STOP_TOKEN = 0
# min_new_tokens per request, per DP rank. Rows 0 (rank 0) and PER_DP_BS (rank 1's first slot) are penalized.
MIN_NEW_TOKENS = [[4], [4, 0]]


def _req(min_new_tokens):
    return SimpleNamespace(
        sampling_params=SamplingParams(min_new_tokens=min_new_tokens),
        tokenizer=SimpleNamespace(additional_stop_token_ids=None, eos_token_id=STOP_TOKEN),
        grammar=None,
    )


def _rank_info(min_new_tokens_per_req):
    info = ScheduleReqsInfo(
        reqs=[_req(n) for n in min_new_tokens_per_req],
        seq_lens=np.full(len(min_new_tokens_per_req), 100, dtype=np.int32),
    )
    # As the scheduler builds it; the merge then has the per-rank orchestrator write the penalty.
    info.sampling_info = SamplingBatchInfo.from_schedule_batch(info, VOCAB)
    return info


def _merged_batch():
    schedule_batch = ScheduleBatch(
        reqs_info=[_rank_info(reqs) for reqs in MIN_NEW_TOKENS],
        dp_size=DP,
        forward_mode=ForwardMode.EXTEND,
    )
    return SimpleNamespace(
        sampling_info=schedule_batch._merge_sampling_info(PER_DP_BS, TOTAL_BS),
        forward_mode=ForwardMode.EXTEND,
        seq_lens=np.full(TOTAL_BS, 100, dtype=np.int32),
        positions=None,
        return_logprob=False,
        top_logprobs_nums=None,
        token_ids_logprobs=None,
    )


def _mesh():
    return Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("data", "tensor"))


def test_merged_penalty_reaches_the_sampler_in_its_dp_slots():
    metadata = SamplingMetadata.from_model_worker_batch(
        _merged_batch(), mesh=_mesh(), vocab_size=VOCAB
    )
    penalty = np.asarray(metadata.linear_penalty)
    assert penalty.shape == (TOTAL_BS, VOCAB)
    np.testing.assert_array_equal(np.flatnonzero(penalty[:, STOP_TOKEN]), [0, PER_DP_BS])
    assert not np.delete(penalty, STOP_TOKEN, axis=1).any()
    assert bool(metadata.do_penalties)


def test_abstract_penalty_has_the_batch_shape():
    metadata = SamplingMetadata.from_model_worker_batch(
        _merged_batch(), mesh=_mesh(), vocab_size=VOCAB, abstract=True
    )
    assert metadata.linear_penalty.shape == (TOTAL_BS, VOCAB)


@pytest.mark.parametrize("rows", [3, TOTAL_BS + 3])
def test_penalty_with_the_wrong_row_count_is_rejected(rows):
    batch = _merged_batch()
    batch.sampling_info.linear_penalty = np.zeros((rows, VOCAB), dtype=np.float32)
    with pytest.raises(ValueError, match="rows"):
        SamplingMetadata.from_model_worker_batch(batch, mesh=_mesh(), vocab_size=VOCAB)
