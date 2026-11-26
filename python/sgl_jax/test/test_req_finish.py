from types import SimpleNamespace

import numpy as np
import pytest

from sgl_jax.srt.managers.schedule_batch import Req
from sgl_jax.srt.managers.scheduler_output_processor_mixin import (
    SchedulerOutputProcessorMixin,
)
from sgl_jax.srt.sampling.penaltylib.min_new_tokens import BatchedMinNewTokensPenalizer
from sgl_jax.srt.sampling.penaltylib.orchestrator import BatchedPenalizerOrchestrator
from sgl_jax.srt.sampling.sampling_params import SamplingParams


@pytest.mark.parametrize("block_size", [1, 4, 16])
def test_length_limit_trims_accepted_block(block_size):
    req = Req(
        rid="length-limit",
        origin_input_text="",
        origin_input_ids=[1],
        sampling_params=SamplingParams(max_new_tokens=5, ignore_eos=True),
    )
    req.output_ids = list(range(4 + block_size))
    req.check_finished(new_accepted_len=block_size)
    assert req.finished_reason.to_json() == {"type": "length", "length": 5}
    assert req.output_ids_through_stop == list(range(5))
    decode_ids, _ = req.init_incremental_detokenize()
    assert decode_ids == [1] + list(range(5))


@pytest.mark.parametrize("eos_position", [2, 4, 6])
@pytest.mark.parametrize("ignore_eos", [False, True])
def test_accepted_block_stops_at_earliest_limit(eos_position, ignore_eos):
    req = Req(
        rid="eos-limit",
        origin_input_text="",
        origin_input_ids=[1],
        sampling_params=SamplingParams(max_new_tokens=5, ignore_eos=ignore_eos),
        eos_token_ids={2},
    )
    req.output_ids = [11] * 8
    req.output_ids[eos_position] = 2
    req.check_finished(new_accepted_len=7)
    stop = 5 if ignore_eos else min(5, eos_position + 1)
    assert req.finished_len == stop
    assert req.output_ids_through_stop == req.output_ids[:stop]
    assert req.finished_reason.to_json()["type"] == ("stop" if stop < 5 else "length")


@pytest.mark.parametrize("output_only", [False, True])
@pytest.mark.parametrize(
    "max_new_tokens,eos_token_ids,steps,expected_stop",
    [(5, None, (4, 8), 5), (0, None, (1,), 0), (5, {12}, (1, 8), 3)],
)
def test_response_logprobs_match_output(
    max_new_tokens, eos_token_ids, steps, expected_stop, output_only
):
    req = Req(
        rid="logprob-boundary",
        origin_input_text="",
        origin_input_ids=[1],
        sampling_params=SamplingParams(max_new_tokens=max_new_tokens),
        eos_token_ids=eos_token_ids,
        return_logprob=not output_only,
        return_output_logprob_only=output_only,
        stream=True,
    )
    req.input_logprob_sent = True
    responses = []
    scheduler = SimpleNamespace(
        skip_tokenizer_init=True,
        stream_interval=1,
        spec_algorithm=None,
        _comm_backend=None,
        send_to_detokenizer=SimpleNamespace(send_pyobj=responses.append),
    )
    tokens = list(range(10, 10 + steps[-1]))
    logprobs = {
        "output_token_logprobs_val": [-i - 0.5 for i in range(len(tokens))],
        "output_token_logprobs_idx": tokens,
        "output_top_logprobs_val": [[-i - 0.5] for i in range(len(tokens))],
        "output_top_logprobs_idx": [[token] for token in tokens],
        "output_token_ids_logprobs_val": [[-i - 1.5] for i in range(len(tokens))],
        "output_token_ids_logprobs_idx": [[42] for _ in tokens],
    }
    previous = 0
    for end in steps:
        req.output_ids = tokens[:end]
        for field, values in logprobs.items():
            setattr(req, field, values[:end])
        req.check_finished(new_accepted_len=end - previous)
        SchedulerOutputProcessorMixin.stream_output_generation(
            scheduler, [req], return_logprob=not output_only, return_output_logprob_only=output_only
        )

        response = responses[-1]
        stop = min(end, expected_stop)
        assert response.output_ids == [tokens[previous:stop]]
        assert response.completion_tokens == [stop]
        for field, values in logprobs.items():
            expected = [values[previous:stop]]
            if output_only and not field.startswith("output_token_logprobs_"):
                expected = None
            assert getattr(response, field) == expected
        previous = end
    assert response.finished_reasons == [req.finished_reason.to_json()]


class _ReqsInfo:
    def __init__(self, reqs):
        self.reqs = reqs


def test_min_new_tokens_masks_every_eos_id_check_finished_accepts():
    # Llama-3 Instruct style: generation_config lists more EOS ids than
    # tokenizer.eos_token_id, and Req.check_finished stops on all of them.
    req = Req(
        rid="min-new-tokens",
        origin_input_text="",
        origin_input_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_new_tokens=32, min_new_tokens=4),
        eos_token_ids={10, 11, 12},
        vocab_size=64,
    )
    req.tokenizer = SimpleNamespace(eos_token_id=12, additional_stop_token_ids={7})
    info = _ReqsInfo([req])
    orchestrator = BatchedPenalizerOrchestrator(64, info, {BatchedMinNewTokensPenalizer})

    penalty = orchestrator.apply()
    for token_id in (7, 10, 11, 12):
        assert np.isneginf(penalty[0, token_id]), token_id
    assert np.count_nonzero(penalty[0]) == 4

    for _ in range(4):
        orchestrator.cumulate_output_tokens(np.array([[3]]))
    assert not np.any(orchestrator.apply()[0])
