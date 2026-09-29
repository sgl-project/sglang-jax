from types import SimpleNamespace

import pytest

from sgl_jax.srt.managers.schedule_batch import Req
from sgl_jax.srt.managers.scheduler_output_processor_mixin import (
    SchedulerOutputProcessorMixin,
)
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
