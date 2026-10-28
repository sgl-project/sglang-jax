"""EAGLE-family verify does not apply the linear penalty, so the scheduler must reject requests that use one."""

from types import MethodType, SimpleNamespace

import pytest

from sgl_jax.srt.managers.io_struct import TokenizedGenerateReqInput
from sgl_jax.srt.managers.scheduler import Scheduler
from sgl_jax.srt.sampling.sampling_params import SamplingParams
from sgl_jax.srt.server_args import ServerArgs
from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm


def _submit(spec_algorithm, **sampling_kwargs):
    """Run one request through Scheduler.handle_generate_request; return (streamed, queued) requests."""
    streamed, queued = [], []
    scheduler = SimpleNamespace(
        model_config=SimpleNamespace(hf_eos_token_id=None, vocab_size=64),
        tokenizer=None,
        spec_algorithm=spec_algorithm,
        server_args=SimpleNamespace(disaggregation_mode="null", allow_auto_truncate=False),
        max_req_input_len=1024,
        max_req_len=2048,
        grammar_backend=None,
        stream_output=lambda reqs, *args: streamed.extend(reqs),
        _add_request_to_queue=queued.append,
    )
    scheduler._is_spec_decode_enabled = MethodType(Scheduler._is_spec_decode_enabled, scheduler)
    request = TokenizedGenerateReqInput(
        rid="guard",
        input_ids=[1, 2],
        radix_input_ids=[1, 2],
        sampling_params=SamplingParams(max_new_tokens=8, **sampling_kwargs),
    )
    Scheduler.handle_generate_request(scheduler, request)
    return streamed, queued


@pytest.mark.parametrize(
    "sampling_kwargs",
    [{"min_new_tokens": 4}, {"frequency_penalty": 0.5}, {"presence_penalty": 0.5}],
)
def test_penalized_request_is_rejected_under_spec_decoding(sampling_kwargs):
    streamed, queued = _submit(SpeculativeAlgorithm.NEXTN, **sampling_kwargs)
    assert queued == []
    assert len(streamed) == 1
    assert streamed[0].finished()
    assert "Speculative decoding does not support" in streamed[0].finished_reason.message


@pytest.mark.parametrize(
    "spec_algorithm, sampling_kwargs",
    [
        (SpeculativeAlgorithm.NEXTN, {"temperature": 0.7}),
        # Not implemented anywhere, so it is ignored with or without speculative decoding.
        (SpeculativeAlgorithm.NEXTN, {"repetition_penalty": 1.1}),
        (SpeculativeAlgorithm.NONE, {"min_new_tokens": 4}),
    ],
)
def test_request_is_queued(spec_algorithm, sampling_kwargs):
    streamed, queued = _submit(spec_algorithm, **sampling_kwargs)
    assert streamed == []
    assert len(queued) == 1
    assert not queued[0].finished()


def _check_preferred(speculative_algorithm, preferred_sampling_params):
    args = SimpleNamespace(
        speculative_algorithm=speculative_algorithm,
        preferred_sampling_params=preferred_sampling_params,
    )
    ServerArgs.check_spec_preferred_sampling_params(args)


def test_server_wide_penalty_default_is_refused_at_startup():
    with pytest.raises(ValueError, match="presence_penalty"):
        _check_preferred("NEXTN", '{"presence_penalty": 1.5, "temperature": 0.7}')


@pytest.mark.parametrize(
    "speculative_algorithm, preferred",
    [
        ("NEXTN", '{"temperature": 0.7, "repetition_penalty": 1.05}'),
        ("NEXTN", '{"presence_penalty": 0.0}'),
        (None, '{"presence_penalty": 1.5}'),
    ],
)
def test_server_wide_defaults_that_spec_decoding_supports_are_accepted(
    speculative_algorithm, preferred
):
    _check_preferred(speculative_algorithm, preferred)
