"""Output-only logprobs are deltas on the scheduler/tokenizer boundary."""

from types import SimpleNamespace

from sgl_jax.srt.managers.schedule_batch import Req
from sgl_jax.srt.managers.scheduler_output_processor_mixin import (
    SchedulerOutputProcessorMixin,
)
from sgl_jax.srt.sampling.sampling_params import SamplingParams


def test_output_logprob_stream_has_one_delta_per_request():
    mixed = True
    sent = []
    scheduler = SimpleNamespace(
        stream_interval=1,
        skip_tokenizer_init=True,
        spec_algorithm=None,
        _comm_backend=None,
        send_to_detokenizer=SimpleNamespace(send_pyobj=sent.append),
    )
    reqs = [
        Req(
            "output-only",
            "",
            [1, 2],
            SamplingParams(),
            return_output_logprob_only=True,
            stream=True,
        ),
        Req("plain", "", [1, 2], SamplingParams(), stream=True),
    ]
    if mixed:
        full = Req("full", "", [1, 2], SamplingParams(), return_logprob=True, stream=True)
        full.input_logprob_sent = True
        reqs.append(full)
    for step in range(1, 5):
        for req in reqs:
            req.output_ids.append(10 + step)
            if req.return_logprob or req.return_output_logprob_only:
                req.output_token_logprobs_val.append(-0.1)
                req.output_token_logprobs_idx.append(10 + step)
        SchedulerOutputProcessorMixin.stream_output_generation(scheduler, reqs, mixed, True)
        out = sent[-1]
        assert len(out.output_token_logprobs_idx) == len(reqs)
        assert out.output_token_logprobs_idx[0] == [10 + step]
        assert out.output_token_logprobs_idx[1] == []
        if mixed:
            assert out.output_token_logprobs_idx[2] == [10 + step]
            assert len(out.output_top_logprobs_idx) == len(reqs)
