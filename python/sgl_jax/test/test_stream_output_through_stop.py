from __future__ import annotations

from types import SimpleNamespace

from sgl_jax.srt.managers.scheduler_output_processor_mixin import (
    SchedulerOutputProcessorMixin,
)


class _Sender:
    def __init__(self):
        self.sent = None

    def send_pyobj(self, obj):
        self.sent = obj


class _Req:
    rid = "rid-0"
    finished_reason = None
    decoded_text = ""
    output_ids = [10, 20, 30, 99]
    finished_len = 3
    send_token_offset = 0
    send_decode_id_offset = 0
    send_output_token_logprobs_offset = 0
    sampling_params = SimpleNamespace(
        skip_special_tokens=True,
        spaces_between_special_tokens=True,
        no_stop_trim=False,
    )
    origin_input_ids = [1, 2]
    cached_tokens = 0
    return_logprob = False
    return_output_logprob_only = False
    return_hidden_states = False
    routed_experts = None
    finished_output = False
    stream = False

    @property
    def output_ids_through_stop(self):
        return self.output_ids[: self.finished_len]

    def finished(self):
        return True

    def init_incremental_detokenize(self):
        return [1, 2] + self.output_ids_through_stop, len(self.origin_input_ids)


def test_skip_tokenizer_stream_output_uses_output_ids_through_stop():
    scheduler = SimpleNamespace(
        skip_tokenizer_init=True,
        spec_algorithm=None,
        _comm_backend=None,
        send_to_detokenizer=_Sender(),
    )
    req = _Req()

    SchedulerOutputProcessorMixin.stream_output_generation(
        scheduler,
        [req],
        return_logprob=False,
        return_output_logprob_only=False,
    )

    out = scheduler.send_to_detokenizer.sent
    assert out.output_ids == [[10, 20, 30]]
    assert out.completion_tokens == [3]
    assert req.send_token_offset == 3
