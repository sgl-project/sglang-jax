from types import SimpleNamespace

from sgl_jax.srt.layers.logits_processor import LogitsProcessorOutput
from sgl_jax.srt.managers.scheduler_output_processor_mixin import (
    SchedulerOutputProcessorMixin,
)


class _DummyScheduler(SchedulerOutputProcessorMixin):
    pass


def test_multi_item_input_logprobs_are_aligned_to_delimiter_minus_one():
    scheduler = _DummyScheduler()
    scheduler.model_config = SimpleNamespace(vocab_size=32000)

    req = SimpleNamespace(
        is_multi_item_scoring=True,
        multi_item_scoring_delimiter=9,
        input_token_logprobs=None,
        temp_input_top_logprobs_val=None,
        temp_input_top_logprobs_idx=None,
        temp_input_token_ids_logprobs_val=None,
        temp_input_token_ids_logprobs_idx=None,
        input_token_logprobs_val=None,
        input_token_logprobs_idx=None,
        input_top_logprobs_val=None,
        input_top_logprobs_idx=None,
        input_token_ids_logprobs_val=None,
        input_token_ids_logprobs_idx=None,
        top_logprobs_num=0,
        token_ids_logprob=[1, 2],
        origin_input_ids=[101, 9, 201, 9, 301, 9],  # query<d>item1<d>item2<d>
        logprob_start_len=0,
        return_logprob=True,
    )

    # Raw per-token values are indexed by current token position.
    # After alignment, delimiter positions (1,3,5) should use raw values (0,2,4).
    raw_logprobs = [0, 1, 2, 3, 4, 5]
    raw_token_id_vals = [["v0"], ["v1"], ["v2"], ["v3"], ["v4"], ["v5"]]
    raw_token_id_idxs = [["i0"], ["i1"], ["i2"], ["i3"], ["i4"], ["i5"]]
    output = LogitsProcessorOutput(
        next_token_logits=None,
        input_token_logprobs=raw_logprobs,
        input_token_ids_logprobs_val=[raw_token_id_vals],
        input_token_ids_logprobs_idx=[raw_token_id_idxs],
    )

    scheduler.add_input_logprob_return_values(
        i=0,
        req=req,
        output=output,
        logprob_pt=0,
        num_input_logprobs=len(raw_logprobs),
        last_prefill_chunk=True,
    )

    assert req.input_token_logprobs_val == [0, 2, 4]
    assert req.input_token_ids_logprobs_val == [["v0"], ["v2"], ["v4"]]
    assert req.input_token_ids_logprobs_idx == [["i0"], ["i2"], ["i4"]]
