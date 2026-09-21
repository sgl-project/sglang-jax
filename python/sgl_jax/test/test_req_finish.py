import pytest

from sgl_jax.srt.managers.schedule_batch import Req
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
