"""SpecInput protocol conformance for EagleDraftInput / EagleVerifyInput."""

import numpy as np

from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sgl_jax.srt.speculative.eagle_util import EagleDraftInput, EagleVerifyInput
from sgl_jax.srt.speculative.spec_info import SpecInput


def test_eagle_draft_input_is_spec_input():
    di = EagleDraftInput(
        accept_length_cpu=np.array([2, 3, 1], dtype=np.int32),
        allocate_lens=np.array([10, 12, 8], dtype=np.int32),
    )
    assert isinstance(di, SpecInput)
    assert di.is_draft_input() and not di.is_verify_input()
    assert (di.get_logical_token_num(bs=3) == np.array([2, 3, 1])).all()
    assert di.get_verify_token_num(bs=3) == 0
    assert (di.get_allocated_token_num() == np.array([10, 12, 8])).all()
    assert di.get_spec_adjust_token_coefficient() >= 1


def test_eagle_verify_input_is_spec_input():
    vi = EagleVerifyInput(
        draft_token=np.zeros(8, dtype=np.int32),
        custom_mask=np.zeros(1, dtype=np.int32),
        positions=np.zeros(8, dtype=np.int32),
        retrive_index=np.zeros(8, dtype=np.int32),
        retrive_next_token=np.zeros(8, dtype=np.int32),
        retrive_next_sibling=np.zeros(8, dtype=np.int32),
        retrive_cum_len=np.zeros(3, dtype=np.int32),
        seq_lens_cpu=np.array([5, 7], dtype=np.int32),
        spec_steps=3,
        topk=1,
        draft_token_num=4,
        seq_lens_sum=12,
        capture_hidden_mode=CaptureHiddenMode.FULL,
    )
    assert isinstance(vi, SpecInput)
    assert vi.is_verify_input() and not vi.is_draft_input()
    assert vi.get_verify_token_num(bs=2) == 8
    assert vi.get_spec_adjust_token_coefficient() == 4
    assert vi.get_allocated_token_num() is None


def test_three_token_counts_are_distinct():
    """RFC #1053: logical / allocated / verify must be exposed independently."""
    di = EagleDraftInput(
        accept_length_cpu=np.array([2, 2], dtype=np.int32),
        allocate_lens=np.array([100, 100], dtype=np.int32),
    )
    assert int(di.get_logical_token_num(bs=2).sum()) == 4
    assert int(di.get_allocated_token_num().sum()) == 200
    assert di.get_verify_token_num(bs=2) == 0
