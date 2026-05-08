import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sgl_jax.srt.speculative.eagle_util import EagleDraftInput, EagleVerifyInput
from sgl_jax.srt.speculative.spec_info import SpecInput, SpecInputType


def test_eagle_draft_input_is_spec_input():
    draft_input = EagleDraftInput.create_idle_input(
        hidden_size=16,
        dtype=np.float32,
        topk=1,
        capture_hidden_mode=CaptureHiddenMode.LAST,
    )

    assert isinstance(draft_input, SpecInput)
    assert draft_input.spec_input_type == SpecInputType.EAGLE_DRAFT
    assert draft_input.is_draft_input()
    assert not draft_input.is_verify_input()
    assert draft_input.get_spec_adjust_token_coefficient() == (1, 1)


def test_eagle_verify_input_is_spec_input():
    verify_input = EagleVerifyInput(
        draft_token=jnp.empty((0,), dtype=jnp.int32),
        custom_mask=jnp.empty((0,), dtype=jnp.bool_),
        positions=jnp.empty((0,), dtype=jnp.int32),
        retrive_index=jnp.empty((0, 0), dtype=jnp.int32),
        retrive_next_token=jnp.empty((0, 0), dtype=jnp.int32),
        retrive_next_sibling=jnp.empty((0, 0), dtype=jnp.int32),
        retrive_cum_len=None,
        seq_lens_cpu=np.empty((0,), dtype=np.int32),
        spec_steps=3,
        topk=1,
        draft_token_num=4,
        seq_lens_sum=0,
        capture_hidden_mode=CaptureHiddenMode.FULL,
    )

    assert isinstance(verify_input, SpecInput)
    assert verify_input.spec_input_type == SpecInputType.EAGLE_VERIFY
    assert not verify_input.is_draft_input()
    assert verify_input.is_verify_input()
    assert verify_input.get_spec_adjust_token_coefficient() == (4, 4)
