import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.managers.tp_worker import ModelWorker
from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sgl_jax.srt.speculative.base_spec_worker import BaseDraftWorker, BaseSpecWorker
from sgl_jax.srt.speculative.eagle_draft_worker import EagleDraftWorker
from sgl_jax.srt.speculative.eagle_util import EagleDraftInput, EagleVerifyInput
from sgl_jax.srt.speculative.eagle_worker import EAGLEWorker
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


def test_eagle_draft_worker_implements_base_contract():
    assert issubclass(EagleDraftWorker, BaseDraftWorker)
    assert EagleDraftWorker.__mro__.index(ModelWorker) < EagleDraftWorker.__mro__.index(
        BaseDraftWorker
    )
    for name in ("draft", "draft_extend_for_prefill", "draft_extend_for_decode"):
        assert callable(getattr(EagleDraftWorker, name))


def test_eagle_worker_implements_base_contract():
    assert issubclass(EAGLEWorker, BaseSpecWorker)
    for name in (
        "target_worker",
        "draft_worker",
        "forward_batch_speculative_generation",
        "verify",
    ):
        assert hasattr(EAGLEWorker, name)


def test_eagle_draft_worker_generate_model_worker_batch_delegates_to_compilation_manager():
    class FakeCompilationManager:
        def __init__(self):
            self.calls = []

        def generate_model_worker_batch(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return "batch"

    worker = EagleDraftWorker.__new__(EagleDraftWorker)
    worker.compilation_manager = FakeCompilationManager()

    result = worker.generate_model_worker_batch(1, "arg", mode="decode")

    assert result == "batch"
    assert worker.compilation_manager.calls == [((1, "arg"), {"mode": "decode"})]
