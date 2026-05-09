import inspect

import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.managers.tp_worker import ModelWorker
from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sgl_jax.srt.speculative import eagle_draft_worker, eagle_worker, spec_utils
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


def test_eagle_worker_does_not_own_draft_prefill_method():
    assert hasattr(EagleDraftWorker, "draft_extend_for_prefill")
    assert hasattr(EagleDraftWorker, "capture_for_decode")
    assert not hasattr(EAGLEWorker, "forward_draft_extend")


def test_eagle_draft_worker_init_does_not_require_capture_callback():
    parameters = inspect.signature(EagleDraftWorker.__init__).parameters

    assert "capture_for_decode" not in parameters


def test_task5_spec_utils_exports_draft_decode_helpers():
    for name in (
        "topk_probs_from_logits",
        "fast_topk",
        "update_eagle_lists",
        "update_forward_batch_info",
        "select_top_k_tokens",
        "select_top_k_tokens_step_0",
        "select_top_k_tokens_step_greater_0",
    ):
        assert callable(getattr(spec_utils, name))


def test_task5_draft_decode_methods_owned_by_draft_worker():
    for name in ("draft", "draft_forward", "padding_for_decode", "get_padding_bs"):
        assert callable(getattr(EagleDraftWorker, name))


def test_task5_precompile_methods_owned_by_draft_worker_with_eagle_worker_wrappers():
    for name in (
        "run_spec_decode_precompile",
        "precompile_spec_extend",
        "precompile_spec_decode",
    ):
        assert callable(getattr(EagleDraftWorker, name))
        assert callable(getattr(EAGLEWorker, name))
        source = inspect.getsource(getattr(EAGLEWorker, name))
        assert "return self.draft_worker" in source
        assert "tqdm" not in source
        assert "product" not in source
        assert "generate_model_worker_batch" not in source


def test_task5_eagle_worker_verify_explicitly_accepts_verify_input():
    parameters = inspect.signature(EAGLEWorker.verify).parameters

    assert "verify_input" in parameters


def test_task5_eagle_worker_no_longer_owns_draft_decode_methods():
    for name in (
        "draft",
        "draft_forward",
        "padding_for_decode",
        "get_padding_bs",
        "copy_model_worker_batch_to_cpu",
    ):
        assert name not in EAGLEWorker.__dict__


def test_task6_eagle_worker_does_not_own_draft_extend_after_verify():
    assert hasattr(EagleDraftWorker, "draft_extend_for_decode")
    assert not hasattr(EAGLEWorker, "draft_extend_after_verify")


def test_task7_eagle_worker_is_orchestration_only():
    assert EAGLEWorker.__bases__ == (BaseSpecWorker,)
    assert ModelWorker not in EAGLEWorker.__mro__

    source = inspect.getsource(EAGLEWorker)

    forbidden = (
        "is_draft_worker=True",
        "self.draft_model_runner.forward",
        "build_tree_kernel_efficient(",
        "select_top_k_tokens(",
    )
    for text in forbidden:
        assert text not in source

    required = (
        "self.draft_worker.draft_extend_for_prefill",
        "self.draft_worker.draft(",
        "self.draft_worker.draft_extend_for_decode",
        "self.target_worker.forward_batch_generation",
    )
    for text in required:
        assert text in source


def test_task5_eagle_draft_worker_does_not_define_local_topk_helper():
    source = inspect.getsource(eagle_draft_worker)

    assert "def _topk_probs_from_logits" not in source


def test_task5_eagle_worker_does_not_define_moved_helper_functions():
    source = inspect.getsource(eagle_worker)

    for function_definition in (
        "def topk_probs_from_logits",
        "def fast_topk",
        "def update_eagle_lists",
        "def update_forward_batch_info",
        "def select_top_k_tokens",
        "def select_top_k_tokens_step_0",
        "def select_top_k_tokens_step_greater_0",
    ):
        assert function_definition not in source
