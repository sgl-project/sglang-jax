from types import SimpleNamespace
from unittest.mock import Mock

import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.multimodal.manager.scheduler.encoder_scheduler import EncoderScheduler
from sgl_jax.srt.multimodal.manager.stage import get_model_class, get_scheduler_class
from sgl_jax.srt.multimodal.manager.utils import load_stage_configs_from_yaml
from sgl_jax.srt.multimodal.model_executor.encoder.encoder_model_runner import (
    EncoderModelRunner,
)
from sgl_jax.srt.multimodal.model_executor.encoder.encoder_model_worker import (
    EncoderModelWorker,
)
from sgl_jax.srt.multimodal.models.encoders.t5 import T5EncoderModel, UMT5EncoderModel
from sgl_jax.srt.multimodal.models.static_configs import get_stage_config_path


@pytest.mark.parametrize(
    "model_path",
    [
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    ],
)
def test_wan_text_encoder_uses_encoder_pipeline(model_path):
    stage = load_stage_configs_from_yaml(get_stage_config_path(model_path))[0]

    assert get_scheduler_class(stage.scheduler) is EncoderScheduler
    assert get_model_class(stage.model_class) is UMT5EncoderModel


@pytest.mark.parametrize("max_length", [8, 226, 512, 768, 2048])
def test_encoder_precompile_covers_every_runtime_length(max_length):
    runner = object.__new__(EncoderModelRunner)
    runner.server_args = SimpleNamespace(model_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    spec = SimpleNamespace(model_class=UMT5EncoderModel, max_length=max_length)

    precompile_lengths = runner.get_precompile_lengths(spec, encoder_idx=0)

    assert precompile_lengths == sorted(set(precompile_lengths))
    assert all(
        runner._select_precompiled_max_length(actual_length, max_length) in precompile_lengths
        for actual_length in range(1, max_length + 1)
    )


def test_flux_t5_precompiles_only_its_fixed_runtime_length():
    runner = object.__new__(EncoderModelRunner)
    runner.server_args = SimpleNamespace(model_path="black-forest-labs/FLUX.1-dev")
    spec = SimpleNamespace(model_class=T5EncoderModel, max_length=512)

    assert runner.get_precompile_lengths(spec, encoder_idx=1) == [512]


def test_encoder_worker_precompiles_every_runtime_bucket():
    runner = object.__new__(EncoderModelRunner)
    runner.server_args = SimpleNamespace(model_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    spec = SimpleNamespace(
        model_class=UMT5EncoderModel,
        max_length=512,
        jitted_forward=Mock(),
    )
    runner.encoder_specs = [spec]
    worker = object.__new__(EncoderModelWorker)
    worker.model_runner = runner

    worker.run_precompile()

    expected_lengths = runner.get_precompile_lengths(spec, encoder_idx=0)
    actual_lengths = [call.args[0].shape[1] for call in spec.jitted_forward.call_args_list]
    assert actual_lengths == expected_lengths


def test_wan_t5_bucket_padding_is_zeroed():
    runner = object.__new__(EncoderModelRunner)
    runner.server_args = SimpleNamespace(model_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    spec = SimpleNamespace(model_class=UMT5EncoderModel)
    hidden_states = jnp.arange(1, 17, dtype=jnp.float32).reshape(1, 4, 4)
    tokenizer_mask = jnp.array([[1, 1, 0, 0]], dtype=jnp.int32)

    model_mask = runner._get_model_attention_mask(spec, tokenizer_mask, encoder_idx=0)
    result = runner.t5_postprocess_text({"last_hidden_state": hidden_states}, model_mask)

    np.testing.assert_array_equal(result[:, :2], hidden_states[:, :2])
    np.testing.assert_array_equal(result[:, 2:], jnp.zeros_like(hidden_states[:, 2:]))


def test_flux_t5_effective_mask_preserves_padded_outputs():
    runner = object.__new__(EncoderModelRunner)
    runner.server_args = SimpleNamespace(model_path="black-forest-labs/FLUX.1-dev")
    spec = SimpleNamespace(model_class=T5EncoderModel)
    hidden_states = jnp.arange(1, 17, dtype=jnp.float32).reshape(1, 4, 4)
    tokenizer_mask = jnp.array([[1, 1, 0, 0]], dtype=jnp.int32)

    model_mask = runner._get_model_attention_mask(spec, tokenizer_mask, encoder_idx=1)
    result = runner.t5_postprocess_text({"last_hidden_state": hidden_states}, model_mask)

    np.testing.assert_array_equal(model_mask, jnp.ones_like(tokenizer_mask))
    np.testing.assert_array_equal(result, hidden_states)
