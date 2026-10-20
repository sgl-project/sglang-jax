"""Exercise selective EPD weight loading through the shared loader on CPU."""

import jax
import numpy as np
import pytest
from flax import nnx
from jax.sharding import AxisType, Mesh
from transformers import Qwen2_5_VLConfig

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration


@pytest.mark.parametrize(
    "encoder_only,language_only",
    [(True, False), (False, True), (False, False)],
    ids=["encoder", "language", "combined"],
)
def test_shared_loader_materializes_only_selected_submodules(encoder_only, language_only):
    hf_config = Qwen2_5_VLConfig(
        text_config={
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "vocab_size": 32,
        },
        vision_config={
            "hidden_size": 16,
            "intermediate_size": 32,
            "depth": 1,
            "num_heads": 2,
            "out_hidden_size": 128,
            "patch_size": 2,
            "temporal_patch_size": 1,
            "spatial_merge_size": 2,
            "window_size": 4,
            "fullatt_block_indexes": [0],
        },
    )
    hf_config.architectures = ["Qwen2_5_VLForConditionalGeneration"]
    hf_config.text_config.rope_parameters = {"rope_type": "default", "rope_theta": 10000.0}
    config = ModelConfig(
        "unused",
        hf_config=hf_config,
        encoder_only=encoder_only,
        language_only=language_only,
        model_weights="unused-weights",
        dtype="float32",
    )
    config._dummy_mode = True
    assert config.model_weights == "unused-weights"
    assert not hasattr(hf_config, "encoder_only")
    assert not hasattr(hf_config, "language_only")

    mesh = Mesh(
        np.array(jax.devices()[:1]).reshape(1, 1),
        ("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )
    with jax.set_mesh(mesh):
        model = nnx.eval_shape(
            lambda: Qwen2_5_VLForConditionalGeneration(
                config.hf_config, dtype=config.dtype, mesh=mesh
            )
        )
        assert hasattr(model, "model") == (not encoder_only)
        assert hasattr(model, "visual") == (not language_only)
        model.load_weights(config)

    # A missed mapping leaves an abstract parameter behind instead of a device array.
    parameters = jax.tree.leaves(nnx.state(model, nnx.Param))
    assert parameters
    assert all(isinstance(param, jax.Array) for param in parameters)
