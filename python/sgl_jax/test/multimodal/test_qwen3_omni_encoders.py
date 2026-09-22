"""Offline Omni encoder parity using tiny HF models and the real weight loader."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from flax import nnx
from safetensors.torch import save_file
from transformers import Qwen3OmniMoeAudioEncoderConfig, Qwen3OmniMoeVisionEncoderConfig
from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf

from sgl_jax.srt.multimodal.models.qwen3_omni_moe.audio_encoder import (
    Qwen3OmniMoeAudioEncoder,
)
from sgl_jax.srt.multimodal.models.qwen3_omni_moe.qwen3_omni_thinker_embedding import (
    Qwen3OmniMoeThinkerEmbedding,
)
from sgl_jax.srt.multimodal.models.qwen3_omni_moe.vision_encoder import (
    Qwen3OmniMoeVisionEncoder,
)
from sgl_jax.srt.utils.mesh_utils import create_device_mesh
from sgl_jax.srt.utils.weight_utils import WeightLoader


def _assert_close(actual, expected):
    assert actual.dtype == (jnp.bfloat16 if expected.dtype == torch.bfloat16 else jnp.float32)
    assert np.isfinite(np.asarray(actual, dtype=np.float32)).all()
    assert torch.isfinite(expected).all()
    # Same tolerances as the existing checkpoint-based Omni encoder tests.
    atol, rtol = (1e-2, 1e-1) if expected.dtype == torch.bfloat16 else (1e-5, 1e-3)
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float32), expected.float().numpy(), atol=atol, rtol=rtol
    )


@pytest.fixture(scope="module", params=[jnp.float32, jnp.bfloat16])
def encoders(tmp_path_factory, request):
    dtype = request.param
    torch_dtype = torch.float32 if dtype == jnp.float32 else torch.bfloat16
    mesh = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])
    configs = {
        "audio_tower": Qwen3OmniMoeAudioEncoderConfig(
            encoder_layers=2,
            encoder_attention_heads=8,
            encoder_ffn_dim=64,
            d_model=32,
            output_dim=32,
            downsample_hidden_size=8,
            n_window=50,
            n_window_infer=800,
            conv_chunksize=2,
            _attn_implementation="eager",
        ),
        "visual": Qwen3OmniMoeVisionEncoderConfig(
            depth=2,
            hidden_size=32,
            intermediate_size=64,
            num_heads=8,
            patch_size=2,
            out_hidden_size=32,
            num_position_embeddings=16,
            deepstack_visual_indexes=[0],
            _attn_implementation="eager",
        ),
    }
    pairs = {}
    for name, hf_cls, jax_cls in [
        ("audio_tower", hf.Qwen3OmniMoeAudioEncoder, Qwen3OmniMoeAudioEncoder),
        ("visual", hf.Qwen3OmniMoeVisionEncoder, Qwen3OmniMoeVisionEncoder),
    ]:
        with torch.random.fork_rng():
            torch.manual_seed(0)
            reference = hf_cls(configs[name]).eval().to(torch_dtype)
        wrapper = nnx.Module()
        with jax.set_mesh(mesh):
            encoder = jax_cls(configs[name], mesh=mesh, dtype=dtype, rngs=nnx.Rngs(0))
            setattr(wrapper, name, encoder)
            mapping = getattr(Qwen3OmniMoeThinkerEmbedding, f"_create_{name}_weight_mappings")(
                configs[name]
            )
            weights = {
                f"thinker.{name}.{key}": value for key, value in reference.state_dict().items()
            }
            assert set(weights) == set(mapping)
            directory = tmp_path_factory.mktemp(name)
            save_file(weights, directory / "model.safetensors")
            WeightLoader(
                wrapper, SimpleNamespace(model_path=str(directory)), mesh, dtype
            ).load_weights_from_safetensors(mapping, validate_checkpoint_coverage=True)
        pairs[name] = (reference, encoder)
    return mesh, pairs


@pytest.mark.parametrize("grid", [[[1, 2, 2]], [[1, 4, 4]], [[2, 4, 4]], [[1, 6, 4], [1, 4, 2]]])
def test_vision_forward_matches_hf(encoders, grid):
    mesh, pairs = encoders
    reference, encoder = pairs["visual"]
    grid = np.asarray(grid, dtype=np.int32)
    patches = (
        np.random.default_rng(0).normal(size=(np.prod(grid, axis=1).sum(), 24)).astype(np.float32)
    )
    with torch.no_grad(), jax.set_mesh(mesh), jax.default_matmul_precision("highest"):
        expected = reference(torch.from_numpy(patches).to(reference.dtype), torch.from_numpy(grid))
        actual = encoder(jnp.asarray(patches, dtype=encoder.dtype), jnp.asarray(grid))
    for field in ("last_hidden_state", "pooler_output"):
        _assert_close(actual[field], getattr(expected, field))
    for result, feature in zip(
        actual["deepstack_features"], expected.deepstack_features, strict=True
    ):
        _assert_close(result, feature)


@pytest.mark.parametrize(
    "lengths",
    [[1], [7], [8], [9], [99], [100], [101], [799], [800], [801], [1, 101, 800, 9]],
)
def test_audio_forward_matches_hf(encoders, lengths):
    mesh, pairs = encoders
    reference, encoder = pairs["audio_tower"]
    features = np.random.default_rng(0).normal(size=(128, sum(lengths))).astype(np.float32)
    with torch.no_grad(), jax.set_mesh(mesh), jax.default_matmul_precision("highest"):
        # Use native CPU convolutions: oneDNN can produce NaNs for bf16
        # single-frame outputs on TPU hosts after JAX initialization.
        with torch.backends.mkldnn.flags(enabled=False):
            expected = reference(
                torch.from_numpy(features).to(reference.dtype), torch.tensor(lengths)
            ).last_hidden_state
        actual = encoder(jnp.asarray(features, dtype=encoder.dtype), jnp.asarray(lengths))
    _assert_close(actual, expected)


def test_audio_non_checkpoint_window_size(encoders, monkeypatch):
    mesh, pairs = encoders
    _, encoder = pairs["audio_tower"]
    # The HF default is 100, while released Omni checkpoints use 50. CNN
    # lengths must follow the actual chunk size, not a hard-coded 100 frames.
    monkeypatch.setattr(encoder, "n_window", 100)
    with jax.set_mesh(mesh):
        output = encoder(jnp.zeros((128, 256), dtype=encoder.dtype), jnp.asarray([256]))
    assert output.shape == (25 + 7, 32)  # ceil(200/8) + ceil(56/8)
