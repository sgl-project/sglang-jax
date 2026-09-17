"""Forward-pass tests for the Kimi-K2.5 vision tower on video inputs.

These build a reduced-depth model with random weights on an explicit CPU mesh.
The CPU mesh matters: it makes the Pallas attention run in interpret mode, so the
numbers are exact enough to assert the temporal-merge invariant below. (The Kimi
stage config also schedules the ViT on CPU.) On TPU the same comparison holds
only to roughly bfloat16 epsilon, which is kernel precision rather than a
property of the video path.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.sharding import Mesh

from sgl_jax.srt.multimodal.configs.kimi.kimi_k25_config import KimiK25ModelVitConfig
from sgl_jax.srt.multimodal.models.kimi_k25.kimi_k25_vit import Kimi_K25_VisionModel

DTYPE = jnp.float32
PATCH_SIZE = 14
MERGE_H, MERGE_W = 2, 2
NUM_LAYERS = 2


@pytest.fixture(scope="module")
def vision_model():
    config = KimiK25ModelVitConfig()
    # Hidden sizes stay at their real values so weight shapes remain
    # self-consistent; only the depth is cut so the test stays quick.
    config.vt_num_hidden_layers = NUM_LAYERS
    config.dtype = DTYPE

    mesh = Mesh(np.array(jax.devices("cpu")[:1]).reshape(1, 1), ("data", "model"))
    with mesh:
        model = Kimi_K25_VisionModel(config, dtype=DTYPE, rngs=nnx.Rngs(0), mesh=mesh)
    return model, mesh, config


def _patches(grid_thws, seed=0):
    total = sum(t * h * w for t, h, w in grid_thws)
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.standard_normal((total, 3, PATCH_SIZE, PATCH_SIZE)), dtype=DTYPE)


def _encode(vision_model, grid_thws, pixel_values=None):
    model, mesh, _ = vision_model
    values = _patches(grid_thws) if pixel_values is None else pixel_values
    with mesh:
        return model.encode_vision(values, grid_thws)


@pytest.mark.parametrize(
    "grid_thws",
    [
        [(1, 4, 6)],
        [(1, 4, 6), (1, 8, 8)],
        [(4, 4, 6)],
        [(16, 2, 2)],
        [(1, 4, 6), (4, 4, 4)],
        [(3, 2, 2), (4, 2, 2)],
        [(1, 4, 4), (3, 2, 2), (2, 2, 4)],
    ],
)
def test_encode_vision_token_count_is_frame_independent(vision_model, grid_thws):
    _, _, config = vision_model
    out = _encode(vision_model, grid_thws)

    # Frames are pooled away, so only the spatial grid decides the token count.
    expected_tokens = sum((h // MERGE_H) * (w // MERGE_W) for _, h, w in grid_thws)
    assert out.shape == (expected_tokens, config.text_hidden_size)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_duplicated_frame_video_matches_single_image(vision_model):
    """A video of N identical frames must encode like the single image.

    Both the position embedding and the 2D RoPE repeat per frame, and softmax
    over exactly duplicated keys is unchanged, so temporal pooling has to give
    back the image embedding. This is what catches a mis-weighted merge.
    """
    height, width, frames = 4, 6, 3
    image_patches = _patches([(1, height, width)], seed=7)
    video_patches = jnp.tile(image_patches, (frames, 1, 1, 1))

    image_out = _encode(vision_model, [(1, height, width)], image_patches)
    video_out = _encode(vision_model, [(frames, height, width)], video_patches)

    assert image_out.shape == video_out.shape
    np.testing.assert_allclose(
        np.asarray(video_out, dtype=np.float32),
        np.asarray(image_out, dtype=np.float32),
        rtol=1e-4,
        atol=1e-4,
    )
