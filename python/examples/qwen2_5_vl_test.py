import os
from functools import reduce
import numpy as np
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig

import jax
import jax.numpy as jnp
from flax import nnx

from sgl_jax.srt.models.qwen2_5_vl import Qwen2_5_VisionTransformer


def qwen2_5_vl():
    os.environ["XLA_FLAGS"] = " ".join(["--xla_force_host_platform_device_count=8", os.environ.get("XLA_FLAGS", "")])

    config = Qwen2_5_VLVisionConfig(
        window_size=112,
        patch_size=14,
        fullatt_block_indexes=[7, 15, 23, 31],
        temporal_patch_size=2,
        spatial_merge_size=2,
        in_channels=3,
        hidden_size=1280,
        num_heads=16,
        depth=32,
        rms_norm_eps=1e-05,
        intermediate_size=3456,
        rope_theta=1000000,
        rope_scaling=None,
        head_dim=None,
    )
    dtype = jnp.float32
    rngs = nnx.Rngs(0)
    mesh = jax.sharding.Mesh(
        devices=np.array(jax.devices()).reshape(2, 4),
        axis_names=("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
    )
    jax.set_mesh(mesh)
    norm_eps = 1e-6
    grid_thw = ((1, 10, 10), (1, 10, 10))
    num_patches = reduce(lambda x, y: x + y,
                         [reduce(lambda x, y: x * y, thw) for thw in grid_thw],
                         )
    data = jax.random.normal(jax.random.PRNGKey(0),
                             shape=[num_patches,
                                    config.in_channels * config.temporal_patch_size * config.patch_size * config.patch_size],
                             )

    vit = Qwen2_5_VisionTransformer(
        config=config,
        norm_eps=norm_eps,
        dtype=dtype,
        rngs=rngs,
        mesh=mesh,
    )

    ret = vit(data, grid_thw)
    print(f"x shape: {data.shape}, grid_thw: {grid_thw}")
    print(f"result shape: {ret.shape}")


if __name__ == '__main__':
    qwen2_5_vl()
