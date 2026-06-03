import os
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx
from mimo_vision_test_utils import (
    MIMO_MODEL_PATH,
    load_hf_vision_transformer,
    load_vision_config_dict,
    make_checkpoint_vision_config,
)

from sgl_jax.srt.utils.mesh_utils import create_device_mesh

E2E_GRID_CASES = {
    "mixed_temporal_and_multiple_segments": ((1, 4, 6), (2, 2, 2)),
    "regular_temporal_video": ((2, 4, 4),),
    "large_grid_crosses_window_boundary": ((1, 16, 16),),
    "wide_aspect_ratio_col_order": ((1, 2, 32),),
}


@unittest.skipUnless(
    os.path.exists(MIMO_MODEL_PATH), f"MiMo model path not found: {MIMO_MODEL_PATH}"
)
class TestMiMoVisionEncoderE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sgl_jax.srt.multimodal.models.mimo_vision.vision_encoder import (
            MiMoVisionTransformer,
        )

        vision_config = load_vision_config_dict()
        cls.config = make_checkpoint_vision_config(vision_config)
        cls.hf_transformer = load_hf_vision_transformer(MIMO_MODEL_PATH, cls.config)
        cls.mesh = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])
        with jax.set_mesh(cls.mesh):
            cls.jax_transformer = MiMoVisionTransformer(
                cls.config,
                norm_eps=1e-6,
                dtype=jnp.float32,
                rngs=nnx.Rngs(0),
            )
            cls.jax_transformer.load_weights_from_safetensors(MIMO_MODEL_PATH, cls.config)

    def _run(self, grid_thw):
        config = self.config
        torch_grid_thw = torch.tensor(grid_thw, dtype=torch.int32)

        patch_dim = (
            config.in_channels * config.temporal_patch_size * config.patch_size * config.patch_size
        )
        total_tokens = sum(t * h * w for t, h, w in grid_thw)
        rng = np.random.default_rng(13)
        pixel_values = rng.normal(size=(total_tokens, patch_dim)).astype(np.float32)

        with torch.no_grad():
            torch_output = self.hf_transformer(torch.from_numpy(pixel_values), torch_grid_thw)
            if isinstance(torch_output, tuple):
                torch_output = torch_output[0]

        with jax.set_mesh(self.mesh), jax.default_matmul_precision("highest"):
            jax_output = self.jax_transformer(jnp.asarray(pixel_values), grid_thw)

        np.testing.assert_allclose(
            np.asarray(jax_output),
            torch_output.detach().cpu().numpy(),
            rtol=1e-3,
            atol=1e-5,
        )

    def test_checkpoint_weights_match_hf_remote_code_across_grid_shapes(self):
        for case_name, grid_thw in E2E_GRID_CASES.items():
            with self.subTest(case=case_name, grid_thw=grid_thw):
                self._run(grid_thw)


if __name__ == "__main__":
    unittest.main()
