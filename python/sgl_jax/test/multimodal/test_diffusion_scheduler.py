import dataclasses
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
from jax.lax import Precision

from sgl_jax.srt.multimodal.common.ServerArgs import MultimodalServerArgs
from sgl_jax.srt.multimodal.manager.schedule_batch import Req


# Small config for unit testing to avoid OOM
@dataclasses.dataclass(frozen=True)
class SmallWanModelConfig:
    """Small config for unit testing."""

    weights_dtype: jnp.dtype = jnp.bfloat16
    dtype: jnp.dtype = jnp.bfloat16
    precision = Precision.HIGHEST
    num_layers: int = 2  # Reduced from 30
    hidden_dim: int = 1536
    in_channels: int = 16
    out_channels: int = 16
    ffn_dim: int = 8960
    freq_dim: int = 256
    num_heads: int = 12
    attention_head_dim: int = 128
    text_dim: int = 4096
    image_dim: int = 4096
    max_text_len: int = 512
    num_frames: int = 1  # Reduced from 11
    latent_size: tuple[int, int] = (8, 8)  # Reduced from (60, 90)
    num_inference_steps: int = 30
    guidance_scale: float = 5.0
    patch_size: tuple[int, int, int] = (1, 2, 2)
    cross_attn_norm: bool = True
    qk_norm: str | None = "rms_norm_across_heads"
    epsilon: float = 1e-6
    added_kv_proj_dim: int | None = None
    rope_max_seq_len: int = 1024
    latent_input_dim: int = 16
    latent_output_dim: int = 16
    head_dim: int = 128
    text_embed_dim: int = 4096
    num_attention_heads: int = 12


class TestDiffusionScheduler(unittest.TestCase):
    """Test DiffusionScheduler full load and forward flow."""

    @classmethod
    def setUpClass(cls):
        cls.mesh = jax.sharding.Mesh(jax.devices(), ("data",))
        cls.server_args = MultimodalServerArgs(
            model_path="Wan-AI/Wan2.1-T2V-1.3B",
            download_dir="/tmp",
        )
        # Patch WanModelConfig with small config before importing DiffusionScheduler
        with patch(
            "sgl_jax.srt.multimodal.model_executor.diffusion.diffusion_model_runner.WanModelConfig",
            SmallWanModelConfig,
        ):
            from sgl_jax.srt.multimodal.manager.scheduler.diffusion_scheduler import (
                DiffusionScheduler,
            )

            with jax.default_device(jax.devices()[0]):
                cls.scheduler = DiffusionScheduler(
                    server_args=cls.server_args,
                    mesh=cls.mesh,
                    communication_backend=None,
                )

    def test_run_diffusion_step(self):
        """Test full scheduler forward pass."""
        # Create request with required fields
        # Use guidance_scale=1.0 to avoid CFG batch doubling (saves memory)
        req = Req(
            rid="test_001",
            prompt="a cat walking in the garden",
            prompt_embeds=jnp.zeros((1, 512, 4096), dtype=jnp.float32),
            num_inference_steps=2,
            guidance_scale=1.0,
        )

        # Run diffusion step
        self.scheduler.run_diffusion_step(req)

        # Verify latents are generated
        self.assertIsNotNone(req.latents)


if __name__ == "__main__":
    unittest.main()
