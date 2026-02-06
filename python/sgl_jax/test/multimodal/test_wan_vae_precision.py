import unittest

import jax
import jax.numpy as jnp
import numpy as np
import torch
from diffusers import AutoencoderKLWan as DiffusersWan

from sgl_jax.srt.configs.load_config import LoadConfig
from sgl_jax.srt.model_loader.loader import get_model_loader
from sgl_jax.srt.multimodal.models.wan.vaes.wanvae import AutoencoderKLWan

MODEL_PATH = "/models/Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


class TestWanVaePrecision(unittest.TestCase):
    """Test wan vae decode precision: diffusers (CPU) vs JAX (TPU)."""

    @classmethod
    def setUpClass(cls):
        # --- diffusers model (CPU) ---
        cls.torch_vae = DiffusersWan.from_pretrained(
            MODEL_PATH,
            subfolder="vae",
            torch_dtype=torch.float32,
        )
        cls.torch_vae.eval()

        # --- JAX model (TPU) ---
        cls.mesh = jax.sharding.Mesh(jax.devices(), axis_names=("data",))
        model_loader = get_model_loader(
            load_config=LoadConfig(
                model_class=AutoencoderKLWan,
                sub_dir="vae",
            ),
            mesh=cls.mesh,
        )
        model_config = AutoencoderKLWan.get_config_class()()
        model_config.model_path = MODEL_PATH
        model_config.model_class = AutoencoderKLWan
        cls.jax_vae = model_loader.load_model(model_config=model_config)

    def test_decode_precision(self):
        """Test decode precision: 5 latent frames."""
        print("Testing decode precision (5 latent frames)...")
        latents = np.arange(1 * 5 * 3 * 4 * 16, dtype=np.float32).reshape(1, 5, 3, 4, 16)

        # diffusers (CPU): (B, C, T, H, W)
        latents_torch = torch.tensor(latents, dtype=torch.float32).permute(0, 4, 1, 2, 3)
        with torch.no_grad():
            torch_output = self.torch_vae.decode(latents_torch).sample.detach().numpy()
        torch_output = torch_output.transpose(0, 2, 3, 4, 1)  # -> (B, T, H, W, C)

        # JAX (TPU): (B, T, H, W, C)
        jax_output = np.array(self.jax_vae.decode(jnp.array(latents, dtype=jnp.float32)))

        print(f"Diffusers shape: {torch_output.shape}, JAX shape: {jax_output.shape}")
        max_diff = np.max(np.abs(torch_output - jax_output))
        print(f"Max absolute diff: {max_diff:.8f}")
        np.testing.assert_allclose(torch_output, jax_output, rtol=1e-4, atol=1e-4)

    def test_encode_precision(self):
        """Test encode precision: 5 video frames."""
        print("Testing encode precision (5 video frames)...")
        # Small spatial size for speed
        input_np = np.arange(1 * 5 * 24 * 32 * 3, dtype=np.float32).reshape(1, 5, 24, 32, 3)

        # diffusers (CPU): (B, C, T, H, W)
        input_torch = torch.tensor(input_np, dtype=torch.float32).permute(0, 4, 1, 2, 3)
        with torch.no_grad():
            torch_latents = self.torch_vae.encode(input_torch)
        # parameters: (B, C, T, H, W) -> (B, T, H, W, C)
        torch_output = (
            torch_latents.latent_dist.parameters.detach().numpy().transpose(0, 2, 3, 4, 1)
        )

        # JAX (TPU): (B, T, H, W, C)
        jax_latents = self.jax_vae.encode(jnp.array(input_np, dtype=jnp.float32))
        jax_output = np.array(jax_latents.parameters)

        print(f"Diffusers shape: {torch_output.shape}, JAX shape: {jax_output.shape}")
        max_diff = np.max(np.abs(torch_output - jax_output))
        print(f"Max absolute diff: {max_diff:.8f}")
        np.testing.assert_allclose(torch_output, jax_output, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
