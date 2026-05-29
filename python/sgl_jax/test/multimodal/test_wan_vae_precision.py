import os
import unittest
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.configs.load_config import LoadConfig
from sgl_jax.srt.model_loader.loader import get_model_loader
from sgl_jax.srt.multimodal.common.ServerArgs import MultimodalServerArgs

# import torch
# from diffusers import AutoencoderKLWan as diffusersWan
from sgl_jax.srt.multimodal.models.wan.vaes.wanvae import AutoencoderKLWan


class TestWanVaePrecision(unittest.TestCase):
    """Test wan vae encode and decode precision"""

    @classmethod
    def setUpClass(cls):
        cls.mesh = jax.sharding.Mesh(jax.devices(), axis_names=("data",))
        cls.server_args = MultimodalServerArgs(
            model_path="/models/Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        )
        # cls.vae = diffusersWan.from_pretrained(
        #     cls.server_args.model_path,
        #     subfolder="vae",
        #     torch_dtype=torch.float32,
        # )
        # cls.vae.eval()
        cls.model_class = AutoencoderKLWan
        model_loader = get_model_loader(
            load_config=LoadConfig(
                model_class=cls.model_class,
                sub_dir="vae",
            ),
            mesh=cls.mesh,
        )

        model_config = cls.model_class.get_config_class()()
        model_config.model_path = cls.server_args.model_path
        model_config.model_class = cls.model_class
        cls.jax_vae = model_loader.load_model(
            model_config=model_config,
        )

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("JAX_PLATFORMS", None)

    def _get_diffusers_encode_output(self):
        # input = (
        # torch.tensor(np.arange(1 * 5 * 192 * 192 * 3), dtype=torch.float32)
        # .reshape(1, 5, 192, 192, 3)
        # .permute((0, 4, 1, 2, 3))
        # )
        # latents = self.vae.encode(input)
        # print(latents.latent_dist.parameters.shape)
        current_dir = str(Path(__file__).resolve().parent)
        return np.load(current_dir + "/data/wan_vae_diffusers_encode_output.npy")
        # return latents.latent_dist.parameters.detach().numpy()

    def _get_jax_encode_output(
        self,
    ):
        input = jnp.array(np.arange(1 * 5 * 192 * 192 * 3), dtype=jnp.float32).reshape(
            1, 5, 192, 192, 3
        )
        latents = self.jax_vae.encode(input)
        print(latents.parameters.shape)
        return latents.parameters.transpose((0, 4, 1, 2, 3))

    def _get_diffusers_decode_output(self):
        # latents = (
        #     torch.tensor(np.arange(1 * 5 * 3 * 4 * 16), dtype=torch.float32)
        #     .reshape(1, 5, 3, 4, 16)
        #     .permute((0, 4, 1, 2, 3))
        # )
        # y = self.vae.decode(latents)
        # print(y.sample.shape)
        current_dir = str(Path(__file__).resolve().parent)
        return np.load(current_dir + "/data/wan_vae_diffusers_decode_output.npy")
        # return y.sample.detach().numpy()

    def _get_jax_decode_output(self):
        latents = jnp.array(np.arange(1 * 5 * 3 * 4 * 16), dtype=jnp.float32).reshape(
            1, 5, 3, 4, 16
        )
        y = self.jax_vae.decode(latents)
        print(y.shape)
        return y.transpose((0, 4, 1, 2, 3))

    def test_encode_precision(self):
        print("Testing encode precision...")
        torch_output = self._get_diffusers_encode_output()
        jax_output = self._get_jax_encode_output()
        np.testing.assert_allclose(torch_output, jax_output, rtol=1e-5, atol=1e-5)

    def test_decode_precision(self):
        print("Testing decode precision...")
        torch_output = self._get_diffusers_decode_output()
        jax_output = self._get_jax_decode_output()
        np.testing.assert_allclose(torch_output, jax_output, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
