from __future__ import annotations

import os
import re
import unittest
from pathlib import Path

try:
    import torch
    from diffusers.models.autoencoders.autoencoder_kl import (
        AutoencoderKL as HFAutoencoderKL,
    )
except ImportError:  # pragma: no cover
    torch = None
    HFAutoencoderKL = None

try:
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx
except ImportError:  # pragma: no cover
    jax = None
    jnp = None
    np = None
    nnx = None

if jax is not None:
    from sgl_jax.srt.multimodal.configs.vaes.flux_vae_config import FluxVAEConfig
    from sgl_jax.srt.multimodal.models.vaes.autoencoder import AutoencoderKL
    from sgl_jax.srt.multimodal.models.vaes.flux_vae_weight_mappings import to_mappings


FLUX1_VAE_ROOT = Path(os.environ.get("FLUX1_VAE_ROOT", "/models/FLUX1.0/vae"))


@unittest.skipIf(
    torch is None or HFAutoencoderKL is None or jax is None or nnx is None,
    "torch/diffusers/jax/flax not installed",
)
class TestFluxAutoencoder(unittest.TestCase):
    @staticmethod
    def _make_image_like_input(
        *,
        batch_size: int = 1,
        height: int = 32,
        width: int = 32,
    ) -> jax.Array:
        # Use a bounded image-like signal instead of pure Gaussian noise so CPU parity
        # tests better resemble the normalized [-1, 1] inputs a VAE sees in practice.
        y = jnp.linspace(-1.0, 1.0, height, dtype=jnp.float32)
        x = jnp.linspace(-1.0, 1.0, width, dtype=jnp.float32)
        yy, xx = jnp.meshgrid(y, x, indexing="ij")

        channel_r = 0.65 * xx + 0.35 * jnp.sin(jnp.pi * yy)
        channel_g = 0.55 * yy + 0.25 * jnp.cos(2.0 * jnp.pi * xx)
        channel_b = 0.5 * jnp.sin(jnp.pi * (xx + yy)) + 0.25 * (xx * yy)
        image = jnp.stack((channel_r, channel_g, channel_b), axis=0)
        image = jnp.clip(image, -1.0, 1.0)

        batch = []
        for i in range(batch_size):
            shifted = jnp.roll(image, shift=i, axis=2)
            contrast = 1.0 - 0.08 * i
            bias = -0.05 + 0.05 * i
            batch.append(jnp.clip(shifted * contrast + bias, -1.0, 1.0))

        return jnp.stack(batch, axis=0)

    @staticmethod
    def _set_jax_param(model: AutoencoderKL, path: str, value: np.ndarray) -> None:
        current = nnx.state(model)
        for key in path.split("."):
            current = current[int(key)] if key.isdigit() else current[key]
        current[...] = jnp.asarray(value)

    def _copy_hf_weights_to_jax(
        self,
        hf_model: HFAutoencoderKL,
        jax_model: AutoencoderKL,
        config: FluxVAEConfig,
    ) -> None:
        pt_state_dict = hf_model.state_dict()
        mappings = to_mappings(config)
        for hf_key, tensor in pt_state_dict.items():
            mapping = None
            target_path = None

            for pattern, candidate in mappings.items():
                if "*" not in pattern:
                    if hf_key == pattern:
                        mapping = candidate
                        target_path = candidate.target_path
                        break
                    continue

                match = re.fullmatch(re.escape(pattern).replace(r"\*", r"(.*?)"), hf_key)
                if match is None:
                    continue
                mapping = candidate
                target_path = candidate.target_path.replace("*", "{}").format(*match.groups())
                break

            if mapping is None or target_path is None:
                continue

            weight = tensor.detach().cpu().float().numpy()
            if mapping.transpose_axes is not None and not hf_key.endswith(".bias"):
                weight = np.transpose(weight, mapping.transpose_axes)
            elif mapping.transpose and not hf_key.endswith(".bias"):
                weight = np.transpose(weight, (1, 0))

            self._set_jax_param(jax_model, target_path, weight)

    @staticmethod
    def _build_hf_model(config: FluxVAEConfig) -> HFAutoencoderKL:
        return HFAutoencoderKL(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            down_block_types=tuple(config.down_block_types),
            up_block_types=tuple(config.up_block_types),
            block_out_channels=tuple(config.block_out_channels),
            layers_per_block=config.layers_per_block,
            act_fn=config.act_fn,
            latent_channels=config.latent_channels,
            norm_num_groups=config.norm_num_groups,
            sample_size=config.sample_size,
            scaling_factor=config.scaling_factor,
            shift_factor=config.shift_factor,
            latents_mean=config.latents_mean,
            latents_std=config.latents_std,
            force_upcast=config.force_upcast,
            use_quant_conv=config.use_quant_conv,
            use_post_quant_conv=config.use_post_quant_conv,
            mid_block_add_attention=config.mid_block_add_attention,
        ).eval()

    def _assert_jax_matches_hf(
        self,
        jax_model: AutoencoderKL,
        hf_model: HFAutoencoderKL,
        x_nchw: jax.Array,
    ) -> None:
        x_np = np.array(x_nchw, copy=True)
        x_torch = torch.from_numpy(x_np)

        with torch.no_grad():
            hf_posterior = hf_model.encode(x_torch).latent_dist
            hf_decoded = hf_model.decode(hf_posterior.mode()).sample

        jax_latents = jax_model.encode(x_nchw)
        jax_decoded = jax_model.decode(jax_latents)

        self._assert_outputs_close(
            torch_output=hf_posterior.mode(),
            jax_output=jax_latents,
            rtol=1e-4,
            atol=1e-4,
        )
        self._assert_outputs_close(
            torch_output=hf_decoded,
            jax_output=jax_decoded,
            rtol=1e-4,
            atol=1e-4,
        )

    def _assert_outputs_close(
        self,
        *,
        torch_output: torch.Tensor,
        jax_output: jax.Array,
        rtol: float,
        atol: float,
    ) -> None:
        torch_np = torch_output.detach().cpu().numpy()
        jax_np = np.asarray(jax_output)

        self.assertEqual(jax_np.shape, torch_np.shape)
        np.testing.assert_allclose(torch_np, jax_np, rtol=rtol, atol=atol)

    @unittest.skipUnless(
        (FLUX1_VAE_ROOT / "config.json").is_file(),
        "local FLUX VAE config not found",
    )
    def test_flux_autoencoder_random_init_matches_hf_on_cpu(self):
        # For random initialization, copy HF parameters directly into the JAX model and
        # compare outputs on CPU without going through a checkpoint file.
        config = FluxVAEConfig.from_pretrained(FLUX1_VAE_ROOT)
        hf_model = self._build_hf_model(config)
        jax_model = AutoencoderKL(config)
        self._copy_hf_weights_to_jax(hf_model, jax_model, config)
        x = self._make_image_like_input()
        self._assert_jax_matches_hf(jax_model, hf_model, x)

    @unittest.skipUnless(
        (FLUX1_VAE_ROOT / "diffusion_pytorch_model.safetensors").is_file(),
        "local FLUX VAE weights not found",
    )
    def test_flux_autoencoder_loaded_weights_match_hf_on_cpu(self):
        # Real checkpoint parity on CPU: JAX and HF should produce the same latent moments
        # and reconstruction when fed the same input.
        config = FluxVAEConfig.from_pretrained(FLUX1_VAE_ROOT)
        config.model_path = str(FLUX1_VAE_ROOT)
        jax_model = AutoencoderKL(config)
        jax_model.load_weights(config)
        hf_model = HFAutoencoderKL.from_pretrained(FLUX1_VAE_ROOT).eval()

        x = self._make_image_like_input()
        self._assert_jax_matches_hf(jax_model, hf_model, x)


if __name__ == "__main__":
    unittest.main()
