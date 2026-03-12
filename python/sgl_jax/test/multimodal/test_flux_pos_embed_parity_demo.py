from __future__ import annotations

import os
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np

try:
    import torch
    from diffusers.models.embeddings import (
        get_1d_rotary_pos_embed as hf_get_1d_rotary_pos_embed,
    )
    from diffusers.models.transformers.transformer_flux import (
        FluxPosEmbed as HFFluxPosEmbed,
    )
except ImportError:  # pragma: no cover
    torch = None
    hf_get_1d_rotary_pos_embed = None
    HFFluxPosEmbed = None

try:
    import jax
    import jax.numpy as jnp
except ImportError:  # pragma: no cover
    jax = None
    jnp = None

if jax is not None:
    from sgl_jax.srt.multimodal.models.dits.flux import FluxPosEmbed as JaxFluxPosEmbed
    from sgl_jax.srt.multimodal.models.dits.flux import (
        _get_1d_rotary_pos_embed as jax_get_1d_rotary_pos_embed,
    )


@unittest.skipIf(
    torch is None or hf_get_1d_rotary_pos_embed is None or HFFluxPosEmbed is None or jax is None,
    "torch/diffusers/jax not installed",
)
class TestFluxPosEmbedParityDemo(unittest.TestCase):
    def _assert_pair_close(self, torch_pair, jax_pair, atol=1e-5, rtol=1e-5):
        torch_cos, torch_sin = torch_pair
        jax_cos, jax_sin = jax_pair
        np.testing.assert_allclose(
            torch_cos.detach().cpu().numpy(),
            np.asarray(jax_cos),
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            torch_sin.detach().cpu().numpy(),
            np.asarray(jax_sin),
            rtol=rtol,
            atol=atol,
        )

    def test_get_1d_rotary_pos_embed_parity(self):
        dim = 2048
        theta = 10000.0
        pos_torch = torch.arange(1024, dtype=torch.float32)
        pos_jax = jnp.asarray(pos_torch.detach().cpu().numpy())

        torch_pair = hf_get_1d_rotary_pos_embed(
            dim,
            pos_torch,
            theta=theta,
            use_real=True,
            repeat_interleave_real=True,
            freqs_dtype=torch.float64,
        )
        jax_pair = jax_get_1d_rotary_pos_embed(dim, pos_jax, theta=theta)

        self._assert_pair_close(torch_pair, jax_pair)

    def test_get_1d_rotary_pos_embed_concat_real_parity(self):
        dim = 128
        theta = 10000.0
        pos_torch = torch.arange(33, dtype=torch.float32)
        pos_jax = jnp.asarray(pos_torch.detach().cpu().numpy())

        torch_pair = hf_get_1d_rotary_pos_embed(
            dim,
            pos_torch,
            theta=theta,
            use_real=True,
            repeat_interleave_real=False,
            freqs_dtype=torch.float32,
        )
        jax_pair = jax_get_1d_rotary_pos_embed(
            dim,
            pos_jax,
            theta=theta,
            use_real=True,
            repeat_interleave_real=False,
            freqs_dtype=jnp.float32,
        )

        self._assert_pair_close(torch_pair, jax_pair)

    def test_get_1d_rotary_pos_embed_complex_parity(self):
        dim = 128
        theta = 10000.0
        pos_torch = torch.arange(17, dtype=torch.float32)
        pos_jax = jnp.asarray(pos_torch.detach().cpu().numpy())

        torch_freqs = hf_get_1d_rotary_pos_embed(
            dim,
            pos_torch,
            theta=theta,
            use_real=False,
            freqs_dtype=torch.float32,
        )
        jax_freqs = jax_get_1d_rotary_pos_embed(
            dim,
            pos_jax,
            theta=theta,
            use_real=False,
            freqs_dtype=jnp.float32,
        )

        np.testing.assert_allclose(
            torch_freqs.detach().cpu().numpy(),
            np.asarray(jax_freqs),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_flux_pos_embed_parity(self):
        theta = 10000
        axes_dim = [16, 56, 56]
        ids_torch = torch.tensor(
            [[0, 0, 0], [17, 29, 31], [89, 144, 233], [377, 610, 987]],
            dtype=torch.int64,
        )
        ids_jax = jnp.asarray(ids_torch.detach().cpu().numpy())

        torch_pos_embed = HFFluxPosEmbed(theta=theta, axes_dim=axes_dim).cpu()
        jax_pos_embed = JaxFluxPosEmbed(theta=theta, axes_dim=axes_dim)

        torch_pair = torch_pos_embed(ids_torch)
        jax_pair = jax_pos_embed(ids_jax)

        self._assert_pair_close(torch_pair, jax_pair)


if __name__ == "__main__":
    unittest.main()
