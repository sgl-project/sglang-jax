from __future__ import annotations

import os
import unittest
from contextlib import nullcontext

import numpy as np

try:
    import torch
    from diffusers.models.transformers.transformer_flux import (
        FluxSingleTransformerBlock as HFFluxSingleTransformerBlock,
    )
    from diffusers.models.transformers.transformer_flux import (
        FluxTransformerBlock as HFFluxTransformerBlock,
    )
except ImportError:  # pragma: no cover
    torch = None
    HFFluxSingleTransformerBlock = None
    HFFluxTransformerBlock = None

try:
    import jax
    import jax.numpy as jnp
except ImportError:  # pragma: no cover
    jax = None
    jnp = None
os.environ.setdefault("JAX_PLATFORMS", "cpu")
for _tpu_env in ("TPU_ACCELERATOR_TYPE", "TPU_WORKER_HOSTNAMES"):
    os.environ.pop(_tpu_env, None)
if jax is not None:
    from sgl_jax.srt.multimodal.models.dits.flux import (
        FluxSingleTransformerBlock as JaxFluxSingleTransformerBlock,
    )
    from sgl_jax.srt.multimodal.models.dits.flux import (
        FluxTransformerBlock as JaxFluxTransformerBlock,
    )
    from sgl_jax.srt.multimodal.models.dits.flux_weights_mapping import to_mappings


def _copy_hf_state_dict_to_jax(*args, **kwargs):
    from sgl_jax.test.multimodal.test_flux_transformer_2d_model_parity_demo import (
        copy_hf_state_dict_to_jax,
    )

    return copy_hf_state_dict_to_jax(*args, **kwargs)


def _make_mesh():
    devices = np.array(jax.devices("cpu")[:1]).reshape((1, 1))
    try:
        return jax.sharding.Mesh(
            devices,
            ("data", "tensor"),
            axis_types=(
                jax.sharding.AxisType.Explicit,
                jax.sharding.AxisType.Explicit,
            ),
        )
    except TypeError:
        return jax.sharding.Mesh(devices, ("data", "tensor"))


def _mesh_context(mesh):
    try:
        return jax.sharding.use_mesh(mesh)
    except AttributeError:
        try:
            return jax.set_mesh(mesh)
        except AttributeError:
            return nullcontext()


@unittest.skipIf(
    torch is None
    or HFFluxSingleTransformerBlock is None
    or HFFluxTransformerBlock is None
    or jax is None,
    "torch/diffusers/jax not installed",
)
class TestFluxTransformerBlockParityDemo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mesh = _make_mesh()

    def _assert_pair_close(self, torch_pair, jax_pair, atol=1e-4, rtol=1e-4):
        self.assertEqual(len(torch_pair), len(jax_pair))
        for torch_item, jax_item in zip(torch_pair, jax_pair, strict=True):
            np.testing.assert_allclose(
                torch_item.detach().cpu().numpy(),
                np.asarray(jax_item),
                atol=atol,
                rtol=rtol,
            )

    def test_flux_single_transformer_block_parity(self):
        torch.manual_seed(0)
        dim = 128
        num_heads = 4
        head_dim = 32
        batch_size = 2
        seq_len = 128
        text_seq_len = 128

        torch_block = (
            HFFluxSingleTransformerBlock(
                dim=dim,
                num_attention_heads=num_heads,
                attention_head_dim=head_dim,
            )
            .cpu()
            .eval()
        )
        with _mesh_context(self.mesh):
            jax_block = JaxFluxSingleTransformerBlock(
                dim=dim,
                num_attention_heads=num_heads,
                attention_head_dim=head_dim,
                mesh=self.mesh,
                attention_impl="sdpa",
                params_dtype=jnp.float32,
            )
        _copy_hf_state_dict_to_jax(
            torch_block.state_dict(),
            jax_block,
            to_mappings(),
            hf_prefix="single_transformer_blocks.0",
            target_prefix="single_transformer_blocks.0",
        )

        hidden_states_torch = torch.randn(batch_size, seq_len, dim, dtype=torch.float32)
        encoder_hidden_states_torch = torch.randn(
            batch_size, text_seq_len, dim, dtype=torch.float32
        )
        temb_torch = torch.randn(batch_size, dim, dtype=torch.float32)

        hidden_states_jax = jnp.asarray(hidden_states_torch.detach().cpu().numpy())
        encoder_hidden_states_jax = jnp.asarray(encoder_hidden_states_torch.detach().cpu().numpy())
        temb_jax = jnp.asarray(temb_torch.detach().cpu().numpy())

        with torch.no_grad():
            torch_pair = torch_block(
                hidden_states=hidden_states_torch,
                encoder_hidden_states=encoder_hidden_states_torch,
                temb=temb_torch,
            )
        jax_pair = jax_block(
            hidden_states=hidden_states_jax,
            encoder_hidden_states=encoder_hidden_states_jax,
            temb=temb_jax,
        )

        self._assert_pair_close(torch_pair, jax_pair)

    def test_flux_transformer_block_parity(self):
        torch.manual_seed(0)
        dim = 128
        num_heads = 4
        head_dim = 32
        batch_size = 2
        seq_len = 128
        text_seq_len = 128

        torch_block = (
            HFFluxTransformerBlock(
                dim=dim,
                num_attention_heads=num_heads,
                attention_head_dim=head_dim,
                eps=1e-6,
            )
            .cpu()
            .eval()
        )
        with _mesh_context(self.mesh):
            jax_block = JaxFluxTransformerBlock(
                dim=dim,
                num_attention_heads=num_heads,
                attention_head_dim=head_dim,
                mesh=self.mesh,
                eps=1e-6,
                attention_impl="sdpa",
                params_dtype=jnp.float32,
            )
        _copy_hf_state_dict_to_jax(
            torch_block.state_dict(),
            jax_block,
            to_mappings(),
            hf_prefix="transformer_blocks.0",
            target_prefix="transformer_blocks.0",
        )

        hidden_states_torch = torch.randn(batch_size, seq_len, dim, dtype=torch.float32)
        encoder_hidden_states_torch = torch.randn(
            batch_size, text_seq_len, dim, dtype=torch.float32
        )
        temb_torch = torch.randn(batch_size, dim, dtype=torch.float32)

        hidden_states_jax = jnp.asarray(hidden_states_torch.detach().cpu().numpy())
        encoder_hidden_states_jax = jnp.asarray(encoder_hidden_states_torch.detach().cpu().numpy())
        temb_jax = jnp.asarray(temb_torch.detach().cpu().numpy())

        with torch.no_grad():
            torch_pair = torch_block(
                hidden_states=hidden_states_torch,
                encoder_hidden_states=encoder_hidden_states_torch,
                temb=temb_torch,
            )
        jax_pair = jax_block(
            hidden_states=hidden_states_jax,
            encoder_hidden_states=encoder_hidden_states_jax,
            temb=temb_jax,
        )

        self._assert_pair_close(torch_pair, jax_pair)


if __name__ == "__main__":
    unittest.main()
