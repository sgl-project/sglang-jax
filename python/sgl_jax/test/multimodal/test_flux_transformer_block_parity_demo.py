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


def _copy_linear_torch_to_jax(torch_linear, jax_linear):
    jax_linear.weight[...] = jnp.asarray(torch_linear.weight.detach().cpu().numpy().T)
    if torch_linear.bias is not None and jax_linear.bias is not None:
        jax_linear.bias[...] = jnp.asarray(torch_linear.bias.detach().cpu().numpy())


def _copy_rmsnorm_torch_to_jax(torch_norm, jax_norm):
    if (
        getattr(torch_norm, "weight", None) is not None
        and getattr(jax_norm, "scale", None) is not None
    ):
        jax_norm.scale[...] = jnp.asarray(torch_norm.weight.detach().cpu().numpy())


def _copy_flux_attention_torch_to_jax(torch_attn, jax_attn):
    _copy_linear_torch_to_jax(torch_attn.to_q, jax_attn.to_q)
    _copy_linear_torch_to_jax(torch_attn.to_k, jax_attn.to_k)
    _copy_linear_torch_to_jax(torch_attn.to_v, jax_attn.to_v)
    _copy_rmsnorm_torch_to_jax(torch_attn.norm_q, jax_attn.norm_q)
    _copy_rmsnorm_torch_to_jax(torch_attn.norm_k, jax_attn.norm_k)

    if hasattr(torch_attn, "to_out") and hasattr(jax_attn, "to_out"):
        _copy_linear_torch_to_jax(torch_attn.to_out[0], jax_attn.to_out[0])

    if getattr(torch_attn, "added_kv_proj_dim", None) is not None:
        _copy_linear_torch_to_jax(torch_attn.add_q_proj, jax_attn.add_q_proj)
        _copy_linear_torch_to_jax(torch_attn.add_k_proj, jax_attn.add_k_proj)
        _copy_linear_torch_to_jax(torch_attn.add_v_proj, jax_attn.add_v_proj)
        _copy_linear_torch_to_jax(torch_attn.to_add_out, jax_attn.to_add_out)
        _copy_rmsnorm_torch_to_jax(torch_attn.norm_added_q, jax_attn.norm_added_q)
        _copy_rmsnorm_torch_to_jax(torch_attn.norm_added_k, jax_attn.norm_added_k)


def _copy_flux_feedforward_torch_to_jax(torch_ff, jax_ff):
    _copy_linear_torch_to_jax(torch_ff.net[0].proj, jax_ff.net[0])
    _copy_linear_torch_to_jax(torch_ff.net[2], jax_ff.net[2])


def _copy_adaln_zero_single_torch_to_jax(torch_norm, jax_norm):
    _copy_linear_torch_to_jax(torch_norm.linear, jax_norm.linear)


def _copy_adaln_zero_torch_to_jax(torch_norm, jax_norm):
    _copy_linear_torch_to_jax(torch_norm.linear, jax_norm.linear)


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

        _copy_adaln_zero_single_torch_to_jax(torch_block.norm, jax_block.norm)
        _copy_linear_torch_to_jax(torch_block.proj_mlp, jax_block.proj_mlp)
        _copy_linear_torch_to_jax(torch_block.proj_out, jax_block.proj_out)
        _copy_flux_attention_torch_to_jax(torch_block.attn, jax_block.attn)

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

        _copy_adaln_zero_torch_to_jax(torch_block.norm1, jax_block.norm1)
        _copy_adaln_zero_torch_to_jax(torch_block.norm1_context, jax_block.norm1_context)
        _copy_flux_attention_torch_to_jax(torch_block.attn, jax_block.attn)
        _copy_flux_feedforward_torch_to_jax(torch_block.ff, jax_block.ff)
        _copy_flux_feedforward_torch_to_jax(torch_block.ff_context, jax_block.ff_context)

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
