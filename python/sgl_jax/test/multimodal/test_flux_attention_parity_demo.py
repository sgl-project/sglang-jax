from __future__ import annotations

import os
import unittest

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
for _tpu_env in ("TPU_ACCELERATOR_TYPE", "TPU_WORKER_HOSTNAMES"):
    os.environ.pop(_tpu_env, None)

try:
    import torch
    from diffusers.models.transformers.transformer_flux import (
        FluxAttention as HFFluxAttention,
    )
except ImportError:  # pragma: no cover
    torch = None
    HFFluxAttention = None

try:
    import jax
    import jax.numpy as jnp
    from flax import nnx
except ImportError:  # pragma: no cover
    jax = None
    jnp = None
    nnx = None

if jax is not None:
    from sgl_jax.srt.multimodal.models.flux.fluxtransformer2dmodel import (
        FluxAttention as JaxFluxAttention,
    )


def _copy_linear_torch_to_jax(torch_linear, jax_linear):
    jax_linear.weight.value = jnp.asarray(torch_linear.weight.detach().cpu().numpy().T)
    if torch_linear.bias is not None and jax_linear.bias is not None:
        jax_linear.bias.value = jnp.asarray(torch_linear.bias.detach().cpu().numpy())


def _copy_rmsnorm_torch_to_jax(torch_norm, jax_norm):
    if getattr(torch_norm, "weight", None) is not None and jax_norm.scale is not None:
        jax_norm.scale.value = jnp.asarray(torch_norm.weight.detach().cpu().numpy())


@unittest.skipIf(
    torch is None or HFFluxAttention is None or jax is None or nnx is None,
    "torch/diffusers/jax/flax not installed",
)
class TestFluxAttentionParityDemo(unittest.TestCase):
    def _build_attention_pair(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
    ):
        torch_attn = HFFluxAttention(
            query_dim=hidden_size,
            heads=num_heads,
            dim_head=head_dim,
            bias=True,
            added_kv_proj_dim=hidden_size,
            out_dim=hidden_size,
            out_bias=True,
            pre_only=False,
            elementwise_affine=True,
        ).cpu()
        torch_attn.eval()

        jax_attn = JaxFluxAttention(
            query_dim=hidden_size,
            heads=num_heads,
            dim_head=head_dim,
            bias=True,
            added_kv_proj_dim=hidden_size,
            out_dim=hidden_size,
            out_bias=True,
            pre_only=False,
            elementwise_affine=True,
            attention_impl="sdpa",
            params_dtype=jnp.float32,
        )

        _copy_linear_torch_to_jax(torch_attn.to_q, jax_attn.to_q)
        _copy_linear_torch_to_jax(torch_attn.to_k, jax_attn.to_k)
        _copy_linear_torch_to_jax(torch_attn.to_v, jax_attn.to_v)
        _copy_linear_torch_to_jax(torch_attn.to_out[0], jax_attn.to_out[0])

        _copy_linear_torch_to_jax(torch_attn.add_q_proj, jax_attn.add_q_proj)
        _copy_linear_torch_to_jax(torch_attn.add_k_proj, jax_attn.add_k_proj)
        _copy_linear_torch_to_jax(torch_attn.add_v_proj, jax_attn.add_v_proj)
        _copy_linear_torch_to_jax(torch_attn.to_add_out, jax_attn.to_add_out)

        _copy_rmsnorm_torch_to_jax(torch_attn.norm_q, jax_attn.norm_q)
        _copy_rmsnorm_torch_to_jax(torch_attn.norm_k, jax_attn.norm_k)
        _copy_rmsnorm_torch_to_jax(torch_attn.norm_added_q, jax_attn.norm_added_q)
        _copy_rmsnorm_torch_to_jax(torch_attn.norm_added_k, jax_attn.norm_added_k)
        return torch_attn, jax_attn

    def _assert_outputs_close(self, torch_hidden, torch_context, jax_hidden, jax_context):
        torch_hidden_np = torch_hidden.detach().cpu().numpy()
        torch_context_np = torch_context.detach().cpu().numpy()
        jax_hidden_np = np.asarray(jax_hidden)
        jax_context_np = np.asarray(jax_context)

        np.testing.assert_allclose(torch_hidden_np, jax_hidden_np, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(torch_context_np, jax_context_np, rtol=1e-4, atol=1e-4)

    def _run_case(
        self,
        *,
        attention_mask_torch: torch.Tensor | None = None,
        image_rotary_emb_torch: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        batch_size, seq_len, context_len = 2, 8, 4
        hidden_size, num_heads, head_dim = 128, 4, 8
        torch_attn, jax_attn = self._build_attention_pair(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
        )

        hidden_states_torch = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)
        encoder_hidden_states_torch = torch.randn(
            batch_size, context_len, hidden_size, dtype=torch.float32
        )

        hidden_states_jax = jnp.asarray(hidden_states_torch.detach().cpu().numpy())
        encoder_hidden_states_jax = jnp.asarray(encoder_hidden_states_torch.detach().cpu().numpy())
        attention_mask_jax = None
        if attention_mask_torch is not None:
            attention_mask_jax = jnp.asarray(attention_mask_torch.detach().cpu().numpy())
        image_rotary_emb_jax = None
        if image_rotary_emb_torch is not None:
            image_rotary_emb_jax = tuple(
                jnp.asarray(x.detach().cpu().numpy()) for x in image_rotary_emb_torch
            )

        with torch.no_grad():
            torch_hidden, torch_context = torch_attn(
                hidden_states=hidden_states_torch,
                encoder_hidden_states=encoder_hidden_states_torch,
                attention_mask=attention_mask_torch,
                image_rotary_emb=image_rotary_emb_torch,
            )

        jax_hidden, jax_context = jax_attn(
            hidden_states=hidden_states_jax,
            encoder_hidden_states=encoder_hidden_states_jax,
            attention_mask=attention_mask_jax,
            image_rotary_emb=image_rotary_emb_jax,
        )

        self._assert_outputs_close(torch_hidden, torch_context, jax_hidden, jax_context)

    def test_flux_attention_cross_attention_cpu_parity(self):
        torch.manual_seed(0)
        self._run_case()

    def test_flux_attention_cross_attention_mask_cpu_parity(self):
        torch.manual_seed(0)
        batch_size = 2
        total_kv_len = 12
        attention_mask_torch = torch.zeros((batch_size, 1, 1, total_kv_len), dtype=torch.float32)
        attention_mask_torch[:, :, :, -2:] = -1.0e4
        self._run_case(attention_mask_torch=attention_mask_torch)

    def test_flux_attention_cross_attention_rope_cpu_parity(self):
        torch.manual_seed(0)
        head_dim = 8
        total_kv_len = 12
        cos_torch = torch.randn(total_kv_len, head_dim, dtype=torch.float32)
        sin_torch = torch.randn(total_kv_len, head_dim, dtype=torch.float32)
        image_rotary_emb_torch = (cos_torch, sin_torch)
        self._run_case(image_rotary_emb_torch=image_rotary_emb_torch)


if __name__ == "__main__":
    unittest.main()
