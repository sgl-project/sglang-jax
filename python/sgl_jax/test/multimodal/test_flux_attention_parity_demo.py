import os
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
for _tpu_env in ("TPU_ACCELERATOR_TYPE", "TPU_WORKER_HOSTNAMES"):
    os.environ.pop(_tpu_env, None)

import numpy as np

try:
    import torch
    import torch.nn as tnn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    tnn = None
    F = None

try:
    import jax
    import jax.numpy as jnp
    from flax import nnx
except ImportError:  # pragma: no cover
    jax = None
    jnp = None
    nnx = None

if jax is not None:
    from sgl_jax.srt.multimodal.models.flux.FluxTransformer2DModel import (
        FluxAttention as JaxFluxAttention,
    )


class TorchFluxAttnProcessor:
    def __call__(
        self,
        attn: "TorchFluxAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb=None,
    ):
        del attention_mask, image_rotary_emb

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states).unflatten(-1, (attn.heads, -1))
            encoder_key = attn.add_k_proj(encoder_hidden_states).unflatten(-1, (attn.heads, -1))
            encoder_value = attn.add_v_proj(encoder_hidden_states).unflatten(
                -1, (attn.heads, -1)
            )

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        hidden_states = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
            context_len = encoder_hidden_states.shape[1]
            encoder_hidden_states = hidden_states[:, :context_len, :]
            hidden_states = hidden_states[:, context_len:, :]
            if not attn.pre_only:
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
            return hidden_states, encoder_hidden_states

        if not attn.pre_only:
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class TorchFluxAttention(tnn.Module):
    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        bias: bool = False,
        added_kv_proj_dim: int | None = None,
        added_proj_bias: bool | None = True,
        out_bias: bool = True,
        eps: float = 1e-6,
        out_dim: int | None = None,
        pre_only: bool = False,
        elementwise_affine: bool = True,
    ):
        super().__init__()
        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.pre_only = pre_only
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.added_kv_proj_dim = added_kv_proj_dim

        self.norm_q = tnn.RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
        self.norm_k = tnn.RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
        self.to_q = tnn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = tnn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_v = tnn.Linear(query_dim, self.inner_dim, bias=bias)

        if not self.pre_only:
            self.to_out = tnn.ModuleList(
                [
                    tnn.Linear(self.inner_dim, self.out_dim, bias=out_bias),
                    tnn.Dropout(0.0),
                ]
            )

        if added_kv_proj_dim is not None:
            self.norm_added_q = tnn.RMSNorm(dim_head, eps=eps)
            self.norm_added_k = tnn.RMSNorm(dim_head, eps=eps)
            self.add_q_proj = tnn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.add_k_proj = tnn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.add_v_proj = tnn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.to_add_out = tnn.Linear(self.inner_dim, query_dim, bias=out_bias)

        self.processor = TorchFluxAttnProcessor()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb=None,
    ):
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            image_rotary_emb,
        )


def _copy_linear_torch_to_jax(torch_linear, jax_linear):
    jax_linear.weight.value = jnp.asarray(torch_linear.weight.detach().cpu().numpy().T)
    if torch_linear.bias is not None and jax_linear.bias is not None:
        jax_linear.bias.value = jnp.asarray(torch_linear.bias.detach().cpu().numpy())


def _copy_rmsnorm_torch_to_jax(torch_norm, jax_norm):
    if getattr(torch_norm, "weight", None) is not None and jax_norm.scale is not None:
        jax_norm.scale.value = jnp.asarray(torch_norm.weight.detach().cpu().numpy())


@unittest.skipIf(torch is None or jax is None or nnx is None, "torch/jax/flax not installed")
class TestFluxAttentionParityDemo(unittest.TestCase):
    def test_flux_attention_cross_attention_cpu_parity(self):
        torch.manual_seed(0)

        batch_size = 2
        seq_len = 8
        context_len = 4
        hidden_size = 128
        num_heads = 4
        head_dim = 8

        torch_attn = TorchFluxAttention(
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
            attention_impl="naive",
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

        hidden_states_torch = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)
        encoder_hidden_states_torch = torch.randn(
            batch_size, context_len, hidden_size, dtype=torch.float32
        )

        hidden_states_jax = jnp.asarray(hidden_states_torch.detach().cpu().numpy())
        encoder_hidden_states_jax = jnp.asarray(
            encoder_hidden_states_torch.detach().cpu().numpy()
        )

        with torch.no_grad():
            torch_hidden, torch_context = torch_attn(
                hidden_states=hidden_states_torch,
                encoder_hidden_states=encoder_hidden_states_torch,
            )

        jax_hidden, jax_context = jax_attn(
            hidden_states=hidden_states_jax,
            encoder_hidden_states=encoder_hidden_states_jax,
        )

        torch_hidden_np = torch_hidden.detach().cpu().numpy()
        torch_context_np = torch_context.detach().cpu().numpy()
        jax_hidden_np = np.asarray(jax_hidden)
        jax_context_np = np.asarray(jax_context)

        np.testing.assert_allclose(torch_hidden_np, jax_hidden_np, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(torch_context_np, jax_context_np, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
