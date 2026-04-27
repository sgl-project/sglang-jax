from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from sgl_jax.srt.layers.attention.fla.gated_rmsnorm import GatedRMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.radix_linear_attention import RadixLinearAttention
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch


class KimiDeltaAttention(nnx.Module):
    """Standalone Kimi Delta Attention layer.

    This intentionally implements only the KDA attention module. Full
    KimiDecoderLayer/KimiLinearModel assembly is handled by the model skeleton
    integration work.
    """

    def __init__(
        self,
        config,
        layer_idx: int = 0,
        mesh: jax.sharding.Mesh | None = None,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.mesh = mesh

        linear_config = config.linear_attn_config
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.conv_size = linear_config["short_conv_kernel_size"]
        self.head_dim = linear_config["head_dim"]
        self.k_head_dim = self.head_dim
        self.v_head_dim = getattr(config, "v_head_dim", None) or self.head_dim
        self.num_heads = linear_config["num_heads"]
        self.num_k_heads = self.num_heads
        self.num_v_heads = self.num_heads
        self.projection_k_size = self.num_k_heads * self.k_head_dim
        self.projection_size = self.num_heads * self.head_dim
        self.rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)

        self.q_proj = LinearBase(
            self.hidden_size, self.projection_k_size, mesh,
            use_bias=False, params_dtype=dtype,
            kernel_axes=(None, "tensor"), scope_name="q_proj",
        )
        self.k_proj = LinearBase(
            self.hidden_size, self.projection_k_size, mesh,
            use_bias=False, params_dtype=dtype,
            kernel_axes=(None, "tensor"), scope_name="k_proj",
        )
        self.v_proj = LinearBase(
            self.hidden_size, self.projection_size, mesh,
            use_bias=False, params_dtype=dtype,
            kernel_axes=(None, "tensor"), scope_name="v_proj",
        )

        self.q_conv1d = LinearBase(
            self.conv_size, self.projection_k_size, mesh,
            use_bias=False, params_dtype=jnp.float32,
            kernel_axes=(None, "tensor"), scope_name="q_conv1d",
        )
        self.k_conv1d = LinearBase(
            self.conv_size, self.projection_k_size, mesh,
            use_bias=False, params_dtype=jnp.float32,
            kernel_axes=(None, "tensor"), scope_name="k_conv1d",
        )
        self.v_conv1d = LinearBase(
            self.conv_size, self.projection_size, mesh,
            use_bias=False, params_dtype=jnp.float32,
            kernel_axes=(None, "tensor"), scope_name="v_conv1d",
        )

        self.A_log = nnx.Param(jnp.zeros((1, 1, self.num_heads, 1), dtype=jnp.float32))
        self.dt_bias = nnx.Param(jnp.zeros((self.projection_size,), dtype=jnp.float32))

        self.f_a_proj = LinearBase(
            self.hidden_size, self.head_dim, mesh,
            use_bias=False, params_dtype=dtype,
            kernel_axes=(None, None), scope_name="f_a_proj",
        )
        self.f_b_proj = LinearBase(
            self.head_dim, self.projection_size, mesh,
            use_bias=False, params_dtype=dtype,
            kernel_axes=(None, "tensor"), scope_name="f_b_proj",
        )
        self.b_proj = LinearBase(
            self.hidden_size, self.num_heads, mesh,
            use_bias=False, params_dtype=dtype,
            kernel_axes=(None, "tensor"), scope_name="b_proj",
        )
        self.g_a_proj = LinearBase(
            self.hidden_size, self.head_dim, mesh,
            use_bias=False, params_dtype=dtype,
            kernel_axes=(None, None), scope_name="g_a_proj",
        )
        self.g_b_proj = LinearBase(
            self.head_dim, self.projection_size, mesh,
            use_bias=False, params_dtype=dtype,
            kernel_axes=(None, "tensor"), scope_name="g_b_proj",
        )
        self.o_norm = GatedRMSNorm(self.head_dim, epsilon=self.rms_norm_eps)
        self.o_proj = LinearBase(
            self.projection_size, self.hidden_size, mesh,
            use_bias=False, params_dtype=dtype,
            kernel_axes=("tensor", None), scope_name="o_proj",
        )

        self.attn = RadixLinearAttention(
            layer_id=self.layer_idx,
            num_q_heads=self.num_k_heads,
            num_k_heads=self.num_k_heads,
            num_v_heads=self.num_v_heads,
            head_q_dim=self.k_head_dim,
            head_k_dim=self.k_head_dim,
            head_v_dim=self.v_head_dim,
            conv_weights=(
                self.q_conv1d.weight[...],
                self.k_conv1d.weight[...],
                self.v_conv1d.weight[...],
            ),
            bias=None,
            activation=jax.nn.silu,
            A_log=self.A_log[...],
            dt_bias=self.dt_bias[...],
        )

    def __call__(
        self,
        positions: jax.Array | None,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        recurrent_state_pool,
    ) -> tuple[jax.Array, object]:
        del positions

        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        raw_gate, _ = self.f_b_proj(self.f_a_proj(hidden_states)[0])
        beta = jax.nn.sigmoid(self.b_proj(hidden_states)[0].astype(jnp.float32))

        o, recurrent_state_pool = self.attn(
            forward_batch,
            q,
            k,
            v,
            raw_gate,
            beta,
            recurrent_state_pool,
        )
        o = o.reshape(hidden_states.shape[0], self.num_heads, self.head_dim)

        g_a, _ = self.g_a_proj(hidden_states)
        output_gate, _ = self.g_b_proj(g_a)
        output_gate = output_gate.reshape(hidden_states.shape[0], self.num_heads, self.head_dim)
        o = self.o_norm(o, output_gate).reshape(hidden_states.shape[0], self.projection_size)
        o, _ = self.o_proj(o)

        return o, recurrent_state_pool


__all__ = ["KimiDeltaAttention"]
