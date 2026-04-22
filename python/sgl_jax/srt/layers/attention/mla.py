"""Multi-head Latent Attention (MLA) layer."""

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from sgl_jax.srt.layers.embeddings import get_rope
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch


class MLAAttention(nnx.Module):
    """Multi-head Latent Attention (non-absorbed mode).

    Decompresses latent state into Q/K/V during forward and reuses
    RadixAttention + MHATokenToKVPool. ~43x more KV cache than absorbed
    mode; will be replaced once the MLA Pallas kernel is production-ready.

    Data flow:
      Q path:  hidden -> q_a_proj -> norm -> q_b_proj -> split(q_nope, q_rope)
      KV path: hidden -> kv_a_proj -> split(compressed, k_rope)
               compressed -> norm -> kv_b_proj -> split(k_nope, v)
      RoPE:    applied only to q_rope and k_rope
      Assembly: Q = concat(q_nope, q_rope'), K = concat(k_nope, k_rope')
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        rope_theta: float = 10000.0,
        rope_scaling: dict[str, Any] | None = None,
        rope_interleave: bool = True,
        max_position_embeddings: int = 163840,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        super().__init__()

        self.mesh = mesh
        self.num_heads = num_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank

        if q_lora_rank is None:
            self.q_proj = LinearBase(
                hidden_size,
                num_heads * self.qk_head_dim,
                mesh,
                use_bias=False,
                params_dtype=dtype,
                kernel_axes=(None, "tensor"),
                scope_name="q_proj",
            )
        else:
            self.q_a_proj = LinearBase(
                hidden_size,
                q_lora_rank,
                mesh,
                use_bias=False,
                params_dtype=dtype,
                kernel_axes=(None, None),
                scope_name="q_a_proj",
            )
            # dtype=bf16 forces RMSNorm output back to bf16 (it would otherwise
            # promote to fp32 via the fp32 scale param), so QuantizedLinear
            # downstream stays in bf16.
            self.q_a_layernorm = RMSNorm(q_lora_rank, dtype=dtype)
            self.q_b_proj = LinearBase(
                q_lora_rank,
                num_heads * self.qk_head_dim,
                mesh,
                use_bias=False,
                params_dtype=dtype,
                kernel_axes=(None, "tensor"),
                scope_name="q_b_proj",
            )

        self.kv_a_proj = LinearBase(
            hidden_size,
            kv_lora_rank + qk_rope_head_dim,
            mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=(None, None),
            scope_name="kv_a_proj",
        )
        self.kv_a_layernorm = RMSNorm(kv_lora_rank, dtype=dtype)
        self.kv_b_proj = LinearBase(
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
            mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=(None, "tensor"),
            scope_name="kv_b_proj",
        )

        self.o_proj = LinearBase(
            num_heads * v_head_dim,
            hidden_size,
            mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=("tensor", None),
            scope_name="o_proj",
        )

        self.rotary_emb = get_rope(
            head_size=qk_rope_head_dim,
            rotary_dim=qk_rope_head_dim,
            max_position=max_position_embeddings,
            base=int(rope_theta),
            is_neox_style=not rope_interleave,
            rope_scaling=rope_scaling,
            dtype=dtype,
        )

        self.attn = RadixAttention(
            num_heads=num_heads,
            head_dim=self.qk_head_dim,
            scaling=self.qk_head_dim**-0.5,
            num_kv_heads=num_heads,
            layer_id=layer_id,
        )

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> tuple[jax.Array, jax.Array]:
        if self.q_lora_rank is None:
            q, _ = self.q_proj(hidden_states)
        else:
            q_compressed, _ = self.q_a_proj(hidden_states)
            q_compressed = self.q_a_layernorm(q_compressed)
            q, _ = self.q_b_proj(q_compressed)
        q = q.reshape(-1, self.num_heads, self.qk_head_dim)
        q_nope = q[:, :, : self.qk_nope_head_dim]
        q_rope = q[:, :, self.qk_nope_head_dim :]

        kv_a_out, _ = self.kv_a_proj(hidden_states)
        compressed = kv_a_out[:, : self.kv_lora_rank]
        k_rope_raw = kv_a_out[:, self.kv_lora_rank :]

        compressed = self.kv_a_layernorm(compressed)
        kv_out, _ = self.kv_b_proj(compressed)
        kv_out = kv_out.reshape(-1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope = kv_out[:, :, : self.qk_nope_head_dim]
        v = kv_out[:, :, self.qk_nope_head_dim :]

        # Pad V to qk_head_dim to match K, required by fused MHATokenToKVPool.
        v = jnp.pad(v, ((0, 0), (0, 0), (0, self.qk_head_dim - self.v_head_dim)))

        k_rope = k_rope_raw.reshape(-1, 1, self.qk_rope_head_dim)

        q_rope, k_rope = self.rotary_emb(positions, q_rope, k_rope)
        k_rope = jnp.broadcast_to(
            k_rope,
            (k_rope.shape[0], self.num_heads, self.qk_rope_head_dim),
            out_sharding=jax.sharding.PartitionSpec(None, "tensor", None),
        )

        q = jnp.concatenate([q_nope, q_rope], axis=-1)
        k = jnp.concatenate([k_nope, k_rope], axis=-1)

        attn_output, kv_fused = self.attn(
            q,
            k,
            v,
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
        )

        # Strip V padding: o_proj expects num_heads * v_head_dim.
        attn_output = attn_output.reshape(-1, self.num_heads, self.qk_head_dim)
        attn_output = attn_output[:, :, : self.v_head_dim].reshape(
            -1, self.num_heads * self.v_head_dim
        )

        output, _ = self.o_proj(attn_output)
        return output, kv_fused
