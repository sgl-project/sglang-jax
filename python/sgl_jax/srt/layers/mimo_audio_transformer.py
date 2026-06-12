"""Neutral MiMo-Audio transformer building blocks.

These classes (``Qwen2ConfigAdapter``, ``MiMoAudioMLP``, ``MiMoAudioAttention``,
``MiMoAudioDecoderLayer``, ``MiMoAudioTransformer``) are shared between the
generation-side ``MiMoAudioForCausalLM`` (``multimodal/models/mimo_audio``) and
the understanding-side MiMo-V2.5 audio tower (``srt/models/mimo_v2_5``). They are
plain jax/flax + ``srt.layers.*`` + ``srt.models.qwen2`` and take plain kwargs
(NOT ``MiMoAudioBackboneConfig``), so they live here in ``srt.layers`` as a
neutral home: the understanding tower can import them without re-introducing an
``srt -> multimodal`` import edge.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx

from sgl_jax.srt.layers.embeddings import Embed, RotaryEmbedding
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.models.qwen2 import Qwen2DecoderLayer


@dataclass
class Qwen2ConfigAdapter:
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    max_position_embeddings: int
    rope_theta: float
    rms_norm_eps: float
    head_dim: int = None
    rope_scaling: dict = None


class MiMoAudioMLP(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_id = layer_id

        self.gate_proj = LinearBase(
            input_size=hidden_size,
            output_size=intermediate_size,
            kernel_axes=(None, "tensor"),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
        )

        self.up_proj = LinearBase(
            input_size=hidden_size,
            output_size=intermediate_size,
            kernel_axes=(None, "tensor"),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
        )

        self.down_proj = LinearBase(
            input_size=intermediate_size,
            output_size=hidden_size,
            kernel_axes=("tensor", None),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
        )

        self.act_fn = jax.nn.silu

    def __call__(self, hidden_states: jax.Array):
        a1, _ = self.gate_proj(hidden_states)
        a2, _ = self.up_proj(hidden_states)
        intermediate_parallel = a2 * self.act_fn(a1)
        output, _ = self.down_proj(intermediate_parallel)
        return output


class MiMoAudioAttention(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        use_bias: bool = True,
        use_causal_mask: bool = True,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_id = layer_id
        self.head_dim = head_dim
        self.q_head_num = num_heads
        self.kv_head_num = num_kv_heads
        self.use_causal_mask = use_causal_mask
        self.scaling = head_dim**-0.5

        self.q_proj = LinearBase(
            input_size=hidden_size,
            output_size=num_heads * head_dim,
            use_bias=use_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.k_proj = LinearBase(
            input_size=hidden_size,
            output_size=num_kv_heads * head_dim,
            use_bias=use_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.v_proj = LinearBase(
            input_size=hidden_size,
            output_size=num_kv_heads * head_dim,
            use_bias=use_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.o_proj = LinearBase(
            input_size=num_heads * head_dim,
            output_size=hidden_size,
            use_bias=False,
            kernel_axes=("tensor", None),
            params_dtype=dtype,
            mesh=mesh,
        )

        self.rotary_emb = RotaryEmbedding(
            head_size=head_dim,
            rotary_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            is_neox_style=True,
            dtype=dtype,
        )

        self.attn = RadixAttention(
            num_heads=num_heads,
            head_dim=head_dim,
            scaling=self.scaling,
            num_kv_heads=num_kv_heads,
            layer_id=layer_id,
        )

    def __call__(
        self,
        hidden_states: jax.Array,
        positions: jax.Array,
        forward_batch: ForwardBatch | None = None,
        token_to_kv_pool: KVCache | None = None,
    ) -> tuple[jax.Array, jax.Array | None]:
        """Forward pass with two branches: RadixAttention or Standard Attention.

        Branch 1 (RadixAttention): Used when token_to_kv_pool is provided (main model).
        Branch 2 (Standard): Used when token_to_kv_pool is None (Patch Encoder/Decoder).
        """
        # Branch 1: RadixAttention (for main model with KV cache)
        if token_to_kv_pool is not None:
            q, _ = self.q_proj(hidden_states)
            k, _ = self.k_proj(hidden_states)
            v, _ = self.v_proj(hidden_states)

            q = q.reshape(-1, self.q_head_num, self.head_dim)
            k = k.reshape(-1, self.kv_head_num, self.head_dim)
            v = v.reshape(-1, self.kv_head_num, self.head_dim)

            q, k = self.rotary_emb(positions, q, k)

            attn_output, kv_fused = self.attn(q, k, v, forward_batch, token_to_kv_pool)

            output, _ = self.o_proj(attn_output)
            return output, kv_fused

        # Branch 2: Standard Attention (Stateless / Patch Decoder)
        batch_size, seq_len, _ = hidden_states.shape

        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        q = q.reshape(batch_size, seq_len, self.q_head_num, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.kv_head_num, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.kv_head_num, self.head_dim)

        positions_flat = (
            positions.reshape(-1) if positions.ndim > 1 else jnp.tile(positions, batch_size)
        )
        q_flat = q.reshape(-1, self.q_head_num, self.head_dim)
        k_flat = k.reshape(-1, self.kv_head_num, self.head_dim)
        q_rot, k_rot = self.rotary_emb(positions_flat, q_flat, k_flat)
        q = q_rot.reshape(batch_size, seq_len, self.q_head_num, self.head_dim)
        k = k_rot.reshape(batch_size, seq_len, self.kv_head_num, self.head_dim)

        if self.kv_head_num < self.q_head_num:
            k = jnp.repeat(k, self.q_head_num // self.kv_head_num, axis=2)
            v = jnp.repeat(v, self.q_head_num // self.kv_head_num, axis=2)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        attn_weights = jnp.einsum("bhqd,bhkd->bhqk", q, k) * self.scaling

        if self.use_causal_mask:
            causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
            attn_weights = jnp.where(causal_mask[None, None, :, :], attn_weights, -jnp.inf)

        attn_weights = jax.nn.softmax(attn_weights, axis=-1).astype(v.dtype)
        attn_output = jnp.einsum("bhqk,bhkd->bhqd", attn_weights, v)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)

        output, _ = self.o_proj(attn_output)
        return output, None


class MiMoAudioDecoderLayer(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        max_position_embeddings: int,
        rope_theta: float,
        rms_norm_eps: float,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        use_bias: bool = True,
        use_causal_mask: bool = True,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.self_attn = MiMoAudioAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            mesh=mesh,
            layer_id=layer_id,
            use_bias=use_bias,
            use_causal_mask=use_causal_mask,
            dtype=dtype,
        )
        self.mlp = MiMoAudioMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            mesh=mesh,
            layer_id=layer_id,
            dtype=dtype,
        )
        self.input_layernorm = RMSNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
        )

    def __call__(
        self,
        hidden_states: jax.Array,
        positions: jax.Array,
        forward_batch: ForwardBatch | None = None,
        token_to_kv_pool: KVCache | None = None,
        residual: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array | None, list]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states += residual
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        attn_output, kv_fused = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
        )

        hidden_states = attn_output + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states)

        return mlp_output, residual, kv_fused, []


class MiMoAudioTransformer(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        max_position_embeddings: int,
        rope_theta: float,
        rms_norm_eps: float,
        vocab_size: int,
        mesh: jax.sharding.Mesh,
        use_bias: bool = True,
        use_causal_mask: bool = True,
        has_embedder: bool = True,
        use_qwen2_layers: bool = False,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.has_embedder = has_embedder
        self.hidden_size = hidden_size
        self.use_qwen2_layers = use_qwen2_layers

        if has_embedder:
            self.embed_tokens = Embed(
                num_embeddings=vocab_size,
                features=hidden_size,
                dtype=dtype,
                kernel_axes=("tensor", None),
                param_dtype=dtype,
                mesh=mesh,
            )

        if use_qwen2_layers:
            config = Qwen2ConfigAdapter(
                hidden_size=hidden_size,
                num_attention_heads=num_heads,
                num_key_value_heads=num_kv_heads,
                intermediate_size=intermediate_size,
                max_position_embeddings=max_position_embeddings,
                rope_theta=rope_theta,
                rms_norm_eps=rms_norm_eps,
                head_dim=head_dim,
            )
            self.layers = nnx.List(
                [
                    Qwen2DecoderLayer(
                        config=config,
                        mesh=mesh,
                        layer_id=i,
                        dtype=dtype,
                    )
                    for i in range(num_layers)
                ]
            )
        else:
            self.layers = nnx.List(
                [
                    MiMoAudioDecoderLayer(
                        hidden_size=hidden_size,
                        num_heads=num_heads,
                        num_kv_heads=num_kv_heads,
                        head_dim=head_dim,
                        intermediate_size=intermediate_size,
                        max_position_embeddings=max_position_embeddings,
                        rope_theta=rope_theta,
                        rms_norm_eps=rms_norm_eps,
                        mesh=mesh,
                        layer_id=i,
                        use_bias=use_bias,
                        use_causal_mask=use_causal_mask,
                        dtype=dtype,
                    )
                    for i in range(num_layers)
                ]
            )

        self.norm = RMSNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
        )

    def __call__(
        self,
        input_ids_or_embeds: jax.Array,
        positions: jax.Array,
        forward_batch: ForwardBatch | None = None,
        token_to_kv_pool: KVCache | None = None,
    ) -> tuple[jax.Array, list, list]:
        if self.has_embedder and jnp.issubdtype(input_ids_or_embeds.dtype, jnp.integer):
            hidden_states = self.embed_tokens(input_ids_or_embeds)
        else:
            hidden_states = input_ids_or_embeds

        residual = None
        layers_kv_fused = []
        layers_callback_flag = []

        for layer_idx, layer in enumerate(self.layers):
            if self.use_qwen2_layers:
                hidden_states, residual, kv_fused, callback_flag = layer(
                    positions,
                    hidden_states,
                    forward_batch,
                    token_to_kv_pool,
                    residual,
                )
            else:
                hidden_states, residual, kv_fused, callback_flag = layer(
                    hidden_states,
                    positions,
                    forward_batch,
                    token_to_kv_pool,
                    residual,
                )
            layers_kv_fused.append(kv_fused)
            layers_callback_flag.extend(callback_flag)

        if residual is not None:
            hidden_states += residual
        hidden_states = self.norm(hidden_states)

        return hidden_states, layers_kv_fused, layers_callback_flag
