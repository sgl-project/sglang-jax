"""Inference-only Step 3.5 Flash model.

Hybrid sliding/full-attention GQA + sigmoid-gated MoE. ``model_type="step3p5"``.
Reference: HF modeling_step3p5.py / configuration_step3p5.py.
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
from flax import nnx
from transformers import PretrainedConfig

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead, get_rope
from sgl_jax.srt.layers.layernorm import GemmaRMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.mem_cache.memory_pool import KVCache, MemoryPools
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

_HEAD_DIM = 128


class Step3p5Attention(nnx.Module):
    """Per-head QK-norm + per-layer partial RoPE + head-wise gate + optional SWA.

    full: 64 Q-heads, theta 5e6, partial 0.5, llama3 scaling. sliding: 96 Q-heads,
    theta 1e4, partial 1.0, no scaling. Both 8 KV-heads, head_dim 128.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        self.layer_id = layer_id
        self.dtype = dtype
        self.head_dim = _HEAD_DIM

        layer_types = config.layer_types or []
        layer_type = layer_types[layer_id] if layer_id < len(layer_types) else "full_attention"
        self._layer_type = layer_type

        is_sliding = layer_type == "sliding_attention"

        if is_sliding and config.attention_other_setting is not None:
            self.num_heads = config.attention_other_setting["num_attention_heads"]
            self.num_kv_heads = config.attention_other_setting.get(
                "num_attention_groups", config.num_attention_groups
            )
        else:
            self.num_heads = config.num_attention_heads
            self.num_kv_heads = config.num_attention_groups

        self.q_size = self.num_heads * _HEAD_DIM
        self.kv_size = self.num_kv_heads * _HEAD_DIM
        self.scaling = _HEAD_DIM**-0.5

        self.q_proj = LinearBase(
            input_size=config.hidden_size,
            output_size=self.q_size,
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="q_proj",
        )
        self.k_proj = LinearBase(
            input_size=config.hidden_size,
            output_size=self.kv_size,
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="k_proj",
        )
        self.v_proj = LinearBase(
            input_size=config.hidden_size,
            output_size=self.kv_size,
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="v_proj",
        )
        self.o_proj = LinearBase(
            input_size=self.q_size,
            output_size=config.hidden_size,
            use_bias=False,
            kernel_axes=("tensor", None),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="o_proj",
        )

        # Per-head zero-centered QK-norm (weight [head_dim], shared across heads), applied before RoPE.
        self.q_norm = GemmaRMSNorm(_HEAD_DIM, epsilon=config.rms_norm_eps, add_unit_offset=True)
        self.k_norm = GemmaRMSNorm(_HEAD_DIM, epsilon=config.rms_norm_eps, add_unit_offset=True)

        if isinstance(config.rope_theta, list):
            rope_theta_i = float(config.rope_theta[layer_id])
        else:
            rope_theta_i = float(config.rope_theta)

        if config.partial_rotary_factors is not None:
            partial_rotary_factor_i = float(config.partial_rotary_factors[layer_id])
        else:
            partial_rotary_factor_i = 1.0

        # llama3 scaling only on yarn_only_types (full_attention) layers.
        yarn_only_types = config.yarn_only_types or []
        rope_scaling_i = config.rope_scaling if layer_type in yarn_only_types else None

        self.rotary_emb = get_rope(
            head_size=_HEAD_DIM,
            rotary_dim=_HEAD_DIM,
            max_position=config.max_position_embeddings,
            base=rope_theta_i,  # type: ignore[arg-type]
            rope_scaling=rope_scaling_i,
            partial_rotary_factor=partial_rotary_factor_i,
            is_neox_style=True,
            dtype=dtype,
        )

        # Head-wise gate: g_proj(hidden) -> per-head sigmoid scalar, applied before o_proj.
        self.use_head_wise_attn_gate: bool = bool(getattr(config, "use_head_wise_attn_gate", True))
        if self.use_head_wise_attn_gate:
            self.g_proj = LinearBase(
                input_size=config.hidden_size,
                output_size=self.num_heads,
                use_bias=False,
                kernel_axes=(None, "tensor"),
                params_dtype=dtype,
                mesh=mesh,
                scope_name="g_proj",
            )

        # sliding_window_size=0 means full attention (gemma2 convention).
        sliding_window_size = config.sliding_window if is_sliding else 0
        self.attn = RadixAttention(
            num_heads=self.num_heads,
            head_dim=_HEAD_DIM,
            scaling=self.scaling,
            num_kv_heads=self.num_kv_heads,
            sliding_window_size=sliding_window_size,
            layer_id=layer_id,
        )

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> tuple[jax.Array, jax.Array]:
        # Gate reads the attention INPUT (HF ref line 489), not the attn output.
        if self.use_head_wise_attn_gate:
            gate_states, _ = self.g_proj(hidden_states)

        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        q = q.reshape(-1, self.num_heads, _HEAD_DIM)
        k = k.reshape(-1, self.num_kv_heads, _HEAD_DIM)

        # QK-norm before RoPE.
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)

        v = v.reshape(-1, self.num_kv_heads, _HEAD_DIM)
        attn_output, kv_fused = self.attn(q, k, v, forward_batch, token_to_kv_pool)

        # Per-head gate before o_proj (HF ref lines 528-531).
        if self.use_head_wise_attn_gate:
            gate = jax.nn.sigmoid(gate_states)
            out = attn_output.reshape(-1, self.num_heads, _HEAD_DIM) * gate[..., None]
            attn_output = out.reshape(-1, self.num_heads * _HEAD_DIM)

        output, _ = self.o_proj(attn_output)
        return output, kv_fused


class Step3p5ForCausalLM(nnx.Module):
    """Step 3.5 Flash causal LM. Decoder layers (``self.model``) land in a follow-up commit."""

    @classmethod
    def patch_model_config(cls, mc: ModelConfig) -> None:
        # Pin head_dim=128 for KV pool / attention backend sizing.
        mc.head_dim = 128

    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        self.config = config
        self.mesh = mesh
        self.dtype = dtype
        logger.info("Step3p5ForCausalLM dtype=%s", dtype)

        # Untied lm_head (tie_word_embeddings=False).
        self.embed_tokens = Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=dtype,
            param_dtype=dtype,
            kernel_axes=("tensor", None),
            mesh=mesh,
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            dtype=dtype,
            param_dtype=dtype,
            kernel_axes=("tensor", None),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size, mesh=mesh)

        self.model = None  # TODO(step3p5): build Step3p5Model

    def __call__(
        self,
        forward_batch: ForwardBatch,
        memory_pools: MemoryPools,
        logits_metadata: LogitsMetadata,
    ):
        raise NotImplementedError("TODO(step3p5): forward pass not yet implemented")

    def load_weights(self, model_config: ModelConfig) -> None:
        raise NotImplementedError("TODO(step3p5): weight loading not yet implemented")


EntryClass = [Step3p5ForCausalLM]
