"""Inference-only Step 3.5 Flash model skeleton.

Architecture: hybrid sliding/full-attention GQA + sigmoid-gated MoE.
``architectures=["Step3p5ForCausalLM"]``, ``model_type="step3p5"``.

This file registers the class in the sgl-jax model registry via ``EntryClass``.
Decoder layer internals, forward pass, and weight loading are implemented in
follow-up commits; this skeleton is the minimal instantiable stub needed to
wire the registry, config plumbing, and ``patch_model_config``.

Reference: HF modeling_step3p5.py / configuration_step3p5.py
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

# Head dimension for Step 3.5 Flash (fixed across all layers)
_HEAD_DIM = 128


class Step3p5Attention(nnx.Module):
    """Step 3.5 Flash attention with per-head QK-norm, per-layer partial RoPE,
    head-wise gate, and optional sliding-window attention.

    Two layer variants:
    - ``full_attention``:    64 Q-heads, rope_theta=5e6, partial_rotary=0.5, llama3 scaling
    - ``sliding_attention``: 96 Q-heads, rope_theta=1e4, partial_rotary=1.0, no scaling

    Both use 8 KV-heads (num_attention_groups=8) and head_dim=128.
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

        # Head counts differ by layer type
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

        # --- Projections ---
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

        # --- Per-head QK-norm (zero-centered, GemmaRMSNorm with add_unit_offset=True) ---
        # Norm is applied BEFORE RoPE. Weight shape [head_dim], shared across all heads.
        self.q_norm = GemmaRMSNorm(
            _HEAD_DIM,
            epsilon=config.rms_norm_eps,
            add_unit_offset=True,
        )
        self.k_norm = GemmaRMSNorm(
            _HEAD_DIM,
            epsilon=config.rms_norm_eps,
            add_unit_offset=True,
        )

        # --- Per-layer RoPE ---
        rope_theta_i: float
        if isinstance(config.rope_theta, list):
            rope_theta_i = float(config.rope_theta[layer_id])
        else:
            rope_theta_i = float(config.rope_theta)

        partial_rotary_factor_i: float
        if config.partial_rotary_factors is not None:
            partial_rotary_factor_i = float(config.partial_rotary_factors[layer_id])
        else:
            partial_rotary_factor_i = 1.0

        # Only full_attention layers (in yarn_only_types) get llama3 scaling
        yarn_only_types = config.yarn_only_types or []
        rope_scaling_i = config.rope_scaling if layer_type in yarn_only_types else None

        self.rotary_emb = get_rope(
            head_size=_HEAD_DIM,
            rotary_dim=_HEAD_DIM,
            max_position=config.max_position_embeddings,
            base=rope_theta_i,  # type: ignore[arg-type]  # float accepted despite int annotation
            rope_scaling=rope_scaling_i,
            partial_rotary_factor=partial_rotary_factor_i,
            is_neox_style=True,
            dtype=dtype,
        )

        # --- Head-wise gate (data-dependent sink) ---
        # g_proj maps hidden_size -> num_heads; gate is applied BEFORE o_proj.
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

        # --- Sliding-window attention ---
        # gemma2 passes config.sliding_window directly; mirror that convention.
        # RadixAttention treats sliding_window_size=0 as full attention.
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
        """Forward pass.

        Args:
            positions: ``[num_tokens]`` token position indices.
            hidden_states: ``[num_tokens, hidden_size]`` post-input-layernorm activations.
            forward_batch: Batch metadata for the attention backend.
            token_to_kv_pool: KV cache pool.

        Returns:
            ``(output, kv_fused)`` matching qwen3/gemma2 attention convention.
        """
        # Capture the attention INPUT for the head-wise gate (HF reference line 489)
        if self.use_head_wise_attn_gate:
            gate_states, _ = self.g_proj(hidden_states)  # [T, num_heads]

        # Project
        q, _ = self.q_proj(hidden_states)  # [T, num_heads * head_dim]
        k, _ = self.k_proj(hidden_states)  # [T, num_kv_heads * head_dim]
        v, _ = self.v_proj(hidden_states)  # [T, num_kv_heads * head_dim]

        # Reshape to [..., num_heads, head_dim] for per-head norm
        q = q.reshape(-1, self.num_heads, _HEAD_DIM)
        k = k.reshape(-1, self.num_kv_heads, _HEAD_DIM)

        # Per-head QK-norm BEFORE RoPE (sglang reference forward_prepare_native)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE (after QK-norm)
        q, k = self.rotary_emb(positions, q, k)

        # Reshape v for the attention backend
        v = v.reshape(-1, self.num_kv_heads, _HEAD_DIM)

        # Attention
        attn_output, kv_fused = self.attn(q, k, v, forward_batch, token_to_kv_pool)
        # attn_output: [T, num_heads * head_dim]

        # Head-wise gate BEFORE o_proj (HF reference lines 528-531)
        if self.use_head_wise_attn_gate:
            # gate_states: [T, num_heads] -> sigmoid -> [T, num_heads, 1]
            gate = jax.nn.sigmoid(gate_states)  # [T, num_heads]
            out = attn_output.reshape(-1, self.num_heads, _HEAD_DIM)
            out = out * gate[..., None]  # [T, num_heads, head_dim]
            attn_output = out.reshape(-1, self.num_heads * _HEAD_DIM)

        output, _ = self.o_proj(attn_output)
        return output, kv_fused


class Step3p5ForCausalLM(nnx.Module):
    """Step 3.5 Flash causal LM — registry skeleton.

    Only ``embed_tokens``, ``lm_head``, and ``logits_processor`` are
    built here. Decoder layers (``self.model``) are left as ``None``
    until the attention/MoE implementation lands.
    """

    @classmethod
    def patch_model_config(cls, mc: ModelConfig) -> None:
        """Set head_dim=128 on the ModelConfig.

        Step 3.5 Flash always uses 128-dim heads for both full-attention
        and sliding-attention layers. The HF config carries ``head_dim``
        already, but we pin it here so the KV pool and attention backend
        receive the correct value regardless of how the config was loaded.
        """
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

        # Embed + untied lm_head (tie_word_embeddings=False in HF config).
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

        # Decoder layers — placeholder until the decoder implementation lands.
        self.model = None  # TODO(step3p5): build Step3p5Model here

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
